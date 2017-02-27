//
// Created by ollie on 2/16/17.
//

#ifndef FLECSI_GAME_OF_LIFE_H
#define FLECSI_GAME_OF_LIFE_H

#include <cinchtest.h>
#include <numeric>
#include <mpi.h>

#include "flecsi/partition/weaver.h"
#include "flecsi/topology/index_space.h"
#include "flecsi/topology/graph_utils.h"
#include "flecsi/execution/execution.h"
#include "flecsi/data/data.h"

template<typename T>
using accessor_t = flecsi::data::serial::sparse_accessor_t<T, flecsi::data::serial_meta_data_t<flecsi::default_user_meta_data_t> >;

void stencil_task(flecsi::io::simple_definition_t& sd,
                  const std::set<size_t>& primary_cells,
                  accessor_t<int>& a0,
                  accessor_t<int>& a1)
{
  // This should be replace with some kind of "task"
  for (auto cell : primary_cells) {
    std::set <size_t> me{cell};
    // FIXME: connectivity information should be part of distributed data as well.
    auto closure = flecsi::topology::entity_closure<2, 2, 0>(sd, me);
    auto nearest_neighbors = flecsi::utils::set_difference(closure, me);

    int count = 0;
    for (auto neighbor : nearest_neighbors) {
      if (neighbor != -1 && a0(neighbor, 0) == 1) {
        count++;
      }
    }
    if (a0(cell, 0)) {
      if (count < 2)
        a1(cell, 0) = 0;
      else if (count <= 3)
        a1(cell, 0) = 1;
      else
        a1(cell, 0) = 0;
    } else {
      if (count == 3)
        a1(cell, 0) = 1;
    }
  }
}

void halo_exchange_task(const flecsi::io::simple_definition_t& sd,
                        std::set<flecsi::dmp::entry_info_t>& shared_cells,
                        std::set<flecsi::dmp::entry_info_t>& ghost_cells,
                        accessor_t<int>& acc)
{
  std::vector <MPI_Request> requests;

  // Post receive for ghost cells
  for (auto ghost : ghost_cells) {
    requests.push_back({});
    MPI_Irecv(&acc(ghost.id, 0), 1, MPI_INT,
              ghost.rank, 0, MPI_COMM_WORLD, &requests[requests.size() - 1]);
  }

  // Send shared cells
  for (auto shared : shared_cells) {
    for (auto dest: shared.shared) {
      MPI_Send(&acc(shared.id, 0), 1, MPI_INT,
               dest, 0, MPI_COMM_WORLD);
    }
  }

  std::vector <MPI_Status> status(requests.size());
  MPI_Waitall(requests.size(), &requests[0], &status[0]);
}

register_task(stencil_task, loc, single);
register_task(halo_exchange_task, loc, single);

enum mesh_index_spaces_t : size_t {
  vertices,
  edges,
  faces,
  cells
}; // enum mesh_index_spaces_t

struct mesh_t : public flecsi::data::data_client_t {

  size_t indices(size_t index_space_id) const override {

    switch(index_space_id) {
      case cells:
        // FIXME: hardcoded for 8x8 mesh.
        return 64;
      default:
        // FIXME: lookup user-defined index space
        clog_fatal("unknown index space");
        return 0;
    } // switch
  }
};

void driver(int argc, char **argv) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  flecsi::io::simple_definition_t sd("simple2d-8x8.msh");
  flecsi::dmp::weaver weaver(sd);

  using entry_info_t = flecsi::dmp::entry_info_t;

  // primary cells are the ids of all the cell owned by this node that includes
  // both exclusive and shared cells . Since it is backed by a std::set, they
  // are ordered by the ids. However, it does not start from 0 and is not
  // consecutive thus should not be used to index into an local array.
  std::set<size_t> primary_cells = weaver.get_primary_cells();

  // Thus we need to create a map that maps cell ids to either index of an array or
  // just a map from cell id to field of the cell. Currently alive is an unordered_map
  // for cell id to data field. this should be encapsulate into the data accessor/handler.
  mesh_t m;

  // TODO: figure out all the parameters to register_data and get_mutator
  register_data(m, gof, alive, int, sparse, 2, cells, 1);

  // FIXME: what does slot mean?
  auto am0 = get_mutator(m, gof, alive, int, sparse, 0, 2);
  auto am1 = get_mutator(m, gof, alive, int, sparse, 1, 2);

  // populate sparse data storage.
  for (auto cell: primary_cells) {
    am0(cell, 0) = am1(cell, 0) = 0;
  }
  // initialize the center 3 cells to be alive (a row), it is a period 2 blinker
  // going horizontal and then vertical.
  if (primary_cells.count(27) != 0) {
    am0(27, 0) = am1(27, 0) = 1;
  }
  if (primary_cells.count(35) != 0) {
    am0(35, 0) = am1(35, 0) = 1;
  }
  if (primary_cells.count(43) != 0) {
    am0(43, 0) = am1(43, 0) = 1;
  }

  std::set <entry_info_t> shared_cells = weaver.get_shared_cells();
  std::set <entry_info_t> ghost_cells = weaver.get_ghost_cells();

  // add entries for ghost cells (that is not own by us)
  for (auto ghost : ghost_cells) {
    am0(ghost.id, 0) = am1(ghost.id, 0) = 0;
  }
  am0.commit();
  am1.commit();

  auto a0 = get_accessor(m, gof, alive, int, sparse, 0);
  auto a1 = get_accessor(m, gof, alive, int, sparse, 1);

  for (int i = 0; i < 5; i++) {
    execute_task(halo_exchange_task, loc, single, sd, shared_cells, ghost_cells, a0);
    if (i % 2 == 0) {
      execute_task(stencil_task, loc, single, sd, primary_cells, a0, a1);
      ASSERT_EQ(a1(27, 0), 0);
      ASSERT_EQ(a1(43, 0), 0);

      ASSERT_EQ(a1(34, 0), 1);
      ASSERT_EQ(a1(35, 0), 1);
      ASSERT_EQ(a1(36, 0), 1);
    } else {
      execute_task(stencil_task, loc, single, sd, primary_cells, a1, a0);
      ASSERT_EQ(a0(34, 0), 0);
      ASSERT_EQ(a0(36, 0), 0);

      ASSERT_EQ(a0(27, 0), 1);
      ASSERT_EQ(a0(35, 0), 1);
      ASSERT_EQ(a0(43, 0), 1);
    }
  }

} // driver

#endif //FLECSI_GAME_OF_LIFE_H
