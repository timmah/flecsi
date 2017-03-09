/*~-------------------------------------------------------------------------~~*
 *  @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
 * /@@/////  /@@          @@////@@ @@////// /@@
 * /@@       /@@  @@@@@  @@    // /@@       /@@
 * /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
 * /@@////   /@@/@@@@@@@/@@       ////////@@/@@
 * /@@       /@@/@@//// //@@    @@       /@@/@@
 * /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
 * //       ///  //////   //////  ////////  //
 *
 * Copyright (c) 2017 Los Alamos National Laboratory, LLC
 * All rights reserved
 *~-------------------------------------------------------------------------~~*/


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
using accessor_t = flecsi::data::serial::dense_accessor_t<T, flecsi::data::serial_meta_data_t<flecsi::default_user_meta_data_t> >;

void stencil_task(flecsi::io::simple_definition_t& sd,
                  const std::set<size_t>& primary_cells,
                  accessor_t<int>& a0,
                  accessor_t<int>& a1,
                  std::map<size_t, size_t>& g2l)
{
  // This should be replace with some kind of "task"
  for (auto cell : primary_cells) {
    std::set <size_t> me{cell};
    // FIXME: connectivity information should be part of distributed data as well.
    auto closure = flecsi::topology::entity_closure<2, 2, 0>(sd, me);
    auto nearest_neighbors = flecsi::utils::set_difference(closure, me);

    int count = 0;
    for (auto neighbor : nearest_neighbors) {
      if (a0(g2l[neighbor]) == 1) {
        count++;
      }
    }
    if (a0(g2l[cell])) {
      if (count < 2)
        a1(g2l[cell]) = 0;
      else if (count <= 3)
        a1(g2l[cell]) = 1;
      else
        a1(g2l[cell]) = 0;
    } else {
      if (count == 3)
        a1(g2l[cell]) = 1;
    }
  }
}

void halo_exchange_task(const flecsi::io::simple_definition_t& sd,
                        std::set<flecsi::dmp::entry_info_t>& shared_cells,
                        std::set<flecsi::dmp::entry_info_t>& ghost_cells,
                        accessor_t<int>& acc,
                        std::map<size_t, size_t>& g2l)
{
  std::vector <MPI_Request> requests;

  // Post receive for ghost cells
  for (auto ghost : ghost_cells) {
    requests.push_back({});
    MPI_Irecv(&acc(g2l[ghost.id]), 1, MPI_INT,
              ghost.rank, 0, MPI_COMM_WORLD, &requests[requests.size() - 1]);
  }

  // Send shared cells
  for (auto shared : shared_cells) {
    for (auto dest: shared.shared) {
      MPI_Send(&acc(g2l[shared.id]), 1, MPI_INT,
               dest, 0, MPI_COMM_WORLD);
    }
  }

  std::vector <MPI_Status> status(requests.size());
  MPI_Waitall(requests.size(), &requests[0], &status[0]);
}

flecsi_register_task(stencil_task, loc, single);
flecsi_register_task(halo_exchange_task, loc, single);

enum mesh_index_spaces_t : size_t {
  vertices,
  edges,
  faces,
  cells
}; // enum mesh_index_spaces_t

struct mesh_t : public flecsi::data::data_client_t {

  mesh_t(flecsi::dmp::weaver& _weaver) : weaver(_weaver) {}
  size_t indices(size_t index_space_id) const override {

    switch(index_space_id) {
      case cells:
        return weaver.get_primary_cells().size() + weaver.get_ghost_cells().size();
      default:
        // FIXME: lookup user-defined index space
        clog_fatal("unknown index space");
        return 0;
    } // switch
  }

  flecsi::dmp::weaver weaver;
};

void driver(int argc, char **argv) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  flecsi::io::simple_definition_t sd("simple2d-8x8.msh");
  flecsi::dmp::weaver weaver(sd);

  using entry_info_t = flecsi::dmp::entry_info_t;

  // primary cells are the global ids of all the cell owned by this node that
  // includes both exclusive and shared cells. Since it is backed by a std::set,
  // they are ordered by the ids. However, it does not start from 0 and is not
  // consecutive thus should not be used to index into an local array.
  std::set<size_t> primary_cells = weaver.get_primary_cells();
  std::set<entry_info_t> shared_cells = weaver.get_shared_cells();
  std::set<entry_info_t> ghost_cells  = weaver.get_ghost_cells();

//  if (rank == 0)
//    for (auto cell : primary_cells) {
//      std::cout << "primary cell: " << cell << std::endl;
//    }
  // Thus we need to create a map that maps global cell ids provided by the graph
  // definition to indices of an array.
  // This should be encapsulate into the data accessor/handler.
  std::map<size_t, size_t> g2l;
  size_t idx = 0;

  // FIXME: Ben wants to make ghost cells from lower ranks to come first such that the cells
  // will be ordered in the order of global id in the local storage (I guess). However,
  // it does not actually work with the way ParMetis assign cells. For examples, when
  // partition the mesh into 4 pieces, ParMetis assigns cells at the lower right corners
  // to rank 0, rather than rank 1 as I thought.
  for (auto cell : ghost_cells) {
    if (cell.rank < rank)
      g2l[cell.id] = idx++;
  }
  for (auto cell : primary_cells) {
    g2l[cell] = idx++;
  }
  for (auto cell : ghost_cells) {
    if (cell.rank > rank)
      g2l[cell.id] = idx++;
  }

//  if (rank == 0)
//    for (auto pair : g2l) {
//      std::cout << "global id: " << pair.first << ", local index: " << pair.second << std::endl;
//    }
  mesh_t m(weaver);

  flecsi_register_data(m, gof, alive, int, dense, 2, cells);

  auto acc0 = flecsi_get_accessor(m, gof, alive, int, dense, 0);
  auto acc1 = flecsi_get_accessor(m, gof, alive, int, dense, 1);

  // populate data storage.
  for (auto cell: primary_cells) {
    acc0(g2l[cell]) = acc1(g2l[cell]) = 0;
  }
  // initialize the center 3 cells to be alive (a row), it is a period 2 blinker
  // going horizontal and then vertical.
  // FIXME: if we renumber global id, how are we going to do this kind of initialization?
  if (primary_cells.count(27) != 0) {
    acc0(g2l[27]) = acc1(g2l[27]) = 1;
  }
  if (primary_cells.count(35) != 0) {
    acc0(g2l[35]) = acc1(g2l[35]) = 1;
  }
  if (primary_cells.count(43) != 0) {
    acc0(g2l[43]) = acc1(g2l[43]) = 1;
  }

  // add entries for ghost cells (that is not own by us)
  for (auto ghost : ghost_cells) {
    acc0(g2l[ghost.id]) = acc1(g2l[ghost.id]) = 0;
  }

  for (int i = 0; i < 5; i++) {
    if (i % 2 == 0) {
      flecsi_execute_task(halo_exchange_task, loc, single, sd, shared_cells, ghost_cells, acc0, g2l);
      flecsi_execute_task(stencil_task, loc, single, sd, primary_cells, acc0, acc1, g2l);
      if (primary_cells.count(27) != 0) {
        ASSERT_EQ(acc1(g2l[27]), 0);
      }
      if (primary_cells.count(43) != 0) {
        ASSERT_EQ(acc1(g2l[43]), 0);
      }
      if (primary_cells.count(34) != 0) {
        ASSERT_EQ(acc1(g2l[34]), 1);
      }
      if (primary_cells.count(35) != 0) {
        ASSERT_EQ(acc1(g2l[35]), 1);
      }
      if (primary_cells.count(36) != 0) {
        ASSERT_EQ(acc1(g2l[36]), 1);
      }
    } else {
      flecsi_execute_task(halo_exchange_task, loc, single, sd, shared_cells, ghost_cells, acc1, g2l);
      flecsi_execute_task(stencil_task, loc, single, sd, primary_cells, acc1, acc0, g2l);
      if (primary_cells.count(34) != 0) {
        ASSERT_EQ(acc0(g2l[34]), 0);
      }
      if (primary_cells.count(36) != 0) {
        ASSERT_EQ(acc0(g2l[36]), 0);
      }
      if (primary_cells.count(27) != 0) {
        ASSERT_EQ(acc0(g2l[27]), 1);
      }
      if (primary_cells.count(35) != 0) {
        ASSERT_EQ(acc0(g2l[35]), 1);
      }
      if (primary_cells.count(43) != 0) {
        ASSERT_EQ(acc0(g2l[43]), 1);
      }
    }
  }
} // driver

#endif //FLECSI_GAME_OF_LIFE_H
/*~------------------------------------------------------------------------~--*
 * Formatting options
 * vim: set tabstop=2 shiftwidth=2 expandtab :
 *~------------------------------------------------------------------------~--*/
