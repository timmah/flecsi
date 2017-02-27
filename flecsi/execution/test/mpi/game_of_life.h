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

void stencil_task(flecsi::io::simple_definition_t& sd,
                  const std::set<size_t>& primary_cells,
                  std::unordered_map<size_t, int>& alive,
                  std::unordered_map<size_t, int>& alive2)
{
  // This should be replace with some kind of "task"
  for (auto cell : primary_cells) {
    std::set <size_t> me{cell};
    // FIXME: connectivity information should be part of distributed data as well.
    auto closure = flecsi::topology::entity_closure<2, 2, 0>(sd, me);
    auto nearest_neighbors = flecsi::utils::set_difference(closure, me);

    int count = 0;
    for (auto neighbor : nearest_neighbors) {
      if (neighbor != -1 && alive[neighbor] == 1) {
        count++;
      }
    }
    if (alive[cell]) {
      if (count < 2)
        alive2[cell] = 0;
      else if (count <= 3)
        alive2[cell] = 1;
      else
        alive2[cell] = 0;
    } else {
      if (count == 3)
        alive2[cell] = 1;
    }
  }

  for (auto cell : primary_cells) {
    if (alive2[cell] == 1) {
      std::cout << "primary cell: " << cell
                << ", alive: " << alive2[cell]
                << std::endl;
    }
  }
  std::cout << std::endl;

  std::swap(alive, alive2);
}

void halo_exchange_task(const flecsi::io::simple_definition_t& sd,
                        std::set<flecsi::dmp::entry_info_t>& shared_cells,
                        std::set<flecsi::dmp::entry_info_t>& ghost_cells,
                        std::unordered_map<size_t, int>& alive)
{
  std::vector <MPI_Request> requests;

  // Post receive for ghost cells
  for (auto ghost : ghost_cells) {
    requests.push_back({});
    MPI_Irecv(&alive[ghost.id], 1, MPI_INT,
              ghost.rank, 0, MPI_COMM_WORLD, &requests[requests.size() - 1]);
  }

  // Send shared cells
  for (auto shared : shared_cells) {
    for (auto dest: shared.shared) {
      MPI_Send(&alive[shared.id], 1, MPI_INT,
               dest, 0, MPI_COMM_WORLD);
    }
  }

  std::vector <MPI_Status> status(requests.size());
  MPI_Waitall(requests.size(), &requests[0], &status[0]);
}

register_task(stencil_task, loc, single);
register_task(halo_exchange_task, loc, single);

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
  std::unordered_map<size_t, int> alive;
  for (auto cell: primary_cells) {
    alive[cell] = 0;
  }

  // initialize the center 3 cells to be alive (a row), it is a period 2 blinker
  // going horizontal and then vertical.
  if (primary_cells.count(27) != 0) {
    alive[27] = 1;
  }
  if (primary_cells.count(35) != 0) {
    alive[35] = 1;
  }
  if (primary_cells.count(43) != 0) {
    alive[43] = 1;
  }

  for (auto cell : primary_cells) {
    if (alive[cell] == 1) {
      std::cout << "primary cell: " << cell
                << ", alive: " << alive[cell]
                << std::endl;
    }
  }

  std::set <entry_info_t> shared_cells = weaver.get_shared_cells();
  std::set <entry_info_t> ghost_cells = weaver.get_ghost_cells();

  // add entries for ghost cells (that is not own by us)
  for (auto ghost : ghost_cells) {
    alive[ghost.id] = 0;
  }

  // We need two versions for the state update to work correctly.
  auto alive2 = alive;

  for (auto i = 0; i < 5; i++) {
    execute_task(halo_exchange_task, loc, single, sd, shared_cells, ghost_cells, alive);

    execute_task(stencil_task, loc, single, sd, primary_cells, alive, alive2);
  }
} // driver

#endif //FLECSI_GAME_OF_LIFE_H
