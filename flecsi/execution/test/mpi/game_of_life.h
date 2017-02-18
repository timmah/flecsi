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

void driver(int argc, char **argv) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  flecsi::io::simple_definition_t sd("simple2d-8x8.msh");
  flecsi::dmp::weaver weaver(sd);

  using entry_info_t = flecsi::dmp::entry_info_t;

  // primary cells are the ids of all the cell owned by this node. Since
  // it is backed by a std::set, they are ordered by the ids. However, it
  // does not start from 0 and is not consecutive thus should not be
  // used to index into an local array.
  std::set <size_t> primary_cells = weaver.get_primary_cells();

  // currently alive is an unordered_map for cell id to data field.
  // this should be encapsulate into the data accessor/handler.
  std::unordered_map<size_t, int> alive;
  for (auto cell: primary_cells) {
    alive[cell] = 0;
  }

  // initialize the center 3 cells to be alive (a row), it is a period 2 blinker
  // going horizontal and then vertical.
  if (alive.count(27) != 0) {
    alive[27] = 1;
  }
  if (alive.count(35) != 0) {
    alive[35] = 1;
  }
  if (alive.count(43) != 0) {
    alive[43] = 1;
  }

  for (auto cell : primary_cells) {
    if (alive[cell] == 1) {
      std::cout << "primary cell: " << cell
                << ", alive: " << alive[cell]
                << std::endl;
    }
  }
  std::set <entry_info_t> exclusive_cells = weaver.get_exclusive_cells();
  std::set <entry_info_t> shared_cells = weaver.get_shared_cells();
  std::set <entry_info_t> ghost_cells = weaver.get_ghost_cells();

  // add entries for ghost cells (that is not own by us)
  for (auto ghost : ghost_cells) {
    alive[ghost.id] = 0;
  }

  // We need two versions for the state update to work correctly.
  auto alive2 = alive;

  for (auto i = 0; i < 5; i++) {
    // Do halo exchange first
    std::vector <MPI_Request> requests;
    // Post receive for ghost cells
    for (auto ghost : ghost_cells) {
      requests.push_back({});
      MPI_Irecv(&alive[ghost.id], 1, MPI_INT,
                ghost.rank, 0, MPI_COMM_WORLD, &requests[requests.size() - 1]);
    }
    // Send shared cells
    for (auto entry : shared_cells) {
      for (auto dest: entry.shared) {
        MPI_Send(&alive[entry.id], 1, MPI_INT,
                 dest, 0, MPI_COMM_WORLD);
      }
    }

    std::vector <MPI_Status> status(requests.size());
    MPI_Waitall(requests.size(), &requests[0], &status[0]);

    // This should be replace with some kind of "task"
    for (auto cell : primary_cells) {
      std::set <size_t> me{cell};
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
} // driver

#endif //FLECSI_GAME_OF_LIFE_H
