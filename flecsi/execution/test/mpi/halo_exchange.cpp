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


#include <cinchtest.h>
#include <numeric>
#include <mpi.h>

#include "flecsi/partition/weaver.h"
#include "flecsi/topology/index_space.h"

TEST(halo_exchange, send_receive) {
//test() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  flecsi::io::simple_definition_t sd("simple2d-8x8.msh");
  flecsi::dmp::weaver weaver(sd);

  using entry_info_t = flecsi::dmp::entry_info_t;

  std::set <size_t> primary_cells = weaver.get_primary_cells();

  std::set <entry_info_t> exclusive_cells = weaver.get_exclusive_cells();
  std::set <entry_info_t> shared_cells = weaver.get_shared_cells();
  std::set <entry_info_t> ghost_cells = weaver.get_ghost_cells();

  // Current layout of local_buffer:
  //     [primary cells ordered by mesh id][ghost cells ordered by mesh id]
  // one other possibility:
  //     [ghost cells from lower rank peers][primary cells][ghost cells from higher rank peers]
  std::map<size_t, size_t> g2l;
  size_t idx = 0;

  for (const auto& cell : primary_cells) {
    g2l[cell] = idx++;
  }
  for (const auto& cell : ghost_cells) {
    g2l[cell.id] = idx++;
  }

  if (rank == 1) {
    for (auto cell : ghost_cells) {
      std::cout << "ghost cell id: " << cell.id << ", from rank: "
                << cell.rank << ", offset: " << cell.offset << std::endl;
    }
  }
  std::vector<size_t> local_buffer(primary_cells.size() + ghost_cells.size(), -1);
  // initialize local_buffer with ids of primary cells (exclusive + shared).
  std::copy(primary_cells.begin(), primary_cells.end(), local_buffer.begin());

  std::vector <MPI_Request> requests;
  for (const auto& cell : ghost_cells) {
    requests.push_back({});
    MPI_Irecv(&local_buffer[g2l[cell.id]], 1, MPI_UNSIGNED_LONG_LONG,
              cell.rank, 0, MPI_COMM_WORLD,
              &requests[requests.size() - 1]);
  }

  for (const auto& cell : shared_cells) {
    for (const auto& dest: cell.shared) {
      MPI_Send(&local_buffer[g2l[cell.id]], 1, MPI_UNSIGNED_LONG_LONG,
               dest, 0, MPI_COMM_WORLD);
    }
  }

  std::vector<MPI_Status> status(requests.size());
  MPI_Waitall(requests.size(), &requests[0], &status[0]);

  auto closure = flecsi::topology::entity_closure<2, 2, 0>(sd, primary_cells);
  auto all_cells = std::set<size_t>(local_buffer.begin(), local_buffer.end());
  ASSERT_EQ(all_cells, closure);
}

TEST(halo_exchange, scatter_gather) {
//test() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  flecsi::io::simple_definition_t sd("simple2d-8x8.msh");
  flecsi::dmp::weaver weaver(sd);

  using entry_info_t = flecsi::dmp::entry_info_t;

  std::set <size_t> primary_cells = weaver.get_primary_cells();

  std::set <entry_info_t> exclusive_cells = weaver.get_exclusive_cells();
  std::set <entry_info_t> shared_cells = weaver.get_shared_cells();
  std::set <entry_info_t> ghost_cells = weaver.get_ghost_cells();

  // Current layout of local_buffer:
  //     [primary cells ordered by mesh id][ghost cells ordered by mesh id]
  // one other possibility:
  //     [ghost cells from lower rank peers][primary cells][ghost cells from higher rank peers]
  std::map<size_t, size_t> g2l;
  size_t idx = 0;

  for (const auto& cell : primary_cells) {
    g2l[cell] = idx++;
  }
  for (const auto& cell : ghost_cells) {
    g2l[cell.id] = idx++;
  }

  std::vector<size_t> local_buffer(primary_cells.size() + ghost_cells.size(), -1);
  // initialize local_buffer with ids of primary cells (exclusive + shared).
  std::copy(primary_cells.begin(), primary_cells.end(), local_buffer.begin());

  // gather shared cells into the send_buffer
  std::map<size_t, std::vector<size_t>> send_buffers;
  for (const auto& cell : shared_cells) {
    for (const auto& dest: cell.shared) {
      if (send_buffers.find(dest) == send_buffers.end()) {
        send_buffers[dest] = std::vector<size_t>();
      }
      send_buffers[dest].push_back(local_buffer[g2l[cell.id]]);
    }
  }

  std::map<size_t, size_t> recv_counts;
  for (const auto& cell : ghost_cells) {
    if (recv_counts.find(cell.rank) == recv_counts.end()) {
      recv_counts[cell.rank] = 1;
    } else {
      recv_counts[cell.rank] += 1;
    }
  }

  std::map<size_t, std::vector<size_t>> recv_buffers;
  for (const auto& pair : recv_counts) {
    recv_buffers[pair.first] = std::vector<size_t>(pair.second, - 1);
  }

  std::map<size_t, std::vector<size_t>> scatter_list;
  for (const auto& cell : ghost_cells) {
    if (scatter_list.find(cell.rank) == scatter_list.end()) {
      scatter_list[cell.rank] = std::vector<size_t>();
    }
    scatter_list[cell.rank].push_back(g2l[cell.id]);
  }

  std::vector <MPI_Request> requests;
  for (auto &buffer : recv_buffers) {
    requests.push_back({});
    MPI_Irecv(buffer.second.data(), buffer.second.size(), MPI_UNSIGNED_LONG_LONG,
              buffer.first, 0, MPI_COMM_WORLD,
              &requests[requests.size() - 1]);
  }

  for (const auto &buffer : send_buffers) {
    MPI_Send(&buffer.second[0], buffer.second.size(), MPI_UNSIGNED_LONG_LONG,
             buffer.first, 0, MPI_COMM_WORLD);
  }

  std::vector<MPI_Status> status(requests.size());
  int err = MPI_Waitall(requests.size(), &requests[0], &status[0]);

  // scatter receive buffer for ghost cells into local buffer
  for (const auto &pair : recv_buffers) {
    auto rank = pair.first;
    for (auto i = 0; i < pair.second.size(); i++) {
      auto offset = scatter_list[rank][i];
      local_buffer[offset] = pair.second[i];
    }
  }

  auto closure = flecsi::topology::entity_closure<2, 2, 0>(sd, primary_cells);
  auto all_cells = std::set<size_t>(local_buffer.begin(), local_buffer.end());
  ASSERT_EQ(all_cells, closure);
}

TEST(halo_exchange, one_sided_get) {
//test() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  flecsi::io::simple_definition_t sd("simple2d-8x8.msh");
  flecsi::dmp::weaver weaver(sd);

  using entry_info_t = flecsi::dmp::entry_info_t;

  std::set <size_t> primary_cells = weaver.get_primary_cells();

  std::set <entry_info_t> exclusive_cells = weaver.get_exclusive_cells();
  std::set <entry_info_t> shared_cells = weaver.get_shared_cells();
  std::set <entry_info_t> ghost_cells = weaver.get_ghost_cells();

  // Current layout of local_buffer:
  //     [primary cells ordered by mesh id][ghost cells ordered by mesh id]
  // one other possibility:
  //     [ghost cells from lower rank peers][primary cells][ghost cells from higher rank peers]
  std::map <size_t, size_t> g2l;
  size_t idx = 0;

  for (const auto &cell : primary_cells) {
    g2l[cell] = idx++;
  }
  for (const auto &cell : ghost_cells) {
    g2l[cell.id] = idx++;
  }

  std::vector <size_t> local_buffer(primary_cells.size() + ghost_cells.size(), -1);
  // initialize local_buffer with ids of primary cells (exclusive + shared).
  // leave the ghost cell part un-initialized.
  std::copy(primary_cells.begin(), primary_cells.end(), local_buffer.begin());

  // A pull model using MPI_Get:
  // 1. create MPI window for primary cell portion of the local buffer, some of
  // them are shared cells that can be fetch by peer as ghost cells.
  MPI_Win win;
  MPI_Win_create(local_buffer.data(), primary_cells.size() * sizeof(size_t),
                 sizeof(size_t), MPI_INFO_NULL, MPI_COMM_WORLD,
                 &win);

  // 2. iterate through each ghost cell and MPI_Get from the peer.
  MPI_Win_fence(0, win);
  idx = primary_cells.size();
  for (auto& cell : ghost_cells) {
    MPI_Get(&local_buffer[idx++], 1, MPI_UNSIGNED_LONG_LONG,
            cell.rank, cell.offset, 1, MPI_UNSIGNED_LONG_LONG, win);
  }
  MPI_Win_fence(0, win);

  auto closure = flecsi::topology::entity_closure<2, 2, 0>(sd, primary_cells);
  auto all_cells = std::set<size_t>(local_buffer.begin(), local_buffer.end());
  ASSERT_EQ(all_cells, closure);

}