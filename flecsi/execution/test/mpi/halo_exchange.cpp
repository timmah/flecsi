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
#include <string>

#include "flecsi/partition/weaver.h"
#include "flecsi/topology/index_space.h"
#include "flecsi/execution/execution.h"
#include "flecsi/data/data.h"

#include "simple_distributed_mesh.h"

TEST(halo_exchange, send_receive) {
//test() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  simple_distributed_mesh_t mesh("simple2d-8x8.msh");

  // Current layout of local_buffer:
  //     [primary cells ordered by mesh id][ghost cells ordered by mesh id]
  // one other possibility:
  //     [ghost cells from lower rank peers][primary cells][ghost cells from higher rank peers]
  flecsi_register_data(mesh, halo, cell_gid, size_t, dense, 1, cells);
  auto acc = flecsi_get_accessor(mesh, halo, cell_gid, size_t, dense, 0);
  for (size_t local_cell_id = 0; local_cell_id < mesh.num_primary_cells(); local_cell_id++) {
    acc[local_cell_id] = mesh.global_cell_id(local_cell_id);
  }
  for (size_t local_cell_id = mesh.num_primary_cells(); local_cell_id < mesh.indices(cells); local_cell_id++) {
    acc[local_cell_id] = -1;
  }

  std::vector <MPI_Request> requests;
  for (const auto& cell : mesh.get_ghost_cells_info()) {
    requests.push_back({});
    MPI_Irecv(&acc[mesh.local_cell_id(cell.id)], 1, MPI_UNSIGNED_LONG_LONG,
              cell.rank, 0, MPI_COMM_WORLD,
              &requests[requests.size() - 1]);
  }

  for (const auto& cell : mesh.get_shared_cells_info()) {
    for (const auto& dest: cell.shared) {
      MPI_Send(&acc[mesh.local_cell_id(cell.id)], 1, MPI_UNSIGNED_LONG_LONG,
               dest, 0, MPI_COMM_WORLD);
    }
  }

  std::vector<MPI_Status> status(requests.size());
  MPI_Waitall(requests.size(), &requests[0], &status[0]);

  std::set<size_t> closure;
  for (size_t i = 0; i < mesh.indices(cells); i++) {
    closure.insert(mesh.global_cell_id(i));
  }

  std::set<size_t> all_cells;
  for (size_t i = 0; i < mesh.indices(cells); i++) {
    all_cells.insert(acc[i]);
  }

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

TEST(halo_exchange, one_sided_get_passive) {
//test() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  simple_distributed_mesh_t mesh("simple2d-8x8.msh");

  // Current layout of local_buffer:
  //     [primary cells ordered by mesh id][ghost cells ordered by mesh id]
  // one other possibility:
  //     [ghost cells from lower rank peers][primary cells][ghost cells from higher rank peers]
  // It turns out that the current layout is very convenient for MPI_Get. The way
  // ghost_cell.offset is calculated is correct to be used as target_disp to fetch
  // data from the remote data buffer (because both of them are order by their global
  // mesh id).
  flecsi_register_data(mesh, halo, cell_gid, size_t, dense, 1, cells);
  auto acc = flecsi_get_accessor(mesh, halo, cell_gid, size_t, dense, 0);
  for (size_t local_cell_id = 0; local_cell_id < mesh.num_primary_cells(); local_cell_id++) {
    acc[local_cell_id] = mesh.global_cell_id(local_cell_id);
  }
  for (size_t local_cell_id = mesh.num_primary_cells(); local_cell_id < mesh.indices(cells); local_cell_id++) {
    acc[local_cell_id] = -1;
  }

  // A pull model using MPI_Get:
  // 1. create MPI window for primary cell portion of the local buffer, some of
  // them are shared cells that can be fetch by peer as ghost cells.
  MPI_Win win;
  MPI_Win_create(&acc[0], mesh.indices(cells) * sizeof(size_t),
                 sizeof(size_t), MPI_INFO_NULL, MPI_COMM_WORLD,
                 &win);

  // 2. iterate through each ghost cell and MPI_Get from the peer.
  for (auto& cell : mesh.get_ghost_cells_info()) {
    MPI_Win_lock(MPI_LOCK_SHARED, cell.rank, 0, win);
    MPI_Get(&acc[mesh.local_cell_id(cell.id)], 1, MPI_UNSIGNED_LONG_LONG,
            cell.rank, cell.offset, 1, MPI_UNSIGNED_LONG_LONG, win);
    MPI_Win_unlock(cell.rank, win);
  }

  MPI_Win_free(&win);

  std::set<size_t> closure;
  for (size_t i = 0; i < mesh.indices(cells); i++) {
    closure.insert(mesh.global_cell_id(i));
  }

  std::set<size_t> all_cells;
  for (size_t i = 0; i < mesh.indices(cells); i++) {
    all_cells.insert(acc[i]);
  }

  ASSERT_EQ(all_cells, closure);

}

TEST(halo_exchange, pscw) {
//test() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  simple_distributed_mesh_t mesh("simple2d-8x8.msh");

  // Current layout of local_buffer:
  //     [primary cells ordered by mesh id][ghost cells ordered by mesh id]
  // one other possibility:
  //     [ghost cells from lower rank peers][primary cells][ghost cells from higher rank peers]
  // It turns out that the current layout is very convenient for MPI_Get. The way
  // ghost_cell.offset is calculated is correct to be used as target_disp to fetch
  // data from the remote data buffer (because both of them are order by their global
  // mesh id).
  flecsi_register_data(mesh, halo, cell_gid, size_t, dense, 1, cells);
  auto acc = flecsi_get_accessor(mesh, halo, cell_gid, size_t, dense, 0);
  for (size_t local_cell_id = 0; local_cell_id < mesh.num_primary_cells(); local_cell_id++) {
    acc[local_cell_id] = mesh.global_cell_id(local_cell_id);
  }
  for (size_t local_cell_id = mesh.num_primary_cells(); local_cell_id < mesh.indices(cells); local_cell_id++) {
    acc[local_cell_id] = -1;
  }

  auto ranks = mesh.get_shared_peers();

  MPI_Group comm_grp, rma_group;
  MPI_Comm_group(MPI_COMM_WORLD, &comm_grp);
  MPI_Group_incl(comm_grp, ranks.size(), ranks.data(), &rma_group);
  MPI_Group_free(&comm_grp);

  // A pull model using MPI_Get:
  // 1. create MPI window for primary cell portion of the local buffer, some of
  // them are shared cells that can be fetch by peer as ghost cells.
  MPI_Win win;
  MPI_Win_create(&acc[0], mesh.num_primary_cells() * sizeof(size_t),
                 sizeof(size_t), MPI_INFO_NULL, MPI_COMM_WORLD,
                 &win);
  // 2. iterate through each ghost cell and MPI_Get from the peer.
  MPI_Win_post(rma_group, 0, win);
  MPI_Win_start(rma_group, 0, win);

  size_t idx = mesh.num_primary_cells();
  for (auto& cell : mesh.get_ghost_cells_info()) {
    MPI_Get(&acc[idx++], 1, MPI_UNSIGNED_LONG_LONG,
            cell.rank, cell.offset, 1, MPI_UNSIGNED_LONG_LONG, win);
  }

  MPI_Win_complete(win);
  MPI_Win_wait(win);

  MPI_Group_free(&rma_group);
  MPI_Win_free(&win);

  std::set<size_t> closure;
  for (size_t i = 0; i < mesh.indices(cells); i++) {
    closure.insert(mesh.global_cell_id(i));
  }

  std::set<size_t> all_cells;
  for (size_t i = 0; i < mesh.indices(cells); i++) {
    all_cells.insert(acc[i]);
  }

  ASSERT_EQ(all_cells, closure);

}