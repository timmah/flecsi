//
// Created by ollie on 3/16/17.
//

#ifndef FLECSI_SIMPLE_DISTRIBUTED_MESH_H
#define FLECSI_SIMPLE_DISTRIBUTED_MESH_H

#include "flecsi/partition/weaver.h"
#include "flecsi/topology/index_space.h"
#include "flecsi/topology/graph_utils.h"

enum simple_distributed_mesh_index_spaces_t : size_t {
  vertices,
  edges,
  faces,
  cells
}; // enum mesh_index_spaces_t

class simple_distributed_mesh_t : public flecsi::data::data_client_t {
public:

  using entry_info_t = flecsi::dmp::entry_info_t;

  simple_distributed_mesh_t(const char* filename) : sd(filename), weaver(sd) {
    primary_cells = weaver.get_primary_cells();
    exclusive_cells = weaver.get_exclusive_cells();
    shared_cells = weaver.get_shared_cells();
    ghost_cells = weaver.get_ghost_cells();

    size_t index = 0;
    for (const auto& cell : primary_cells) {
      global_to_local_map[cell] = index++;
      local_to_global_map.push_back(cell);
    }
    for (const auto& cell : ghost_cells) {
      global_to_local_map[cell.id] = index++;
      local_to_global_map.push_back(cell.id);
    }

    // TODO: I don't think that we need cell to cell connectivity for ghost cell
    // but I may be wrong.
    // TODO: properly inherit from mesh_topology_t and deal with connectivity
    // correctly.
    for (const auto& cell : primary_cells) {
      std::set <size_t> me{cell};
      auto closure = flecsi::topology::entity_closure<2, 2, 0>(sd, me);
      auto nearest_neighbors = flecsi::utils::set_difference(closure, me);

      std::vector<size_t> neighbor_global_ids;
      std::vector<size_t> neighbor_local_ids;
      for (auto neigbor : nearest_neighbors) {
        neighbor_global_ids.push_back(neigbor);
        neighbor_local_ids.push_back(global_to_local_map[neigbor]);
      }
      global_cell_to_cell_conn[cell] = std::move(neighbor_global_ids);
      local_cell_to_cell_conn[global_to_local_map[cell]] = std::move(neighbor_local_ids);
    }
  }

  size_t indices(size_t index_space_id) const override {
    switch(index_space_id) {
      case cells:
        return primary_cells.size() + ghost_cells.size();
      default:
        // FIXME: lookup user-defined index space
        clog_fatal("unknown index space");
        return 0;
    } // switch
  }

  // convert a local cell id into a global cell id
  size_t global_cell_id(size_t cell_local_id) {
    if (cell_local_id > local_to_global_map.size()) {
      throw std::invalid_argument("invalid local cell id: " + std::to_string(cell_local_id));
    } else {
      return local_to_global_map[cell_local_id];
    }
  }

  // convert a global cell in into a global cell id
  size_t local_cell_id(size_t cell_global_id) {
    auto iter = global_to_local_map.find(cell_global_id);
    if (iter == global_to_local_map.end()) {
      throw std::invalid_argument("invalid global cell id");
    } else {
      return iter->second;
    }
  }

  // get number of primary cells
  size_t num_primary_cells() const {
    return primary_cells.size();
  }

  // get number of ghost cells
  size_t num_ghost_cells() const {
    return ghost_cells.size();
  }

  // Find the peers to form a MPI group. We need both peers that need our shared
  // cells and the peers that provides us ghost cells.
  std::vector<int> get_shared_peers() const {
    std::set<int> rank_set;
    for (auto cell : shared_cells) {
      for (auto peer : cell.shared) {
        rank_set.insert(peer);
      }
    }
    for (auto cell : ghost_cells) {
      rank_set.insert(cell.rank);
    }
    std::vector<int> peers(rank_set.begin(), rank_set.end());
    return peers;
  }

  // FIXME: Do we really need to expose these information?
  std::set <entry_info_t> get_ghost_cells_info() const {
    return ghost_cells;
  }
  std::set <entry_info_t> get_shared_cells_info() const {
    return shared_cells;
  }
  std::set <size_t> get_primary_cells() const {
    return primary_cells;
  }
  std::vector<size_t> get_local_cell_neighbors(size_t local_id) {
    return local_cell_to_cell_conn[local_id];
  }

private:
  flecsi::io::simple_definition_t sd;
  flecsi::dmp::weaver weaver;

  // TODO: there is paritions member in the data_client_t, how should be reuse it?
  std::set <size_t> primary_cells;
  std::set <entry_info_t> exclusive_cells;
  std::set <entry_info_t> shared_cells;
  std::set <entry_info_t> ghost_cells;

  std::map<size_t, size_t> global_to_local_map;
  std::vector<size_t> local_to_global_map;

  std::map<size_t, std::vector<size_t>> global_cell_to_cell_conn;
  std::map<size_t, std::vector<size_t>> local_cell_to_cell_conn;
};

#endif //FLECSI_SIMPLE_DISTRIBUTED_MESH_H
