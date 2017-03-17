//
// Created by ollie on 3/16/17.
//

#ifndef FLECSI_SIMPLE_DISTRIBUTED_MESH_H
#define FLECSI_SIMPLE_DISTRIBUTED_MESH_H

#include "flecsi/partition/weaver.h"
#include "flecsi/topology/index_space.h"
#include "flecsi/topology/graph_utils.h"
#include <flecsi/topology/mesh_topology.h>

#include "simple_entity_types.h"
#include "simple_types.h"

enum simple_distributed_mesh_index_spaces_t : size_t {
  vertices,
  edges,
  faces,
  cells
}; // enum mesh_index_spaces_t

enum simple_distributed_cell_index_spaces_t : size_t {
  primary,
  ghost
};

class simple_distributed_mesh_t : public flecsi::topology::mesh_topology_t<simple_types_t> {
public:
  using base_t = flecsi::topology::mesh_topology_t<simple_types_t>;

  using entry_info_t = flecsi::dmp::entry_info_t;

  using vertex_t = simple_vertex_t;

  simple_distributed_mesh_t(const char* filename) : sd(filename), weaver(sd) {
    primary_vertices = weaver.get_primary_vertices();
    exclusive_vertices = weaver.get_exclusive_vertices();
    shared_vertices = weaver.get_shared_vertices();
    ghost_vertices = weaver.get_ghost_vertices();

    std::vector<vertex_t *> vs;
    std::map<size_t, size_t> vertex_global_to_local_map;
    size_t index = 0;
    for (auto vertex : primary_vertices) {
      // get the vertex coordinates from sd.
      vs.push_back(create_vertex(sd.vertex(vertex), vertex));
      vertex_global_to_local_map[vertex] = index++;
    }
    for (auto vertex : ghost_vertices) {
      vs.push_back(create_vertex(sd.vertex(vertex.id), vertex.id));
      vertex_global_to_local_map[vertex.id] = index++;
    }

    // TODO: do we need global <-> local mapping for vertices?

    primary_cells = weaver.get_primary_cells();
    exclusive_cells = weaver.get_exclusive_cells();
    shared_cells = weaver.get_shared_cells();
    ghost_cells = weaver.get_ghost_cells();

    // TODO: add ghost cells, add cell.type for primary and ghost.
    for (auto cell : primary_cells) {
      auto vertices = sd.vertices(2, cell);
      create_cell({vs[vertex_global_to_local_map.at(vertices[0])],
                   vs[vertex_global_to_local_map.at(vertices[1])],
                   vs[vertex_global_to_local_map.at(vertices[2])],
                   vs[vertex_global_to_local_map.at(vertices[3])]},
                  primary,
                  cell);
    }

    for (auto cell : ghost_cells) {
      auto vertices = sd.vertices(2, cell.id);
      create_cell({vs[vertex_global_to_local_map.at(vertices[0])],
                   vs[vertex_global_to_local_map.at(vertices[1])],
                   vs[vertex_global_to_local_map.at(vertices[2])],
                   vs[vertex_global_to_local_map.at(vertices[3])]},
                  ghost,
                  cell.id);
    }

    init();


    //size_t index = 0;
    index = 0;
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
  void
  init()
  {
    // Initialize domain 0 of the mesh topology.
    base_t::init<0>();

    // Use a predicate function to create the interior cells
    // index space
    primary_cells_ =
      base_t::entities<2, 0>().filter(is_primary);

    // Use a predicate function to create the domain boundary cells
    // index space
    ghost_cells_ =
      base_t::entities<2, 0>().filter([](auto cell) {return cell->type == ghost;});
  } // init

  static bool
  is_primary(simple_cell_t *cell)
  {
    return cell->type == primary;
  }

  static bool
  is_ghost(simple_cell_t *cell)
  {

  }
  vertex_t *
  create_vertex(const flecsi::point<double, 2>& pos, size_t mesh_id)
  {
    auto v = base_t::make<vertex_t>(*this, pos);
    base_t::add_entity<0, 0>(v, 0, mesh_id);
    return v;
  }

  simple_cell_t *
  create_cell(const std::initializer_list<vertex_t *> & vertices, size_t type, size_t mesh_id)
  {
    auto c = base_t::make<simple_cell_t>(*this, type);
    base_t::add_entity<2, 0>(c, 0, mesh_id);
    base_t::init_entity<0, 2, 0>(c, vertices);
    return c;
  }

  size_t indices(size_t index_space_id) const override {
    switch(index_space_id) {
      case simple_distributed_mesh_index_spaces_t::cells:
        return primary_cells.size() + ghost_cells.size();
      default:
        // FIXME: lookup user-defined index space
        clog_fatal("unknown index space");
        return 0;
    } // switch
  }

  auto
  cells(
    size_t is
  )
  {
    switch(is) {
      case simple_distributed_cell_index_spaces_t::primary:
        return primary_cells_;
      case simple_distributed_cell_index_spaces_t::ghost:
        return ghost_cells_;
      default:
        assert(false && "unknown index space");
    } // switch
  } // cells

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
  // TODO: shoud we retain these two members?
  flecsi::io::simple_definition_t sd;
  flecsi::dmp::weaver weaver;

  // TODO: there is paritions member in the data_client_t, how should be reuse it?
  std::set <size_t> primary_cells;
  std::set <entry_info_t> exclusive_cells;
  std::set <entry_info_t> shared_cells;
  std::set <entry_info_t> ghost_cells;

  std::set <size_t> primary_vertices;
  std::set <entry_info_t> exclusive_vertices;
  std::set <entry_info_t> shared_vertices;
  std::set <entry_info_t> ghost_vertices;

  std::map<size_t, size_t> global_to_local_map;
  std::vector<size_t> local_to_global_map;

  std::map<size_t, std::vector<size_t>> global_cell_to_cell_conn;
  std::map<size_t, std::vector<size_t>> local_cell_to_cell_conn;

  flecsi::topology::index_space<
    flecsi::topology::domain_entity<0, simple_cell_t>, false, true, false> primary_cells_;
  flecsi::topology::index_space<
    flecsi::topology::domain_entity<0, simple_cell_t>, false, true, false> ghost_cells_;
};

#endif //FLECSI_SIMPLE_DISTRIBUTED_MESH_H
