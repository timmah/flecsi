/*~--------------------------------------------------------------------------~*
 * Copyright (c) 2015 Los Alamos National Security, LLC
 * All rights reserved.
 *~--------------------------------------------------------------------------~*/

#ifndef flecsi_sprint_h
#define flecsi_sprint_h

#include <iostream>

#include "flecsi/utils/common.h"
#include "flecsi/execution/context.h"
#include "flecsi/execution/execution.h"
#include "flecsi/partition/index_partition.h"
#include "flecsi/execution/mpilegion/init_partitions_task.h"
#include "flecsi/partition/weaver.h"
#include "flecsi/execution/legion/dpd.h"

///
// \file sprint.h
// \authors bergen
// \date Initial file creation: Aug 23, 2016
///

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

namespace flecsi {
namespace execution {

static const size_t N = 8;

using index_partition_t = dmp::index_partition__<size_t>;

void
mpi_task(
  double d
)
{
  int rank = 0;
  int size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::cout << "My rank: " << rank << std::endl;

  flecsi::io::simple_definition_t sd("simple2d-8x8.msh");
  flecsi::dmp::weaver weaver(sd);

  using entry_info_t = flecsi::dmp::entry_info_t;

  index_partition_t ip_cells;

  ip_cells.primary = weaver.get_primary_cells();
  ip_cells.exclusive = weaver.get_exclusive_cells();
  ip_cells.shared = weaver.get_shared_cells();
  ip_cells.ghost  = weaver.get_ghost_cells();
  ip_cells.entities_per_rank = weaver.get_n_cells_per_rank();

  index_partition_t ip_vertices;

  ip_vertices.primary = weaver.get_primary_vertices();
  ip_vertices.exclusive = weaver.get_exclusive_vertices();
  ip_vertices.shared = weaver.get_shared_vertices();
  ip_vertices.ghost  = weaver.get_ghost_vertices();
  ip_vertices.entities_per_rank = weaver.get_n_vertices_per_rank();

  std::vector<std::pair<size_t, size_t>> raw_conns = 
    weaver.get_raw_cell_vertex_conns();

#if 0
   std::cout <<"DEBUG CELLS"<<std::endl;
   size_t i=0;
   for (auto cells_p : ip_cells.primary)
   {
    std::cout<<"primary["<<i<<"] = " <<cells_p<<std::endl;
    i++;
   }
   i=0;
   for (auto cells_s : ip_cells.shared)
   {
    std::cout<<"shared["<<i<<"] = " <<cells_s.id<< ", offset = "<< 
       cells_s.offset<<std::endl;
    i++;
   }


   std::cout <<"DEBUG VERTICES"<<std::endl;
   i=0;
   for (auto vert_p : ip_vertices.primary)
   {
    std::cout<<"primary["<<i<<"] = " <<vert_p<<std::endl;
    i++;
   }
   i=0;
   for (auto vert_s : ip_vertices.shared)
   {
    std::cout<<"shared["<<i<<"] = " <<vert_s.id<< ", offset = "<<
       vert_s.offset<<std::endl;
    i++;
   }
   i=0;
   for (auto vert_e : ip_vertices.exclusive)
   {
    std::cout<<"exclusive["<<i<<"] = " <<vert_e.id<< ", offset = "<<
       vert_e.offset<<std::endl;
    i++;
   }

#endif

  flecsi::execution::context_t & context_ =
    flecsi::execution::context_t::instance();
  context_.interop_helper_.data_storage_.push_back(
    flecsi::utils::any_t(ip_cells));

  context_.interop_helper_.data_storage_.push_back(
    flecsi::utils::any_t(ip_vertices));

  context_.interop_helper_.data_storage_.push_back(
    flecsi::utils::any_t(raw_conns));
}

  
register_task(mpi_task, mpi, single);

void
driver(
  int argc, 
  char ** argv
)
{
  context_t & context_ = context_t::instance();
  size_t task_key = utils::const_string_t{"driver"}.hash();
  auto runtime = context_.runtime(task_key);
  auto context = context_.context(task_key);

  legion_helper h(runtime, context);

  using legion_domain = LegionRuntime::HighLevel::Domain;
  field_ids_t & fid_t =field_ids_t::instance();

  flecsi::execution::sprint::parts partitions;
  
  // first execute mpi task to setup initial partitions 
  execute_task(mpi_task, mpi, single, 1.0);
  // create a field space to store cells id

  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  LegionRuntime::Arrays::Rect<1> rank_rect(LegionRuntime::Arrays::Point<1>(0),
    LegionRuntime::Arrays::Point<1>(num_ranks - 1));
  Domain rank_domain = Domain::from_rect<1>(rank_rect);

  //call task to calculate total_num_cells and to get number of 
  //cells per partiotioning

  LegionRuntime::HighLevel::ArgumentMap arg_map;

  LegionRuntime::HighLevel::IndexLauncher get_numbers_of_cells_launcher(
    task_ids_t::instance().get_numbers_of_cells_task_id,
    legion_domain::from_rect<1>(context_.interop_helper_.all_processes_),
    LegionRuntime::HighLevel::TaskArgument(0, 0),
    arg_map);

  get_numbers_of_cells_launcher.tag = MAPPER_FORCE_RANK_MATCH;

  FutureMap fm1 = runtime->execute_index_space(context, 
      get_numbers_of_cells_launcher);

  legion_dpd::partitioned_unstructured cells_part;
  legion_dpd::partitioned_unstructured vertices_part;

  size_t total_num_cells=0;
  size_t total_num_ghost_cells=0;
  std::vector<size_t> cells_primary_start_id;
  std::vector<size_t> cells_num_shared;
  std::vector<size_t> cells_num_ghosts;
  std::vector<size_t> cells_num_exclusive;
  std::vector<size_t> num_vertex_conns;

  size_t total_num_vertices=0;
  size_t total_num_ghost_vertices=0;
  std::vector<size_t> vert_primary_start_id;
  std::vector<size_t> vert_num_shared;
  std::vector<size_t> vert_num_ghosts;
  std::vector<size_t> vert_num_exclusive;

  //read dimension information from  get_numbers_of_cells task
  for (size_t i = 0; i < num_ranks; i++) {
    std::cout << "about to call get_results" << std::endl;
    flecsi::execution::sprint::parts received =
      fm1.get_result<flecsi::execution::sprint::parts>(
      DomainPoint::from_point<1>(LegionRuntime::Arrays::make_point(i)));

    cells_primary_start_id.push_back(total_num_cells);
    total_num_cells += received.primary_cells;
    total_num_ghost_cells += received.ghost_cells;
    cells_num_shared.push_back(received.shared_cells);
    cells_num_ghosts.push_back(received.ghost_cells);
    cells_num_exclusive.push_back(received.exclusive_cells);

    cells_part.count_map[i] = received.primary_cells;

    vert_primary_start_id.push_back(total_num_vertices);
    total_num_vertices += received.primary_vertices;
    total_num_ghost_vertices += received.ghost_vertices;
    vert_num_shared.push_back(received.shared_vertices);
    vert_num_ghosts.push_back(received.ghost_vertices);
    vert_num_exclusive.push_back(received.exclusive_vertices);
    num_vertex_conns.push_back(received.vertex_conns);

    vertices_part.count_map[i] = received.primary_vertices;

  }//end for


  // create a field space to store cells id
  FieldSpace cells_fs = runtime->create_field_space(context);
  { 
    FieldAllocator allocator = runtime->create_field_allocator(context,
                                             cells_fs);
    allocator.allocate_field(sizeof(size_t), fid_t.fid_data);
//TOFIX
    allocator.allocate_field(sizeof(legion_dpd::ptr_count),
                      legion_dpd::connectivity_field_id(2, 0));
    allocator.allocate_field(sizeof(Point<2>),fid_t.fid_point );
  }

   //create global IS fnd LR for Vertices

  FieldSpace vertices_fs = runtime->create_field_space(context);
  {
    FieldAllocator allocator = runtime->create_field_allocator(context,
                                             vertices_fs);
    allocator.allocate_field(sizeof(size_t), fid_t.fid_data);
    allocator.allocate_field(sizeof(Point<2>), fid_t.fid_point);
  } 


  //Data compaction: creating an index space with the size of global IS + ghost
  //partition of the global IS. we need it to create partitioning that will
  //be a combination of primary+ghost part

  //Cells:
  
  Rect<2> expanded_cells_bounds = Rect<2>(Point<2>::ZEROES(),
        make_point(num_ranks,total_num_cells));
  Domain expanded_cells_dom(Domain::from_rect<2>(expanded_cells_bounds));
  IndexSpace expanded_cells_is = runtime->create_index_space(context,
        expanded_cells_dom);

  LogicalRegion expanded_cells_lr = runtime->create_logical_region(context,
        expanded_cells_is, cells_fs);

  runtime->attach_name(expanded_cells_lr, "expanded cells logical region");


  //Vertices

  Rect<2> expanded_vertices_bounds = Rect<2>(Point<2>::ZEROES(),
        make_point(num_ranks,total_num_vertices));
  Domain expanded_vertices_dom(Domain::from_rect<2>(expanded_vertices_bounds));
  IndexSpace expanded_vertices_is = runtime->create_index_space(context,
        expanded_vertices_dom);

  LogicalRegion expanded_vertices_lr = runtime->create_logical_region(context,
        expanded_vertices_is, vertices_fs);

  runtime->attach_name(expanded_vertices_lr, "expanded_vertices LR");

  //partition expanded_cells by number of mpi ranks
  Domain cells_launch_domain;
  IndexPartition expanded_cells_ip;
  {
    Rect<2> colorBounds(Point<2>::ZEROES(), make_point(num_ranks-1,0));
    Domain colorDomain = Domain::from_rect<2>(colorBounds);
    cells_launch_domain=colorDomain;
    Point<2> inc = make_point(1,total_num_cells);
    Blockify<2> coloring(inc);   
    expanded_cells_ip = runtime->create_index_partition( context,
      expanded_cells_is, coloring, 0);
  }//end scope
  LogicalPartition expanded_cells_lp = runtime->get_logical_partition(context,
           expanded_cells_lr, expanded_cells_ip); 

  //partition expanded_vertices by number of mpi ranks  
 
  Domain vertices_launch_domain;
  IndexPartition expanded_vertices_ip;
  {
    Rect<2> colorBounds(Point<2>::ZEROES(), make_point(num_ranks-1,0));
    Domain colorDomain = Domain::from_rect<2>(colorBounds);
    vertices_launch_domain=colorDomain;
    Point<2> inc = make_point(1,total_num_vertices);
    Blockify<2> coloring(inc);
    expanded_vertices_ip = runtime->create_index_partition( context,
      expanded_vertices_is, coloring, 0);
  }//end scope
  LogicalPartition expanded_vertices_lp = runtime->get_logical_partition(
    context, expanded_vertices_lr, expanded_vertices_ip);


 
  //call an index task that fills expanded_cells and expended vertices with ids
  LegionRuntime::HighLevel::IndexLauncher fill_expanded_lr_launcher(
    task_ids_t::instance().fill_expanded_lr_task_id,
    cells_launch_domain,
    LegionRuntime::HighLevel::TaskArgument(nullptr,0), arg_map);

  fill_expanded_lr_launcher.tag = MAPPER_TWO_D_RANK_PARTITIONING_LAUNCH;

  fill_expanded_lr_launcher.add_region_requirement(
    RegionRequirement(expanded_cells_lp, 0/*projection ID*/,
      WRITE_DISCARD, EXCLUSIVE, expanded_cells_lr))
                            .add_field(fid_t.fid_data)
                            .add_field(fid_t.fid_point);

  fill_expanded_lr_launcher.add_region_requirement(
    RegionRequirement(expanded_vertices_lp, 0/*projection ID*/,
      WRITE_DISCARD, EXCLUSIVE, expanded_vertices_lr))
                            .add_field(fid_t.fid_data)
                            .add_field(fid_t.fid_point);

  FutureMap fm3 = runtime->execute_index_space( context,
      fill_expanded_lr_launcher);

  fm3.wait_all_results();  


  LegionRuntime::HighLevel::IndexLauncher ghost_access_launcher(
  task_ids_t::instance().ghost_access_task_id,
  cells_launch_domain,
  LegionRuntime::HighLevel::TaskArgument(0, 0),
  arg_map);

  ghost_access_launcher.tag = MAPPER_TWO_D_RANK_PARTITIONING_LAUNCH;

  ghost_access_launcher.add_region_requirement(
  RegionRequirement(expanded_cells_lr,
                    READ_ONLY, SIMULTANEOUS, expanded_cells_lr))
                       .add_field(fid_t.fid_data);

  ghost_access_launcher.add_region_requirement(
  RegionRequirement(expanded_cells_lp, 0/*projection ID*/,
                    READ_ONLY, SIMULTANEOUS, expanded_cells_lr))
                       .add_field(fid_t.fid_data)
                       .add_field(fid_t.fid_point);

  MustEpochLauncher must_epoch_launcher;
  must_epoch_launcher.add_index_task(ghost_access_launcher);

  FutureMap fm7 = runtime->execute_must_epoch(context,must_epoch_launcher);
  fm7.wait_all_results();


  //TOFIX: free all lr physical regions is
  runtime->destroy_index_partition(context,expanded_cells_ip);
  runtime->destroy_index_partition(context, expanded_vertices_ip);
  runtime->destroy_logical_partition(context, expanded_cells_lp);
  runtime->destroy_logical_partition(context, expanded_vertices_lp);
  runtime->destroy_logical_region(context, expanded_vertices_lr);
  runtime->destroy_logical_region(context, expanded_cells_lr);
  runtime->destroy_field_space(context, vertices_fs);
  runtime->destroy_field_space(context, cells_fs);
  runtime->destroy_index_space(context, expanded_cells_is);
  runtime->destroy_index_space(context, expanded_vertices_is);

} // driver

} // namespace execution
} // namespace flecsi

#endif // flecsi_sprint_h

/*~-------------------------------------------------------------------------~-*
 * Formatting options for vim.
 * vim: set tabstop=2 shiftwidth=2 expandtab :
 *~-------------------------------------------------------------------------~-*/
