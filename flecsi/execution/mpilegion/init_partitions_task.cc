/*~--------------------------------------------------------------------------~*
 *  @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
 * /@@/////  /@@          @@////@@ @@////// /@@
 * /@@       /@@  @@@@@  @@    // /@@       /@@
 * /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
 * /@@////   /@@/@@@@@@@/@@       ////////@@/@@
 * /@@       /@@/@@//// //@@    @@       /@@/@@
 * /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
 * //       ///  //////   //////  ////////  //
 *
 * Copyright (c) 2016 Los Alamos National Laboratory, LLC
 * All rights reserved
 *~--------------------------------------------------------------------------~*/

#include <iostream>

#include "flecsi/execution/context.h"
#include "flecsi/partition/index_partition.h"
#include "flecsi/execution/mpilegion/init_partitions_task.h"

#include "flecsi/execution/legion/dpd.h"

namespace flecsi {
namespace execution {
namespace sprint {

parts
get_numbers_of_cells_task(
  const Legion::Task *task, 
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx, Legion::HighLevelRuntime *runtime) 
{
  struct parts partitions; 
  using index_partition_t = flecsi::dmp::index_partition__<size_t>;
  using field_id = LegionRuntime::HighLevel::FieldID;

  flecsi::execution::context_t & context_ =
    flecsi::execution::context_t::instance();
  //getting cells partitioning info from MPI
  index_partition_t ip_cells =
    context_.interop_helper_.data_storage_[0];
  
  //getting vertices partitioning info from MPI
  index_partition_t ip_vertices =
    context_.interop_helper_.data_storage_[1];

  std::vector<std::pair<size_t, size_t>> raw_cell_vertex_conns =
      context_.interop_helper_.data_storage_[2];  
#if 0
  size_t rank; 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout << "in get_numbers_of_cells native legion task rank is " <<
       rank <<  std::endl;

  for(size_t i=0; i<ip.primary.size(); i++) {
    auto element = ip.primary[i];
    std::cout << " Found ghost elemment: " << element <<
        " on rank: " << rank << std::endl;
  }
    
  for(size_t i=0; i< ip.exclusive.size();i++ ) {
    auto element = ip.exclusive[i];
    std::cout << " Found exclusive elemment: " << element << 
        " on rank: " << rank << std::endl;
  }
  
  for(size_t i=0; i< ip.shared.size(); i++) {
    auto element = ip.shared_id(i);
    std::cout << " Found shared elemment: " << element << 
        " on rank: " << rank << std::endl;
  } 

  for(size_t i=0; i<ip.ghost.size(); i++) {
    auto element = ip.ghost_id(i);
    std::cout << " Found ghost elemment: " << element << 
        " on rank: " << rank << std::endl;
  }

#endif
  
  partitions.primary_cells = ip_cells.primary.size();
  partitions.exclusive_cells = ip_cells.exclusive.size();
  partitions.shared_cells = ip_cells.shared.size();
  partitions.ghost_cells = ip_cells.ghost.size();

  partitions.primary_vertices = ip_vertices.primary.size();
  partitions.exclusive_vertices = ip_vertices.exclusive.size();
  partitions.shared_vertices = ip_vertices.shared.size();
  partitions.ghost_vertices = ip_vertices.ghost.size();

  partitions.vertex_conns = raw_cell_vertex_conns.size();

  std::cout << "about to return partitions (primary,exclusive,shared,ghost) ("
            << partitions.primary_cells << "," 
            <<partitions.exclusive_cells << "," << partitions.shared_cells <<
             "," << partitions.ghost_cells << ")" << std::endl;

  return partitions; 
}//get_numbers_of_cells_task


void
fill_expanded_lr_task(
  const Legion::Task *task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx, Legion::HighLevelRuntime *runtime
)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 2);
  assert(task->regions[1].privilege_fields.size() == 2);
  std::cout << "Here I am in fill_expanded_lr_task" << std::endl;
  
  using index_partition_t = flecsi::dmp::index_partition__<size_t>;
  using field_id = LegionRuntime::HighLevel::FieldID;
  
  flecsi::execution::context_t & context_ =
    flecsi::execution::context_t::instance();
  index_partition_t ip_cells =
    context_.interop_helper_.data_storage_[0];
  index_partition_t ip_vert =
    context_.interop_helper_.data_storage_[1];

  field_ids_t & fid_t = field_ids_t::instance();

  //cells:
  LegionRuntime::HighLevel::LogicalRegion lr_cells =
      regions[0].get_logical_region(); 
  LegionRuntime::HighLevel::IndexSpace is_cells = lr_cells.get_index_space();
  
  field_id fid_cell = *(task->regions[0].privilege_fields.begin());
  LegionRuntime::Accessor::RegionAccessor<
    LegionRuntime::Accessor::AccessorType::Generic, size_t>  acc_cells =
    regions[0].get_field_accessor(fid_cell).typeify<size_t>();

  LegionRuntime::Accessor::RegionAccessor<
    LegionRuntime::Accessor::AccessorType::Generic, Point<2> > 
    acc_ghost_cells =
    regions[0].get_field_accessor(fid_t.fid_point).typeify<Point<2> >();
 
  Domain cells_dom = runtime->get_index_space_domain(ctx, is_cells);
  Rect<2> cells_rect = cells_dom.get_rect<2>();

  GenericPointInRectIterator<2> pir(cells_rect);
   
  for (auto primary_cell : ip_cells.primary) {
    size_t id =primary_cell;
    acc_cells.write(DomainPoint::from_point<2>(pir.p), id);
    pir++;
  }
  for (auto ghost_cell : ip_cells.ghost) {
    size_t shard = ghost_cell.rank;
    size_t id =ghost_cell.offset;
    acc_ghost_cells.write(DomainPoint::from_point<2>(pir.p),
        make_point(shard,id));
    pir++;
  }

  //vertices
  LegionRuntime::HighLevel::LogicalRegion lr_vert =
      regions[1].get_logical_region(); 
  LegionRuntime::HighLevel::IndexSpace is_vert = lr_vert.get_index_space();

  field_id fid_vert = *(task->regions[1].privilege_fields.begin());
  LegionRuntime::Accessor::RegionAccessor
    <LegionRuntime::Accessor::AccessorType::Generic, size_t>  acc_vert =
    regions[1].get_field_accessor(fid_vert).typeify<size_t>();

   LegionRuntime::Accessor::RegionAccessor<
    LegionRuntime::Accessor::AccessorType::Generic, Point<2> >
    acc_ghost_vert =
    regions[1].get_field_accessor(fid_t.fid_point).typeify<Point<2> >();

  Domain vert_dom = runtime->get_index_space_domain(ctx, is_vert);
  Rect<2> vert_rect = vert_dom.get_rect<2>();

  GenericPointInRectIterator<2> pir_vert(vert_rect);

  for (auto primary_vert : ip_vert.primary) {
    size_t id =primary_vert;
    acc_vert.write(DomainPoint::from_point<2>(pir_vert.p), id);
    pir_vert++;
  }
  for (auto ghost_vert : ip_vert.ghost) {
    size_t shard = ghost_vert.rank;
    size_t id =ghost_vert.offset;
    acc_ghost_vert.write(DomainPoint::from_point<2>(pir_vert.p), 
        make_point(shard, id));
    pir_vert++;
  }
}//fill_expanded_lr_task




void
init_raw_conn_task(
  const Legion::Task *task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx, Legion::HighLevelRuntime *runtime
)
{
  flecsi::execution::context_t & context_ =
    flecsi::execution::context_t::instance();

  field_ids_t & fid_t = field_ids_t::instance();

  std::vector<std::pair<size_t, size_t>> raw_cell_vertex_conns =
      context_.interop_helper_.data_storage_[2];

  LogicalRegion raw_conn_lr = regions[0].get_logical_region();
  IndexSpace raw_conn_is = raw_conn_lr.get_index_space();

  auto raw_conn_ac = 
    regions[0].get_field_accessor(fid_t.fid_entity_pair).
    typeify<std::pair<size_t, size_t>>();

  size_t i = 0;
  IndexIterator raw_conn_itr(runtime, ctx, raw_conn_is);
  while(raw_conn_itr.has_next()){
    ptr_t ptr = raw_conn_itr.next();
    raw_conn_ac.write(ptr, raw_cell_vertex_conns[i++]);
  }
}

template<class Type>
static size_t archiveScalar(Type scalar, void* bit_stream)
{
    memcpy(bit_stream, (void*)(&scalar), sizeof(Type));
    return sizeof(Type);
}

template<class Type>
static size_t archiveVector(std::vector<Type> vec, void* bit_stream)
{
    unsigned char *serialized = (unsigned char*)(bit_stream) ;

    size_t size_size = archiveScalar(vec.size(), (void*)serialized);
    serialized += size_size;

    size_t vec_size = vec.size() * sizeof(Type);
    memcpy((void*)serialized, (void*)vec.data(), vec_size);

    return size_size + vec_size;
}

void SPMDArgsSerializer::archive(SPMDArgs* spmd_args)
{
    assert(spmd_args != nullptr);

    bit_stream_size = sizeof(PhaseBarrier) + sizeof(size_t)
            + spmd_args->masters_pbarriers.size() * sizeof(PhaseBarrier);
    bit_stream = malloc(bit_stream_size);
    free_bit_stream = true;

    unsigned char *serialized = (unsigned char*)(bit_stream);

    size_t stream_size = 0;
    stream_size += archiveScalar(spmd_args->pbarrier_as_master, (void*)(serialized+stream_size));
    stream_size += archiveVector(spmd_args->masters_pbarriers, (void*)(serialized+stream_size));

    assert(stream_size == bit_stream_size);
}

template<class Type>
static size_t restoreScalar(Type* scalar, void* bit_stream)
{
    memcpy((void*)scalar, bit_stream, sizeof(Type));
    return sizeof(Type);
}

template<class Type>
static size_t restoreVector(std::vector<Type>* vec, void* bit_stream)
{
    unsigned char *serialized = (unsigned char*)(bit_stream) ;

    size_t n_entries;
    size_t size_size = restoreScalar(&n_entries, (void*)serialized);
    serialized += size_size;

    vec->resize(n_entries);
    size_t vec_size = n_entries * sizeof(Type);
    memcpy((void*)vec->data(), (void*)serialized, vec_size);

    return size_size + vec_size;
}

void SPMDArgsSerializer::restore(SPMDArgs* spmd_args)
{
    assert(spmd_args != nullptr);
    assert(bit_stream != nullptr);

    unsigned char *serialized_args = (unsigned char *) bit_stream;

    bit_stream_size = 0;
    bit_stream_size += restoreScalar(&(spmd_args->pbarrier_as_master), (void*)(serialized_args + bit_stream_size));
    bit_stream_size += restoreVector(&(spmd_args->masters_pbarriers), (void*)(serialized_args + bit_stream_size));
}

void* ArgsSerializer::getBitStream()
{
    return bit_stream;
}

size_t ArgsSerializer::getBitStreamSize()
{
    return bit_stream_size;
}

void ArgsSerializer::setBitStream(void* stream)
{
    bit_stream = stream;
};

void
ghost_access_task(
  const Legion::Task *task,
  const std::vector<Legion::PhysicalRegion> & regions,
  Legion::Context ctx, Legion::HighLevelRuntime *runtime
)
{
  using generic_type = LegionRuntime::Accessor::AccessorType::Generic;
  using field_id = LegionRuntime::HighLevel::FieldID;
  using index_partition_t = flecsi::dmp::index_partition__<size_t>;

  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  size_t cells_start_id[num_ranks];

  //assert(task->local_arglen >= sizeof(cells_start_id));
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 2);
  assert(task->index_point.get_dim() == 2);
  const int my_rank = task->index_point.point_data[0];

  std::cout<< "inside ghost_access_task, my rank = " << my_rank <<std::endl;

  flecsi::execution::context_t & context_ =
    flecsi::execution::context_t::instance();
  index_partition_t ip_cells =
    context_.interop_helper_.data_storage_[0];
  index_partition_t ip_vert =
    context_.interop_helper_.data_storage_[1];

#if 0
  SPMDArgs args;
  SPMDArgsSerializer args_serializer;
  args_serializer.setBitStream(task->local_args);
  args_serializer.restore(&args);
#endif

  LegionRuntime::HighLevel::LogicalRegion expanded_cells_lr =
     regions[0].get_logical_region();
  LegionRuntime::HighLevel::IndexSpace expanded_cells_is = 
      expanded_cells_lr.get_index_space();
  field_id expanded_cells_fid = *(task->regions[0].privilege_fields.begin());

  LegionRuntime::HighLevel::LogicalRegion expanded_cells_part_lr =
     regions[1].get_logical_region();
  LegionRuntime::HighLevel::IndexSpace expanded_cells_part_is = 
      expanded_cells_part_lr.get_index_space();
  field_id expanded_cells_part_fid =
      *(task->regions[1].privilege_fields.begin());

  //partition cells_part_lr into primary, ghost, exclusive and shared

  IndexPartition primary_cells_ip;
  {
    Rect<2> colorBounds(Point<2>::ZEROES(), make_point(0,1));
    Point<2> inc = make_point(1,ip_cells.primary.size());    
    Blockify<2> coloring(inc);
    primary_cells_ip = runtime->create_index_partition( ctx,
      expanded_cells_part_is, coloring, 0);
  }  
  LogicalPartition primary_cells_lp=runtime->get_logical_partition(ctx,
           expanded_cells_part_lr, primary_cells_ip);

   

#if 0
  for (int cycle = 0; cycle < 2; cycle++) {

    // phase 1 masters update their halo regions; slaves may not access data

    // as master

    {
      AcquireLauncher acquire_launcher(lr_shared, lr_shared, regions[0]);
      acquire_launcher.add_field(fid_shared);
      acquire_launcher.add_wait_barrier(args.pbarrier_as_master);                     // phase 1
      runtime->issue_acquire(ctx, acquire_launcher);

      // master writes to data
      std::cout << my_rank << " as master writes data; phase 1 of cycle " <<
        cycle << std::endl;

      ReleaseLauncher release_launcher(lr_shared, lr_shared, regions[0]);
      release_launcher.add_field(fid_shared);
      release_launcher.add_arrival_barrier(args.pbarrier_as_master);                  // phase 2
      runtime->issue_release(ctx, release_launcher);
      args.pbarrier_as_master =
            runtime->advance_phase_barrier(ctx, args.pbarrier_as_master);             // phase 2
    }

    // as slave

    for (int master=0; master < args.masters_pbarriers.size(); master++) {
        args.masters_pbarriers[master].arrive(1);                                     // phase 2
        args.masters_pbarriers[master] =
          runtime->advance_phase_barrier(ctx, args.masters_pbarriers[master]);  // phase 2
    }

    // phase 2 slaves can read data; masters may not write to data

    // as master

    args.pbarrier_as_master.arrive(1);                                                // phase cycle + 1
    args.pbarrier_as_master =
            runtime->advance_phase_barrier(ctx, args.pbarrier_as_master);             // phase cycle + 1

    // as slave

    {
      AcquireLauncher acquire_launcher(lr_ghost, lr_ghost, regions[1]);
      acquire_launcher.add_field(fid_ghost);
      for (int master=0; master < args.masters_pbarriers.size(); master++) {
          acquire_launcher.add_wait_barrier(args.masters_pbarriers[master]);            // phase 2
      } // no knowledge of which master has which point in ghost: 
        //so, wait for all
      runtime->issue_acquire(ctx, acquire_launcher);

      // slave reads data
      std::cout << my_rank << " as slave reads data; phase 2 of cycle " <<
        cycle << std::endl;
      RegionRequirement ghost_req(lr_ghost, READ_ONLY, EXCLUSIVE, lr_ghost);
      ghost_req.add_field(fid_ghost);
      InlineLauncher ghost_launcher(ghost_req);
      PhysicalRegion pregion_ghost = runtime->map_region(ctx, ghost_launcher);
      LegionRuntime::Accessor::RegionAccessor<generic_type, size_t>
        acc_ghost= regions[1].get_field_accessor(fid_ghost).typeify<size_t>();
      while(itr_ghost.has_next()){
        ptr_t ptr = itr_ghost.next();
        std::cout << my_rank << " reads " << acc_ghost.read(ptr) << " at " <<
          ptr.value << std::endl;
       }

      ReleaseLauncher release_launcher(lr_ghost, lr_ghost, regions[1]);
      release_launcher.add_field(fid_ghost);
      for (int master=0; master < args.masters_pbarriers.size(); master++) {
          release_launcher.add_arrival_barrier(args.masters_pbarriers[master]);         // phase cycle + 1
          args.masters_pbarriers[master] =
            runtime->advance_phase_barrier(ctx, args.masters_pbarriers[master]);  // phase cycle + 1
      }
      runtime->issue_release(ctx, release_launcher);
    }
  } // cycle


/*
    LegionRuntime::Accessor::RegionAccessor<generic_type, size_t>
      acc_shared= regions[0].get_field_accessor(fid_shared).typeify<size_t>();


*/

#endif
  std::cout << "test ghost access ... passed"
  << std::endl;
}//ghost_access_task

} // namespace sprint
} // namespace execution
} // namespace flecsi

/*~-------------------------------------------------------------------------~-*
 * Formatting options for vim.
 * vim: set tabstop=2 shiftwidth=2 expandtab :
 *~-------------------------------------------------------------------------~-*/

