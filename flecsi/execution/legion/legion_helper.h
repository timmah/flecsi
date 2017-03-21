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

#ifndef flecsi_execution_legion_helper_h
#define flecsi_execution_legion_helper_h

#include <arrays.h>
#include <legion.h>

///
/// \file
/// \date Initial file creation: Nov 29, 2016
///

namespace flecsi {
namespace execution {

///
/// FIXME documentation
///
class legion_helper
{
public:

  ///
  ///
  ///
  legion_helper(
    Legion::Runtime * runtime,
    Legion::Context context
  )
  :
    runtime_(runtime),
    context_(context)
  {}

  ///
  /// Create a structured index space.
  ///
  Legion::IndexSpace
  create_index_space(
    unsigned start,
    unsigned end
  )
  {
    assert(end >= start);
    LegionRuntime::Arrays::Rect<1> rect(LegionRuntime::Arrays::Point<1>(start),
      LegionRuntime::Arrays::Point<1>(end - 0));
    return runtime_->create_index_space(context_,
      Legion::Domain::from_rect<1>(rect));  
  } // create_index_space

  ///
  /// Create a domain point from a size_t.
  ///
  Legion::DomainPoint
  domain_point(
    size_t p
  )
  {
    return Legion::DomainPoint::from_point<1>(
      LegionRuntime::Arrays::make_point(p));
  } // domain_point

  ///
  ///
  ///
  Legion::Domain
  domain_from_point(
    size_t p
  )
  {
    LegionRuntime::Arrays::Rect<1> rect(LegionRuntime::Arrays::Point<1>(p),
      LegionRuntime::Arrays::Point<1>(p - 0));
    return Legion::Domain::from_rect<1>(rect);
  } // domain_from_point

  ///
  ///
  ///
  Legion::Domain
  domain_from_rect(
    size_t start,
    size_t end
  )
  {
    LegionRuntime::Arrays::Rect<1> rect(LegionRuntime::Arrays::Point<1>(start),
      LegionRuntime::Arrays::Point<1>(end - 0));
    return Legion::Domain::from_rect<1>(rect);
  } // domain_from_rect

  ///
  /// Create an unstructured index space.
  ///
  Legion::IndexSpace
  create_index_space(
    size_t n
  )
  const
  {
    assert(n > 0);
    return runtime_->create_index_space(context_, n);  
  } // create_index_space

  ///
  ///
  ///
  Legion::FieldSpace
  create_field_space()
  const
  {
    return runtime_->create_field_space(context_);
  } // create_field_space

  ///
  ///
  ///
  Legion::FieldAllocator
  create_field_allocator(
    Legion::FieldSpace fs
  )
  const
  {
    return runtime_->create_field_allocator(context_, fs);
  } // create_field_allocator

  ///
  ///
  ///
  Legion::LogicalRegion
  create_logical_region(
    Legion::IndexSpace is,
    Legion::FieldSpace fs
  )
  const
  {
    return runtime_->create_logical_region(context_, is, fs);
  } // create_logical_region

  ///
  ///
  ///
  Legion::Domain
  get_index_space_domain(
    Legion::IndexSpace is
  )
  const
  {
    return runtime_->get_index_space_domain(context_, is);
  } // get_index_space_domain

  ///
  ///
  ///
  Legion::DomainPoint
  domain_point(
    size_t i
  )
  const
  {
    return Legion::DomainPoint::from_point<1>(
      LegionRuntime::Arrays::Point<1>(i)); 
  } // domain_point

  ///
  ///
  ///
  Legion::FutureMap
  execute_index_space(
    Legion::IndexLauncher l
  )
  const
  {
    return runtime_->execute_index_space(context_, l);
  } // execute_index_space

  ///
  ///
  ///
  Legion::IndexAllocator
  create_index_allocator(
    Legion::IndexSpace is
  )
  const
  {
    return runtime_->create_index_allocator(context_, is);
  } // create_index_allocator

  ///
  ///
  ///
  Legion::Domain
  get_domain(
    Legion::PhysicalRegion pr
  )
  const
  {
    Legion::LogicalRegion lr = pr.get_logical_region();
    Legion::IndexSpace is = lr.get_index_space();
    return runtime_->get_index_space_domain(context_, is);     
  } // get_domain
  
  ///
  /// FIXME documentation
  ///
  template<class T>
  void get_buffer(
    Legion::PhysicalRegion pr,
    T*& buf,
    size_t field = 0
  )
  const
  {
    auto ac = pr.get_field_accessor(field).typeify<T>();
    Legion::Domain domain = get_domain(pr); 
    LegionRuntime::Arrays::Rect<1> r = domain.get_rect<1>();
    LegionRuntime::Arrays::Rect<1> sr;
    LegionRuntime::Accessor::ByteOffset bo[1];
    buf = ac.template raw_rect_ptr<1>(r, sr, bo);
  } // get_buffer

  ///
  ///
  ///
  char *
  get_raw_buffer(
    Legion::PhysicalRegion pr,
    size_t field = 0
  )
  const
  {
    auto ac = pr.get_field_accessor(field).typeify<char>();
    Legion::Domain domain = get_domain(pr); 
    LegionRuntime::Arrays::Rect<1> r = domain.get_rect<1>();
    LegionRuntime::Arrays::Rect<1> sr;
    LegionRuntime::Accessor::ByteOffset bo[1];
    return ac.template raw_rect_ptr<1>(r, sr, bo);
  } // get_raw_buffer

  ///
  /// FIXME documentation
  ///
  void
  unmap_region(
    Legion::PhysicalRegion pr
  )
  const
  {
    runtime_->unmap_region(context_, pr);
  } // unmap_region

private:
  mutable Legion::Runtime* runtime_;
  mutable Legion::Context context_;
};

} // namespace execution 
} // namespace flecsi

#endif // flecsi_execution_legion_helper_h

/*~-------------------------------------------------------------------------~-*
 * Formatting options
 * vim: set tabstop=2 shiftwidth=2 expandtab :
 *~-------------------------------------------------------------------------~-*/
