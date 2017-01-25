/*~--------------------------------------------------------------------------~*
 * Copyright (c) 2015 Los Alamos National Security, LLC
 * All rights reserved.
 *~--------------------------------------------------------------------------~*/

#ifndef flecsi_data_handle_h
#define flecsi_data_handle_h

#include <iostream>

#include "flecsi/utils/common.h"
#include "flecsi/execution/context.h"
#include "flecsi/execution/execution.h"

///
// \file data-handle.h
// \authors nickm
// \date Initial file creation: Jan 25, 2017
///

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

namespace flecsi {
namespace execution {
  
void task1(double val) {
  std::cout << "Executing task1" << std::endl;
  std::cout << "val = " << val << std::endl;
} // task1

register_task(task1, loc, single);

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

  execute_task(task1, loc, single, 1.0);

} // driver

} // namespace execution
} // namespace flecsi

#endif // flecsi_data_handle_h

/*~-------------------------------------------------------------------------~-*
 * Formatting options for vim.
 * vim: set tabstop=2 shiftwidth=2 expandtab :
 *~-------------------------------------------------------------------------~-*/
