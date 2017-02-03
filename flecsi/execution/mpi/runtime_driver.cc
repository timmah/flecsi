/*~-------------------------------------------------------------------------~~*
 * Copyright (c) 2014 Los Alamos National Security, LLC
 * All rights reserved.
 *~-------------------------------------------------------------------------~~*/

///
// \file mpi/runtime_driver.cc
// \authors bergen
// \date Initial file creation: Aug 01, 2016
///

#include "flecsi/execution/mpi/runtime_driver.h"
#include "flecsi/utils/common.h"

#ifndef FLECSI_DRIVER
  #include "flecsi/execution/default_driver.h"
#else
  #include EXPAND_AND_STRINGIFY(FLECSI_DRIVER)
#endif

namespace flecsi {
namespace execution {

void mpi_runtime_driver(int argc, char ** argv) {
  driver(argc, argv);
} // mpi_runtime_driver

} // namespace execution 
} // namespace flecsi

/*~------------------------------------------------------------------------~--*
 * Formatting options for vim.
 * vim: set tabstop=2 shiftwidth=2 expandtab :
 *~------------------------------------------------------------------------~--*/
