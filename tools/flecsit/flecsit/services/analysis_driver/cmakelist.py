#------------------------------------------------------------------------------#
# Copyright (c) 2014 Los Alamos National Security, LLC
# All rights reserved.
#------------------------------------------------------------------------------#

from string import Template

cc_source_template = Template(
"""
/*~-------------------------------------------------------------------------~~*
 * Copyright (c) 2014 Los Alamos National Security, LLC
 * All rights reserved.
 *~-------------------------------------------------------------------------~~*/

cmake_minimum_required(${CMAKE_VERSION})
project(${PROJECT_NAME})

CMAKE_EXPORT_COMPILE_COMMANDS=1

add_executable(${PROJECT_NAME} runtime_driver.cc runtime_main.cc)


""")

#------------------------------------------------------------------------------#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#------------------------------------------------------------------------------#
