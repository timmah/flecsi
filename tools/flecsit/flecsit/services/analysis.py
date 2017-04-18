#------------------------------------------------------------------------------#
# Copyright (c) 2014 Los Alamos National Security, LLC
# All rights reserved.
#------------------------------------------------------------------------------#

import sys
import os

from flecsit.base import Service
from flecsit.services.analysis_driver.execute import *
from flecsit.services.analysis_driver.cmakelist import *

def dir_exists(path):
    print path
    if not os.path.exists(path):
        os.makedirs(path) 
        
def sym_exists(source,link_name):
    if not os.path.exists(link_name):
        os.symlink(source,link_name)

#------------------------------------------------------------------------------#
# Documentation handler.
#------------------------------------------------------------------------------#

class FleCSIT_Analysis(Service):

    #--------------------------------------------------------------------------#
    # Initialization.
    #--------------------------------------------------------------------------#

    def __init__(self, subparsers):

        """
        """

        # get a command-line parser
        self.parser = subparsers.add_parser('analyze',
            help='Service for static analysis.' +
                 ' With no flags, this command takes a list of' +
                 ' source files to process. The form should be' +
                 ' source:\"defines string\":\"include paths string\", e.g.,' +
                 ' foo.cc:\"-I/path/one -I/path/two\":\"-DDEF1 -DDEF2\"'
        )

        # add command-line options
        self.parser.add_argument('-v', '--verbose', action='store_true',
            help='Turn on verbose output.'
        )

        self.parser.add_argument('project', nargs='*', action='append',
            help='The files to anaylze.'
        )

        # set the callback for this sub-command
        self.parser.set_defaults(func=self.main)

    # __init__

    #--------------------------------------------------------------------------#
    # Main.
    #--------------------------------------------------------------------------#
    def main(self, build, args=None):

        """
        """

        #----------------------------------------------------------------------#
        # Process command-line arguments
        #----------------------------------------------------------------------#
        project_name = args.project[0][0]
        project_header = project_name + '.cc'
        
        # We first need to set up the directory structure for the 
        project_dir = os.getcwd() + '/' + project_name
        build_dir = project_dir + '/build'
        
        dir_exists(project_dir)
        dir_exists(build_dir)
        
        flecsi_install = os.path.realpath(__file__).partition("lib")[0].rstrip("/")
        
        flecsi_runtime = flecsi_install + '/share/flecsi/runtime'
        
        # Create symbolic links to the runtime source files
        sym_exists(flecsi_runtime+'/runtime_driver.cc', project_dir + '/runtime_driver.cc')
        sym_exists(flecsi_runtime+'/runtime_main.cc', project_dir + '/runtime_main.cc')
        
        # Create a symbolic link to the header file
        sym_exists(os.getcwd() +'/' + project_header, project_dir + '/' + project_header)
        
        # We need to put the library and include paths into the proper format for cmake
        mangled_list = build['libraries'].split(" ")
        cmake_lib_dirs = ""
        
        for tmp in mangled_list:
            if(tmp.find('-L')):
                cmake_lib_dirs += tmp.strip('-L') + ' '
                
        
        cmake_include_dirs = build['includes'].replace('-I', '')
        
        cmake_include_dirs += " " +flecsi_install + '/include'
        
        cmake_defines = build['defines']
        
        source_output = cmake_source_template.substitute(
            CMAKE_VERSION="VERSION 2.8",
            PROJECT_NAME=project_name,
            CMAKE_INCLUDE_DIRS=cmake_include_dirs,
            CMAKE_DEFINES=cmake_defines)
            
        print build['libraries']
        
        fd = open(project_dir+'/CMakeLists.txt','w')
        fd.write(source_output[1:-1])
        fd.close()    
        
                
        
        execute()

    # main

    #--------------------------------------------------------------------------#
    # Object factory for service creation.
    #--------------------------------------------------------------------------#

    class Factory:
        def create(self, subparsers): return FleCSIT_Analysis(subparsers)
    # class Factory

# class FleCSIT_Analysis

#------------------------------------------------------------------------------#
# Formatting options for emacs and vim.
# vim: set tabstop=4 shiftwidth=4 expandtab :
#------------------------------------------------------------------------------#
