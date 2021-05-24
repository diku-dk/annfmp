#
# Copyright (C) 2013-2019 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import sys
import numpy

SOURCES_RELATIVE_PATH = "src/"

FILES_TO_BE_COMPILED = [
    "base.c", 
    "helper.c", 
    "kdtree.c", 
    "timing.c", 
    "util.c"
]
DIRS_TO_BE_INCLUDED = ["include"]

# the absolute path to the sources
current_path = os.path.dirname(os.path.abspath(__file__))
sources_abs_path = os.path.abspath(os.path.join(current_path, SOURCES_RELATIVE_PATH))

# all source files
source_files = [os.path.abspath(os.path.join(sources_abs_path, x)) for x in FILES_TO_BE_COMPILED] 
include_paths = [os.path.abspath(os.path.join(sources_abs_path, x)) for x in DIRS_TO_BE_INCLUDED]

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

if sys.version_info >= (3, 0):
    swig_opts = ['-py3']


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('annfmp/openmp', parent_package, top_path)
    
    # CPU + FLOAT
    config.add_extension("_wrapper_cpu_float", 
                        sources=["swig/cpu_float.i"] + source_files,
                        swig_opts=swig_opts,
                        include_dirs=[numpy_include] + [include_paths],
                        define_macros=[
                            ('USE_DOUBLE', 0),
                            ('TIMING', 1)
                        ],
                        libraries=['gomp', 'm'],
                        extra_compile_args=["-fopenmp", "-std=c99", '-O3', '-w'] + ['-I' + ipath for ipath in include_paths])

    # CPU + DOUBLE
    config.add_extension("_wrapper_cpu_double", 
                        sources=["swig/cpu_double.i"] + source_files,
                        swig_opts=swig_opts,
                        include_dirs=[numpy_include] + [include_paths],
                        define_macros=[
                            ('USE_DOUBLE', 1),
                            ('TIMING', 1)
                        ],
                        libraries=['gomp', 'm'],
                        extra_compile_args=["-fopenmp", "-std=c99", '-O3', '-w'] + ['-I' + ipath for ipath in include_paths])

    return config

if __name__ == '__main__':
    
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

