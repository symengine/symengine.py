from __future__ import print_function
from os import getenv, path, makedirs
import os
import subprocess
import sys

# Make sure the system has the right Python version.
if sys.version_info[:2] < (2, 7):
    print("SymEngine requires Python 2.7 or newer. "
          "Python %d.%d detected" % sys.version_info[:2])
    sys.exit(-1)

from skbuild import setup

# cmake_opts = [("PYTHON_BIN", sys.executable),
#               ("CMAKE_INSTALL_RPATH_USE_LINK_PATH", "yes")]
# cmake_build_type = ["Release"]

## global_user_options = [
##     ('symengine-dir=', None,
##      'path to symengine installation or build directory'),
##     ('generator=', None, 'cmake build generator'),
##     ('build-type=', None, 'build type: Release or Debug'),
##     ('define=', 'D',
##      'options to cmake <var>:<type>=<value>'),
## ]

long_description = '''
SymEngine is a standalone fast C++ symbolic manipulation library.
Optional thin Python wrappers (SymEngine) allow easy usage from Python and
integration with SymPy and Sage.'''

setup(name="symengine",
      version="0.2.1.dev",
      description="Python library providing wrappers to SymEngine",
      setup_requires=['cython>=0.19.1'],
      long_description=long_description,
      author="SymEngine development team",
      author_email="symengine@googlegroups.com",
      license="MIT",
      url="https://github.com/symengine/symengine.py",
      classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        ]
      )
