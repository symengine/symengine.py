from __future__ import print_function

import os
import shlex
import sys

try:
    from skbuild import setup
except ImportError:
    print('scikit-build is required to build from source.', file=sys.stderr)
    print('Please run:', file=sys.stderr)
    print('', file=sys.stderr)
    print('  python -m pip install scikit-build')
    sys.exit(1)

from setuptools import find_packages

# BEGIN  
# TODO: remove hack when py27 is dropped
from skbuild.platform_specifics import windows

if sys.version_info < (3, 0):
    windows._get_msvc_compiler_env = lambda _ : {}
# END
    
long_description = '''
SymEngine is a standalone fast C++ symbolic manipulation library.
Optional thin Python wrappers (SymEngine) allow easy usage from Python and
integration with SymPy and Sage.

See https://github.com/symengine/symengine.py for information about License
and dependencies of wheels

'''

setup(name="symengine",
      version="0.3.1.dev1",
      description="Python library providing wrappers to SymEngine",
      long_description=long_description,
      packages=find_packages(),
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        ]
      )
