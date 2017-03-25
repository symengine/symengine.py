from __future__ import print_function
from os import getenv, path, makedirs
import os
import subprocess
import sys
from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.build import build as _build

# Make sure the system has the right Python version.
if sys.version_info[:2] < (2, 7):
    print("SymEngine requires Python 2.7 or newer. "
          "Python %d.%d detected" % sys.version_info[:2])
    sys.exit(-1)

# use setuptools by default as per the official advice at:
# packaging.python.org/en/latest/current.html#packaging-tool-recommendations
use_setuptools = True
# set the environment variable USE_DISTUTILS=True to force the use of distutils
use_distutils = getenv('USE_DISTUTILS')
if use_distutils is not None:
    if use_distutils.lower() == 'true':
        use_setuptools = False
    else:
        print("Value {} for USE_DISTUTILS treated as False".
              format(use_distutils))

if use_setuptools:
    try:
        from setuptools import setup
        from setuptools.command.install import install as _install
    except ImportError:
        use_setuptools = False

if not use_setuptools:
    from distutils.core import setup
    from distutils.command.install import install as _install

cmake_opts = [("PYTHON_BIN", sys.executable),
              ("CMAKE_INSTALL_RPATH_USE_LINK_PATH", "yes")]
# added option for using cysignals from sagemath
# see https://github.com/sagemath/cysignals
# set the environment variable USE_CYSIGNALS=True to force the use of cysignals
use_cysignals = os.getenv('USE_CYSIGNALS')

if use_cysignals:
    cysignals_version = '1.4.0'
    # This directory
    dir_setup = os.path.dirname(os.path.realpath(__file__))
    # handle cysignals deps in the hard way:
    from distutils.version import LooseVersion
    try:
        import cysignals
        if cysignals.__version__ < LooseVersion(cysignals_version):
            raise ImportError
    except ImportError:
        print("Please install the cysignals package with a version >= "
              "%s" % cysignals_version)
        sys.exit(-1)

cmake_opts = [("PYTHON_BIN", sys.executable),
              ("CMAKE_INSTALL_RPATH_USE_LINK_PATH", "yes")]
cmake_generator = [None]
cmake_build_type = ["Release"]


def process_opts(opts):
    return ['-D'+'='.join(o) for o in opts]


def get_build_dir(dist):
    source_dir = path.dirname(path.realpath(__file__))
    build = dist.get_command_obj('build')
    build_ext = dist.get_command_obj('build_ext')
    return source_dir if build_ext.inplace else build.build_platlib


global_user_options = [
    ('symengine-dir=', None,
     'path to symengine installation or build directory'),
    ('generator=', None, 'cmake build generator'),
    ('build-type=', None, 'build type: Release or Debug'),
    ('define=', 'D',
     'options to cmake <var>:<type>=<value>'),
]


class BuildWithCmake(_build):
    sub_commands = [('build_ext', None)]


class BuildExtWithCmake(_build_ext):
    _build_opts = _build_ext.user_options
    user_options = list(global_user_options)
    user_options.extend(_build_opts)

    def initialize_options(self):
        _build_ext.initialize_options(self)
        self.define = None
        self.symengine_dir = None
        self.generator = None
        self.build_type = "Release"

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # The argument parsing will result in self.define being a string, but
        # it has to be a list of 2-tuples.
        # Multiple symbols can be separated with semi-colons.
        if self.define:
            defines = self.define.split(';')
            self.define = [(s.strip(), None) if '=' not in s else
                           tuple(ss.strip() for ss in s.split('='))
                           for s in defines]
            cmake_opts.extend(self.define)
        if self.symengine_dir:
            cmake_opts.extend([('SymEngine_DIR', self.symengine_dir)])

        if self.generator:
            cmake_generator[0] = self.generator

        cmake_build_type[0] = self.build_type

    def cmake_build(self):
        source_dir = path.dirname(path.realpath(__file__))
        build_dir = get_build_dir(self.distribution)
        if not path.exists(build_dir):
            makedirs(build_dir)
        if build_dir != source_dir and path.exists("CMakeCache.txt"):
            os.remove("CMakeCache.txt")

        cmake_cmd = ["cmake", source_dir,
                     "-DCMAKE_BUILD_TYPE=" + cmake_build_type[0]]
        cmake_cmd.extend(process_opts(cmake_opts))
        if not path.exists(path.join(build_dir, "CMakeCache.txt")):
            cmake_cmd.extend(self.get_generator())
        if subprocess.call(cmake_cmd, cwd=build_dir) != 0:
            raise EnvironmentError("error calling cmake")

        if subprocess.call(["cmake", "--build", ".",
                            "--config", cmake_build_type[0]],
                           cwd=build_dir) != 0:
            raise EnvironmentError("error building project")

    def get_generator(self):
        if cmake_generator[0]:
            return ["-G", cmake_generator[0]]
        else:
            import platform
            import sys
            if (platform.system() == "Windows"):
                compiler = str(self.compiler).lower()
                if ("msys" in compiler):
                    return ["-G", "MSYS Makefiles"]
                elif ("mingw" in compiler):
                    return ["-G", "MinGW Makefiles"]
                elif sys.maxsize > 2**32:
                    return ["-G", "Visual Studio 14 2015 Win64"]
                else:
                    return ["-G", "Visual Studio 14 2015"]
            return []

    def run(self):
        self.cmake_build()
        # can't use super() here because
        #  _build_ext is an old style class in 2.7
        _build_ext.run(self)


class InstallWithCmake(_install):
    _install_opts = _install.user_options
    user_options = list(global_user_options)
    user_options.extend(_install_opts)

    def initialize_options(self):
        _install.initialize_options(self)
        self.define = None
        self.symengine_dir = None
        self.generator = None
        self.build_type = "Release"

    def finalize_options(self):
        _install.finalize_options(self)
        # The argument parsing will result in self.define being a string, but
        # it has to be a list of 2-tuples.
        # Multiple symbols can be separated with semi-colons.
        if self.define:
            defines = self.define.split(';')
            self.define = [(s.strip(), None) if '=' not in s else
                           tuple(ss.strip() for ss in s.split('='))
                           for s in defines]
            cmake_opts.extend(self.define)

        cmake_build_type[0] = self.build_type
        cmake_opts.extend([('PYTHON_INSTALL_PATH', self.install_platlib)])
        cmake_opts.extend([('PYTHON_INSTALL_HEADER_PATH',
                            self.install_headers)])

    def cmake_install(self):
        source_dir = path.dirname(path.realpath(__file__))
        build_dir = get_build_dir(self.distribution)
        cmake_cmd = ["cmake", source_dir]
        cmake_cmd.extend(process_opts(cmake_opts))

        # CMake has to be called here to update PYTHON_INSTALL_PATH
        # if build and install were called separately by the user
        if subprocess.call(cmake_cmd, cwd=build_dir) != 0:
            raise EnvironmentError("error calling cmake")

        if subprocess.call(["cmake", "--build", ".",
                            "--config", cmake_build_type[0],
                            "--target", "install"],
                           cwd=build_dir) != 0:
            raise EnvironmentError("error installing")

        import compileall
        compileall.compile_dir(path.join(self.install_platlib, "symengine"))

    def run(self):
        # can't use super() here because _install is an old style class in 2.7
        _install.run(self)
        self.cmake_install()


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
      cmdclass={
          'build': BuildWithCmake,
          'build_ext': BuildExtWithCmake,
          'install': InstallWithCmake,
          },
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
