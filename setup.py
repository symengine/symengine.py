from __future__ import print_function
from os import getenv, path
import subprocess
import sys

# use setuptools by default as per the official advice at:
# packaging.python.org/en/latest/current.html#packaging-tool-recommendations
use_setuptools = True
# set the environment variable USE_DISTUTILS=True to force the use of distutils
use_distutils = getenv('USE_DISTUTILS')
if use_distutils is not None:
    if use_distutils.lower() == 'true':
        use_setuptools = False
    else:
        print("Value {} for USE_DISTUTILS treated as False".\
              format(use_distutils))

from distutils.command.build import build as _build

if use_setuptools:
    try:
        from setuptools import setup
        from setuptools.command.install import install as _install
    except ImportError:
        use_setuptools = False

if not use_setuptools:
    from distutils.core import setup
    from distutils.command.install import install as _install

cmake_opts = [("PYTHON_BIN", sys.executable)]
cmake_generator = [None]
cmake_build_type = ["Release"]

def process_opts(opts):
    return ['-D'+'='.join(o) for o in opts]

class BuildWithCmake(_build):
    _build_opts = _build.user_options
    user_options = [
        ('symengine-dir=', None, 'path to symengine installation or build directory'),
        ('generator=', None, 'cmake build generator'),
        ('build-type=', None, 'build type: Release or Debug'),
        ('define=', 'D',
         'options to cmake <var>:<type>=<value>')
    ]
    user_options.extend(_build_opts)

    def initialize_options(self):
        _build.initialize_options(self)
        self.define = None
        self.symengine_dir = None
        self.generator = None
        self.build_type = "Release"

    def finalize_options(self):
        _build.finalize_options(self)
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
            cmake_opts.extend(['SymEngine_DIR', self.symengine_dir])

        if self.generator:
            cmake_generator[0] = self.generator

        cmake_build_type[0] = self.build_type

    def cmake_build(self):
        dir = path.dirname(path.realpath(__file__))
        cmake_cmd = ["cmake", dir, "-DCMAKE_BUILD_TYPE=" + cmake_build_type[0]]
        cmake_cmd.extend(process_opts(cmake_opts))
        cmake_cmd.extend(self.get_generator())
        if subprocess.call(cmake_cmd) != 0:
            raise EnvironmentError("error calling cmake")

        if subprocess.call(["cmake", "--build", ".", "--config", cmake_build_type[0]]) != 0:
            raise EnvironmentError("error building project")

    def get_generator(self):
        if cmake_generator[0]:
            return ["-G", cmake_generator[0]]
        else:
            import platform, sys
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
        # can't use super() here because _build is an old style class in 2.7
        _build.run(self)

class InstallWithCmake(_install):
    _install_opts = _install.user_options
    user_options = [
        ('symengine-dir=', None, 'path to symengine installation or build directory'),
        ('generator=', None, 'cmake build generator'),
        ('build-type=', None, 'build type: Release or Debug'),
        ('define=', 'D',
         'options to cmake <var>:<type>=<value>')
    ]
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
        if self.symengine_dir:
            cmake_opts.extend([('SymEngine_DIR', self.symengine_dir)])

        if self.generator:
            cmake_generator[0] = self.generator

        cmake_build_type[0] = self.build_type

    def cmake_install(self):
        dir = path.dirname(path.realpath(__file__))
        cmake_cmd = ["cmake", dir]
        cmake_cmd.extend(process_opts(cmake_opts))

        # CMake has to be called here to update CMAKE_INSTALL_PREFIX and PYTHON_INSTALL_PATH
        # if build and install were called separately by the user
        if subprocess.call(cmake_cmd) != 0:
            raise EnvironmentError("error calling cmake")

        if subprocess.call(["cmake", "--build", ".", "--config", cmake_build_type[0], "--target", "install"]) != 0:
            raise EnvironmentError("error installing")

        if self.compile:
            import compileall
            compileall.compile_dir(path.join(sys.prefix), "symengine")

    def run(self):
        # can't use super() here because _install is an old style class in 2.7
        _install.run(self)
        self.cmake_install()

long_description = '''
SymEngine is a standalone fast C++ symbolic manipulation library.
Optional thin Python wrappers (SymEngine) allow easy usage from Python and
integration with SymPy.'''

setup(name = "symengine",
      version = "0.1.0.dev",
      description = "Python library providing wrappers to SymEngine",
      long_description = "",
      author = "",
      author_email = "",
      license = "MIT",
      url = "https://github.com/sympy/symengine",
      cmdclass={
          'build' : BuildWithCmake,
          'install' : InstallWithCmake,
          }
  )
