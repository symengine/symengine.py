# SymEngine Python Wrappers

Python wrappers to the C++ library [SymEngine](https://github.com/symengine/symengine),
a fast C++ symbolic manipulation library.

[![Build Status](https://travis-ci.org/symengine/symengine.py.svg)](https://travis-ci.org/symengine/symengine.py)  
[![Build status](https://ci.appveyor.com/api/projects/status/sl189l9ck3gd8qvk/branch/master?svg=true)](https://ci.appveyor.com/project/symengine/symengine-py/branch/master)

## Installation

### Pip

See License section for information about wheels

```bash
pip install symengine --user
```

### Conda package manager

```bash
conda install python-symengine -c conda-forge
```

### Build from source

Install prerequisites.

```bash
CMake       >= 3.21
Python3     >= 3.9
SymEngine   >= 0.14.0
pip
setuptools_scm           # will be automatically downloaded by pip
scikit-build-core        # will be automatically downloaded by pip
cython      >= 0.29.24   # will be automatically downloaded by pip
cython-cmake             # will be automatically downloaded by pip
```

For **SymEngine**, only a specific commit/tag (see `symengine_version.txt`) is
supported.   The latest git master branch may not work as there may be breaking
changes in **SymEngine**.

Python wrappers can be installed by,

```bash
pip install . -vv
```

If you are building SymEngine for development, you should implement
the dependencies automatically installed by pip and do,
```bash
pip install -e . -vv --no-build-isolation
```

A few additional options to scikit-build-core are mentioned here:

```bash
pip install -e . -vv --no-build-isolation
    -Ccmake.build-type=Release|Debug                                # Set build-type for multi-configuration generators like MSVC
    -Ccmake.define.SymEngine_DIR=/path/to/symengine/install/dir     # Path to SymEngine install directory or build directory
```

### Notes on Dependencies

If you intend to evaluate floating-point expressions (using **lambdify**),
you should consider linking against **LLVM**. Many users might also benefit
from linking against **FLINT**, as it is now LGPL-licensed.

In general, **sudo** is only required if you are installing to the default
prefix (`/usr/local`). We recommend specifying a custom prefix
(`--prefix=$HOME/.local`) to avoid requiring administrative privileges,
which most users can do without using **sudo**.

If you're uncomfortable specifying the prefix manually, we suggest using
**Conda** or installing the pre-built wheels via **pip** instead of building
from source.

## Verification

You can verify the installation of **SymEngine** by using the provided code
snippet in this README. This snippet ensures that the installation works as
expected and that basic functionality is available.

```python
from symengine import var
x, y, z = var('x y z')
e = (x + y + z)**2
expanded_e = e.expand()
print(expanded_e)
```
This will output:
```python
x**2 + y**2 + z**2 + 2*x*y + 2*x*z + 2*y*z
```

Note: The verification code provided above checks the functionality of
SymEngine. For additional verification specific to SymEngine, please refer to
the [official SymEngine Python bindings repository](https://github.com/symengine/symengine.py)
for further tests and examples.

## License

symengine.py is MIT licensed and uses several LGPL, BSD-3, and MIT licensed
libraries.

Licenses for the dependencies of pip wheels are as follows:

- pip wheels on Unix use **GMP** (LGPL-3.0-or-later),
  **MPFR** (LGPL-3.0-or-later), **MPC** (LGPL-3.0-or-later),
  **LLVM** (Apache-2.0), **zlib** (Zlib), **libxml2** (MIT),
  **zstd** (BSD-3-Clause), and **symengine** (MIT AND BSD-3-Clause).
- pip wheels on Windows use **MPIR** (LGPL-3.0-or-later) instead of **GMP**
  above and **pthreads-win32** (LGPL-3.0-or-later) additionally.
- **NumPy** (BSD-3-Clause) and **SymPy** (BSD-3-Clause) are optional
  dependencies.
- Sources for these binary dependencies can be found on
  [symengine-wheels](https://github.com/symengine/symengine-wheels/releases).

