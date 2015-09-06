# SymEngine Python Wrappers

Python wrappers to the C++ library [SymEngine](https://github.com/sympy/symengine),
a fast C++ symbolic manipulation library.

Python wrappers can be installed by,

    python setup.py install

If `libsymengine` was installed at a non-standard location you can do,

    python setup.py install --symengine-dir=/path/to/symengine/install/dir

Use SymEngine from Python as follows:

    >>> from symengine import var
    >>> var("x y z")
    (x, y, z)
    >>> e = (x+y+z)**2
    >>> e.expand()
    2*x*y + 2*x*z + 2*y*z + x**2 + y**2 + z**2

You can read Python tests in `symengine/tests` to see what features are
implemented. Supported versions of Python are: 2.6, 2.7, 3.2, 3.3.
You need Cython >= 0.19.1 in order to compile the wrappers. CMake
will report at configure time if the Cython version is too old.
