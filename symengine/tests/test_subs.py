import unittest

from symengine.test_utilities import raises
from symengine import Symbol, sin, cos, sqrt, Add, function_symbol, have_numpy, log


def test_basic():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    e = x+y+z
    assert e.subs({x: y, z: y}) == 3*y


def test_sin():
    x = Symbol("x")
    y = Symbol("y")
    e = sin(x)
    assert e.subs({x: y}) == sin(y)
    assert e.subs({x: y}) != sin(x)

    e = cos(x)
    assert e.subs({x: 0}) == 1
    assert e.subs(x, 0) == 1


def test_subs_exception():
    x = Symbol("x")
    expr = sin(log(x))
    raises(RuntimeError, lambda: expr.subs({x: 0}))


def test_args():
    x = Symbol("x")
    e = cos(x)
    raises(TypeError, lambda: e.subs(x, 0, 3))


def test_f():
    x = Symbol("x")
    y = Symbol("y")
    f = function_symbol("f", x)
    g = function_symbol("g", x)
    assert f.subs({function_symbol("f", x): function_symbol("g", x)}) == g
    assert ((f+g).subs({function_symbol("f", x): function_symbol("g", x)}) ==
            2*g)

    e = (f+x)**3
    assert e.subs({f: y}) == (x+y)**3
    e = e.expand()
    assert e.subs({f: y}) == ((x+y)**3).expand()


def test_msubs():
    x = Symbol("x")
    y = Symbol("y")
    f = function_symbol("f", x)
    assert f.msubs({f: y}) == y
    assert f.diff(x).msubs({f: y}) == f.diff(x)


def test_xreplace():
    x = Symbol("x")
    y = Symbol("y")
    f = sin(cos(x))
    assert f.xreplace({x: y}) == sin(cos(y))


def test_replace():
    x = Symbol("x")
    y = Symbol("y")
    f = function_symbol("f", x)
    e = sin(x) + cos(x)

    # type -> type
    assert e.replace(sin, cos) == 2*cos(x)
    # type -> callable
    assert e.replace(sin, lambda arg: sin(2*arg)) == sin(2*x) + cos(x)
    # expression -> expression
    assert sin(cos(x)).replace(cos(x), y) == sin(y)
    assert (x + y).replace(x, 2) == y + 2
    # expression -> callable
    assert sin(x).replace(x, lambda: y) == sin(y)
    # callable -> callable
    assert (x**2 + sin(x)).replace(lambda a: a.is_Pow,
                                   lambda a: a.args[0]) == x + sin(x)
    assert f.replace(lambda a: a == x, lambda a: y) == \
        function_symbol("f", y)

    # replacing x with sin(x) must not apply again to the new sin(x)
    assert sin(x).replace(sin, lambda arg: sin(arg)**2) == sin(x)**2
    assert sin(sin(x)).replace(sin, lambda arg: sin(2*arg)) == \
        sin(2*sin(2*x))

    # map=True also returns the replacements
    r, m = e.replace(sin, cos, map=True)
    assert r == 2*cos(x)
    assert m == {sin(x): cos(x)}

    # simultaneous=False allows replaced expressions to match again
    g = 2*sin(x)
    assert g.replace(lambda a: a == sin(x), lambda a: a/2,
                     simultaneous=True) == sin(x)
    assert g.replace(lambda a: a == sin(x), lambda a: a/2,
                     simultaneous=False) == sin(x)/2

    # replace is not xreplace
    raises(TypeError, lambda: e.replace({x: y}))
    raises(TypeError, lambda: e.replace(sin, 2))
    raises(TypeError, lambda: e.replace(x, "a"))
    raises(TypeError, lambda: e.replace(lambda a: True, 2))


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_float32():
    import numpy as np
    x = Symbol("x")
    expr = x * 2
    assert expr.subs({x: np.float32(2)}) == 4.0


@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_float16():
    import numpy as np
    x = Symbol("x")
    expr = x * 2
    assert expr.subs({x: np.float16(2)}) == 4.0
