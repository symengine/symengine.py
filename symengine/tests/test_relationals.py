from symengine.utilities import raises
from symengine import (Symbol, sympify, Eq)


def test_equals_constants():
    assert bool(Eq(3, 3))
    assert bool(Eq(4, 2**2))

    # Short and long are symbolically equivalent, but sufficiently different in form that expand() and evalf() does not
    # catch it. Despite the float evaluation differing ever so slightly, ideally, our equality should still catch
    # symbolically equal expressions.
    short = sympify('(3/2)*sqrt(11 + sqrt(21))')
    long = sympify('sqrt((33/8 + (1/24)*sqrt(27)*sqrt(63))**2 + ((3/8)*sqrt(27) + (-1/8)*sqrt(63))**2)')
    assert bool(Eq(short, short))
    assert bool(Eq(long, long))
    assert bool(Eq(short, long))


def test_not_equals_constants():
    assert not bool(Eq(3, 4))
    assert not bool(Eq(4, 4-.000000001))


def test_equals_symbols():
    x = Symbol("x")
    y = Symbol("y")
    assert bool(Eq(x, x))
    assert bool(Eq(x**2, x*x))
    assert bool(Eq(x*y, y*x))


def test_not_equals_symbols():
    x = Symbol("x")
    y = Symbol("y")
    assert not bool(Eq(x, x+1))
    assert not bool(Eq(x**2, x**2+1))
    assert not bool(Eq(x * y, y * x+1))


def test_not_equals_symbols_raise_typeerror():
    x = Symbol("x")
    y = Symbol("y")
    raises(TypeError, lambda: bool(Eq(x, 1)))
    raises(TypeError, lambda: bool(Eq(x, y)))
    raises(TypeError, lambda: bool(Eq(x**2, x)))