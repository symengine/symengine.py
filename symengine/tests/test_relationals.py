from symengine.utilities import raises
from symengine import (Symbol, sympify, Eq, Ne, Lt, Le, Ge, Gt, sqrt, pi)

from unittest.case import SkipTest

try:
    import sympy
    HAVE_SYMPY = True
except ImportError:
    HAVE_SYMPY = False


def assert_equal(x, y):
    """Asserts that x and y are equal. This will test Equality, Unequality, LE, and GE classes."""
    assert bool(Eq(x, y))
    assert not bool(Ne(x, y))
    assert bool(Ge(x, y))
    assert bool(Le(x, y))


def assert_not_equal(x, y):
    """Asserts that x and y are not equal. This will test Equality and Unequality"""
    assert not bool(Eq(x, y))
    assert bool(Ne(x, y))


def assert_less_than(x, y):
    """Asserts that x is less than y. This will test Le, Lt, Ge, Gt classes."""
    assert bool(Le(x, y))
    assert bool(Lt(x, y))
    assert not bool(Ge(x, y))
    assert not bool(Gt(x, y))


def assert_greater_than(x, y):
    """Asserts that x is greater than y. This will test Le, Lt, Ge, Gt classes."""
    assert not bool(Le(x, y))
    assert not bool(Lt(x, y))
    assert bool(Ge(x, y))
    assert bool(Gt(x, y))


def test_equals_constants_easy():
    assert_equal(3, 3)
    assert_equal(4, 2 ** 2)


def test_equals_constants_hard():
    # Short and long are symbolically equivalent, but sufficiently different in form that expand() does not
    # catch it. Ideally, our equality should still catch these, but until symengine supports as robust simplification as
    # sympy, we can forgive failing, as long as it raises a ValueError
    short = sympify('(3/2)*sqrt(11 + sqrt(21))')
    long = sympify('sqrt((33/8 + (1/24)*sqrt(27)*sqrt(63))**2 + ((3/8)*sqrt(27) + (-1/8)*sqrt(63))**2)')
    assert_equal(short, short)
    assert_equal(long, long)
    if HAVE_SYMPY:
        assert_equal(short, long)
    else:
        raises(ValueError, lambda: bool(Eq(short, long)))


def test_not_equals_constants():
    assert_not_equal(3, 4)
    assert_not_equal(4, 4 - .000000001)


def test_equals_symbols():
    x = Symbol("x")
    y = Symbol("y")
    assert_equal(x, x)
    assert_equal(x ** 2, x * x)
    assert_equal(x * y, y * x)


def test_not_equals_symbols():
    x = Symbol("x")
    y = Symbol("y")
    assert_not_equal(x, x + 1)
    assert_not_equal(x ** 2, x ** 2 + 1)
    assert_not_equal(x * y, y * x + 1)


def test_not_equals_symbols_raise_typeerror():
    x = Symbol("x")
    y = Symbol("y")
    raises(TypeError, lambda: bool(Eq(x, 1)))
    raises(TypeError, lambda: bool(Eq(x, y)))
    raises(TypeError, lambda: bool(Eq(x ** 2, x)))


def test_less_than_constants_easy():
    assert_less_than(1, 2)
    assert_less_than(-1, 1)


def test_less_than_constants_hard():
    # Each of the below pairs are distinct numbers, with the one on the left less than the one on the right.
    # Ideally, Less-than will catch this when evaluated, but until symengine has a more robust simplification,
    # we can forgive a failure to evaluate as long as it raises a ValueError.
    if HAVE_SYMPY:
        assert_less_than(sqrt(2), 2)
        assert_less_than(3.14, pi)
    else:
        raises(ValueError, lambda: bool(Lt(sqrt(2), 2)))
        raises(ValueError, lambda: bool(Lt(3.14, pi)))


def test_greater_than_constants():
    assert_greater_than(2, 1)
    assert_greater_than(1, -1)


def test_greater_than_constants_hard():
    # Each of the below pairs are distinct numbers, with the one on the left less than the one on the right.
    # Ideally, Greater-than will catch this when evaluated, but until symengine has a more robust simplification,
    # we can forgive a failure to evaluate as long as it raises a ValueError.
    if HAVE_SYMPY:
        assert_greater_than(2, sqrt(2))
        assert_greater_than(pi, 3.14)
    else:
        raises(ValueError, lambda: bool(Gt(2, sqrt(2))))
        raises(ValueError, lambda: bool(Gt(pi, 3.14)))


def test_less_than_raises_typeerror():
    x = Symbol("x")
    y = Symbol("y")
    raises(TypeError, lambda: bool(Lt(x, 1)))
    raises(TypeError, lambda: bool(Lt(x, y)))
    raises(TypeError, lambda: bool(Lt(x ** 2, x)))


def test_greater_than_raises_typeerror():
    x = Symbol("x")
    y = Symbol("y")
    raises(TypeError, lambda: bool(Gt(x, 1)))
    raises(TypeError, lambda: bool(Gt(x, y)))
    raises(TypeError, lambda: bool(Gt(x ** 2, x)))
