from symengine import (Integer, symbols, sin, cos, pi, E, I, oo, zoo,
    nan, Add, function_symbol, DenseMatrix, sympify, log)
from symengine.lib.symengine_wrapper import (PyNumber, PyFunction,
    sage_module, wrap_sage_function)


def test_sage_conversions():
    try:
        import sage.all as sage
    except ImportError:
        return

    x, y = sage.SR.var('x y')
    x1, y1 = symbols('x, y')

    # Symbol
    assert x1._sage_() == x
    assert x1 == sympify(x)

    # Integer
    assert Integer(12)._sage_() == sage.Integer(12)
    assert Integer(12) == sympify(sage.Integer(12))

    # Rational
    assert (Integer(1) / 2)._sage_() == sage.Integer(1) / 2
    assert Integer(1) / 2 == sympify(sage.Integer(1) / 2)

    # Operators
    assert x1 + y == x1 + y1
    assert x1 * y == x1 * y1
    assert x1 ** y == x1 ** y1
    assert x1 - y == x1 - y1
    assert x1 / y == x1 / y1

    assert x + y1 == x + y
    assert x * y1 == x * y
    # Doesn't work in Sage 6.1.1ubuntu2
    # assert x ** y1 == x ** y
    assert x - y1 == x - y
    assert x / y1 == x / y

    # Conversions
    assert (x1 + y1)._sage_() == x + y
    assert (x1 * y1)._sage_() == x * y
    assert (x1 ** y1)._sage_() == x ** y
    assert (x1 - y1)._sage_() == x - y
    assert (x1 / y1)._sage_() == x / y

    assert x1 + y1 == sympify(x + y)
    assert x1 * y1 == sympify(x * y)
    assert x1 ** y1 == sympify(x ** y)
    assert x1 - y1 == sympify(x - y)
    assert x1 / y1 == sympify(x / y)

    # Functions
    assert sin(x1) == sin(x)
    assert sin(x1)._sage_() == sage.sin(x)
    assert sin(x1) == sympify(sage.sin(x))

    assert cos(x1) == cos(x)
    assert cos(x1)._sage_() == sage.cos(x)
    assert cos(x1) == sympify(sage.cos(x))

    assert function_symbol('f', x1, y1)._sage_() == sage.function('f', x, y)
    assert (function_symbol('f', 2 * x1, x1 + y1).diff(x1)._sage_() ==
            sage.function('f', 2 * x, x + y).diff(x))

    # For the following test, sage needs to be modified
    # assert sage.sin(x) == sage.sin(x1)

    # Constants
    assert pi._sage_() == sage.pi
    assert E._sage_() == sage.e
    assert I._sage_() == sage.I
    assert oo._sage_() == sage.oo
    assert zoo._sage_() == sage.unsigned_infinity
    assert nan._sage_() == sage.NaN

    assert pi == sympify(sage.pi)
    assert E == sympify(sage.e)
    assert oo == sympify(sage.oo)
    assert zoo == sympify(sage.unsigned_infinity)
    assert nan == sympify(sage.NaN)

    # SympyConverter does not support converting the following
    # assert I == sympify(sage.I)

    # Matrix
    assert DenseMatrix(1, 2, [x1, y1])._sage_() == sage.matrix([[x, y]])

    # SympyConverter does not support converting the following
    # assert DenseMatrix(1, 2, [x1, y1]) == sympify(sage.matrix([[x, y]]))

    # Sage Number
    a = sage.Mod(2, 7)
    b = PyNumber(a, sage_module)

    a = a + 8
    b = b + 8
    assert isinstance(b, PyNumber)
    assert b._sage_() == a

    a = a + x
    b = b + x
    assert isinstance(b, Add)
    assert b._sage_() == a

    # Sage Function
    e = x1 + wrap_sage_function(sage.log_gamma(x))
    assert str(e) == "x + log_gamma(x)"
    assert isinstance(e, Add)
    assert (e + wrap_sage_function(sage.log_gamma(x)) ==
            x1 + 2*wrap_sage_function(sage.log_gamma(x)))

    f = e.subs({x1: 10})
    assert f == 10 + log(362880)

    f = e.subs({x1: 2})
    assert f == 2

    f = e.subs({x1: 100})
    v = f.n(53, real=True)
    assert abs(float(v) - 459.13420537) < 1e-7

    f = e.diff(x1)
    # Enable this once symengine conversions are in
    #  sage as sage.psi is converted to a sympy function in `e.diff(x1)`
    # assert f == 1 + wrap_sage_function(sage.psi(x))


# This string contains Sage doctests, that execute all the functions above.
# When you add a new function, add it here as well.
"""
TESTS::
    sage: test_sage_conversions()
"""
