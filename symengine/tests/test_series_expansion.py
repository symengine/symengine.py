from symengine.utilities import raises

from symengine import Symbol, series, sin, cos, exp, sqrt
from symengine.lib.symengine_wrapper import Rational, E

def test_series_expansion():
    x = Symbol('x')
    ex = series(1/(1-x), x, n=10)
    assert ex._sympy_().coeff(x,9) == 1
    ex = series(sin(x)*cos(x), x, n=10)
    assert ex._sympy_().coeff(x,8) == 0
    assert ex._sympy_().coeff(x,9) == Rational(2,2835)

    x = x._sympy_()
    ex = series(E**x, x, n=10)
    assert ex._sympy_().coeff(x,9) == Rational(1,362880)
    ex1 = series(1/sqrt(4-x), x, n=50)
    ex2 = series((4-x)**Rational(-1,2), x, n=50)
    assert ex1._sympy_().coeff(x,49) == ex2._sympy_().coeff(x,49)

