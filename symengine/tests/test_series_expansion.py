from symengine.utilities import raises

from symengine.lib.symengine_wrapper import Symbol, Integer, series, sin, cos, exp, sqrt, E

def test_series_expansion():
    x = Symbol('x')
    ex = series(1/(1-x), x, n=10, method='symengine')
    assert ex._sympy_().coeff(x,9) == 1
    ex = series(sin(x)*cos(x), x, n=10, method='symengine')
    assert ex._sympy_().coeff(x,8) == 0
    assert ex._sympy_().coeff(x,9) == Integer(2)/Integer(2835)

    ex = series(E**x, x, n=10, method='symengine')
    assert ex._sympy_().coeff(x,9) == Integer(1)/Integer(362880)
    ex1 = series(1/sqrt(4-x), x, n=50, method='symengine')
    ex2 = series((4-x)**(Integer(-1)/Integer(2)), x, n=50, method='symengine')
    assert ex1._sympy_().coeff(x,49) == ex2._sympy_().coeff(x,49)

