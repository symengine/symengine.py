from symengine.utilities import raises

from sympy import Symbol, Integer, sin, cos, exp, sqrt, E
from symengine.lib.symengine_wrapper import series, have_piranha, have_flint, Symbol as SESymbol

def test_series_expansion():
    if have_piranha:
        x = SESymbol('x')
        ex = series(sin(1+x), x, n=10, method='symengine')
        assert ex.coeff(x,7) == -cos(1)/5040

    if not have_flint and not have_piranha:
        return

    x = Symbol('x')
    ex = series(1/(1-x), x, n=10, method='symengine')
    assert ex.coeff(x,9) == 1
    x = SESymbol('x')
    ex = series(1/(1-x), x, n=10, method='symengine')
    assert ex.coeff(x,9) == 1
    ex = series(sin(x)*cos(x), x, n=10, method='symengine')
    assert ex.coeff(x,8) == 0
    assert ex.coeff(x,9) == Integer(2)/Integer(2835)

    ex = series(E**x, x, n=10, method='symengine')
    assert ex.coeff(x,9) == Integer(1)/Integer(362880)
    ex1 = series(1/sqrt(4-x), x, n=50, method='symengine')
    ex2 = series((4-x)**(Integer(-1)/Integer(2)), x, n=50, method='symengine')
    assert ex1.coeff(x,49) == ex2.coeff(x,49)

