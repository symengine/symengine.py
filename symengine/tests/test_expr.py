from symengine import Symbol, Integer, oo, sin
from symengine.test_utilities import raises


def test_as_coefficients_dict():
    x = Symbol('x')
    y = Symbol('y')
    check = [x, y, x*y, Integer(1)]
    assert [(3*x + 2*x + y + 3).as_coefficients_dict()[i] for i in check] == \
        [5, 1, 0, 3]
    assert [(3*x*y).as_coefficients_dict()[i] for i in check] == \
        [0, 0, 3, 0]
    assert (3.0*x*y).as_coefficients_dict()[3.0*x*y] == 0
    assert (3.0*x*y).as_coefficients_dict()[x*y] == 3.0


def test_as_powers_dict():
    x = Symbol('x')
    y = Symbol('y')

    assert (2*x**y).as_powers_dict() == {2: 1, x: y}
    assert (2*x**2*y**3).as_powers_dict() == {2: 1, x: 2, y: 3}
    assert (-oo).as_powers_dict() == {Integer(-1): 1, oo: 1}
    assert (x**y).as_powers_dict() == {x: y}
    assert ((1/Integer(2))**y).as_powers_dict() == {Integer(2): -y}
    assert (x*(1/Integer(2))**y).as_powers_dict() == {x: Integer(1), Integer(2): -y}
    assert (2**y).as_powers_dict() == {2: y}
    assert (2**-y).as_powers_dict() == {2: -y}


def test_Basic__has():
    x = Symbol('x')
    y = Symbol('y')
    xpowy = x**y
    e = sin(xpowy)
    assert e.has(x)
    assert e.has(y)
    assert e.has(xpowy)
    raises(Exception, lambda: e.has(x+1))  # subtree matching of associative operators not yet supported
    assert (x + oo).has(oo)
    assert (x - oo).has(-oo)
    assert not (x + oo).has(-oo)
    #assert not (x - oo).has(oo) <-- not sure we want to test explicitly for "x + NegativeInfinity"
