from symengine.sympy_compat import (Integer, Rational, S, Basic, Add, Mul,
    Pow, symbols, Symbol, log, sin, sech, csch, zeros, atan2, Number, Float,
    symengine)
from symengine.utilities import raises


def test_Integer():
    i = Integer(5)
    assert isinstance(i, Integer)
    assert isinstance(i, Rational)
    assert isinstance(i, Number)
    assert isinstance(i, Basic)
    assert i.p == 5
    assert i.q == 1


def test_Rational():
    i = S(1)/2
    assert isinstance(i, Rational)
    assert isinstance(i, Number)
    assert isinstance(i, Basic)
    assert i.p == 1
    assert i.q == 2
    x = symbols("x")
    assert not isinstance(x, Rational)
    assert not isinstance(x, Number)


def test_Float():
    A = Float("1.23", precision = 53)
    B = Float("1.23")
    C = Float(A)
    assert A == B == C
    assert isinstance(A, Float)
    assert isinstance(B, Float)
    assert isinstance(C, Float)
    assert isinstance(A, symengine.RealDouble)
    assert isinstance(B, symengine.RealDouble)
    assert isinstance(C, symengine.RealDouble)
    raises(ValueError, lambda: Float("1.23", dps = 3, precision = 10))
    raises(ValueError, lambda: Float(A, dps = 3, precision = 16))
    if symengine.have_mpfr:
        A = Float("1.23", dps = 16)
        B = Float("1.23", precision = 56)
        assert A == B
        assert isinstance(A, Float)
        assert isinstance(B, Float)
        assert isinstance(A, symengine.RealMPFR)
        assert isinstance(B, symengine.RealMPFR)
        A = Float(C, dps = 16)
        assert A == B
        assert isinstance(A, Float)
        assert isinstance(A, symengine.RealMPFR)
        A = Float(A, precision = 53)
        assert A == C
        assert isinstance(A, Float)
        assert isinstance(A, symengine.RealDouble)
    if not symengine.have_mpfr:
        raises(ValueError, lambda: Float("1.23", precision = 58))        


def test_Add():
    x, y = symbols("x y")
    i = Add(x, x)
    assert isinstance(i, Mul)
    i = Add(x, y)
    assert isinstance(i, Add)
    assert isinstance(i, Basic)


def test_Mul():
    x, y = symbols("x y")
    i = Mul(x, x)
    assert isinstance(i, Pow)
    i = Mul(x, y)
    assert isinstance(i, Mul)
    assert isinstance(i, Basic)


def test_Pow():
    x = symbols("x")
    i = Pow(x, 1)
    assert isinstance(i, Symbol)
    i = Pow(x, 2)
    assert isinstance(i, Pow)
    assert isinstance(i, Basic)


def test_sin():
    x = symbols("x")
    i = sin(0)
    assert isinstance(i, Integer)
    i = sin(x)
    assert isinstance(i, sin)


def test_sech():
    x = symbols("x")
    i = sech(0)
    assert isinstance(i, Integer)
    i = sech(x)
    assert isinstance(i, sech)


def test_csch():
    x = symbols("x")
    i = csch(x)
    assert isinstance(i, csch)
    i = csch(-1)
    j = csch(1)
    assert i == -j


def test_log():
    x, y = symbols("x y")
    i = log(x, y)
    assert isinstance(i, Mul)
    i = log(x)
    assert isinstance(i, log)


def test_ATan2():
    x, y = symbols("x y")
    i = atan2(x, y)
    assert isinstance(i, atan2)
    i = atan2(0, 1)
    assert i == 0


def test_zeros():
    assert zeros(3, c=2).shape == (3, 2)


def test_has_functions_module():
    import symengine.sympy_compat as sp
    assert sp.functions.sin(0) == 0


def test_subclass_symbol():
    # Subclass of Symbol with an extra attribute
    class Wrapper(Symbol):
        def __new__(cls, name, extra_attribute):
            return Symbol.__new__(cls, name)

        def __init__(self, name, extra_attribute):
            super(Wrapper, self).__init__(name)
            self.extra_attribute = extra_attribute

    # Instantiate the subclass
    x = Wrapper("x", extra_attribute=3)
    assert x.extra_attribute == 3
    two_x = 2 * x
    # Check that after arithmetic, same subclass is returned
    assert two_x.args[1] is x
