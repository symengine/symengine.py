from symengine import Symbol, sin, cos, sqrt, Add, Mul, function_symbol, Integer, log, E, symbols
from symengine.lib.symengine_wrapper import (Subs, Derivative, LambertW, zeta, dirichlet_eta,
                                            zoo, pi, KroneckerDelta, LeviCivita, erf, erfc,
                                            oo, lowergamma, uppergamma, exp, loggamma, beta,
                                            polygamma, digamma, trigamma, EulerGamma)


def test_sin():
    x = Symbol("x")
    e = sin(x)
    assert e == sin(x)
    assert e != cos(x)

    assert sin(x).diff(x) == cos(x)
    assert cos(x).diff(x) == -sin(x)

    e = sqrt(x).diff(x).diff(x)
    f = sin(e)
    g = f.diff(x).diff(x)
    assert isinstance(g, Add)


def test_f():
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    f = function_symbol("f", x)
    g = function_symbol("g", x)
    assert f != g

    f = function_symbol("f", x)
    g = function_symbol("f", x)
    assert f == g

    f = function_symbol("f", x, y)
    g = function_symbol("f", y, x)
    assert f != g

    f = function_symbol("f", x, y)
    g = function_symbol("f", x, y)
    assert f == g


def test_derivative():
    x = Symbol("x")
    y = Symbol("y")
    f = function_symbol("f", x)
    assert f.diff(x) == function_symbol("f", x).diff(x)
    assert f.diff(x).diff(x) == function_symbol("f", x).diff(x).diff(x)
    assert f.diff(y) == 0
    assert f.diff(x).args == (f, x)
    assert f.diff(x).diff(x).args == (f, x, x)

    g = function_symbol("f", y)
    assert g.diff(x) == 0
    assert g.diff(y) == function_symbol("f", y).diff(y)
    assert g.diff(y).diff(y) == function_symbol("f", y).diff(y).diff(y)

    assert f - function_symbol("f", x) == 0

    f = function_symbol("f", x, y)
    assert f.diff(x).diff(y) == function_symbol("f", x, y).diff(x).diff(y)
    assert f.diff(Symbol("z")) == 0

    s = Derivative(function_symbol("f", x), x)
    assert s.expr == function_symbol("f", x)
    assert s.variables == (x,)


def test_abs():
    x = Symbol("x")
    e = abs(x)
    assert e == abs(x)
    assert e != cos(x)

    assert abs(5) == 5
    assert abs(-5) == 5
    assert abs(Integer(5)/3) == Integer(5)/3
    assert abs(-Integer(5)/3) == Integer(5)/3
    assert abs(Integer(5)/3+x) != Integer(5)/3
    assert abs(Integer(5)/3+x) == abs(Integer(5)/3+x)


def test_abs_diff():
    x = Symbol("x")
    y = Symbol("y")
    e = abs(x)
    assert e.diff(x) != e
    assert e.diff(x) != 0
    assert e.diff(y) == 0


def test_Subs():
    x = Symbol("x")
    y = Symbol("y")
    _x = Symbol("_xi_1")
    f = function_symbol("f", 2*x)
    assert str(f.diff(x)) == "2*Subs(Derivative(f(_xi_1), _xi_1), (_xi_1), (2*x))"
    # TODO: fix me
    # assert f.diff(x) == 2 * Subs(Derivative(function_symbol("f", _x), _x), [_x], [2 * x])
    assert Subs(Derivative(function_symbol("f", x, y), x), [x, y], [_x, x]) \
                == Subs(Derivative(function_symbol("f", x, y), x), [y, x], [x, _x])

    s = f.diff(x)/2
    _xi_1 = Symbol("_xi_1")
    assert s.expr == Derivative(function_symbol("f", _xi_1), _xi_1)
    assert s.variables == (_xi_1,)
    assert s.point == (2*x,)


def test_FunctionWrapper():
    import sympy
    n, m, theta, phi = sympy.symbols("n, m, theta, phi")
    r = sympy.Ynm(n, m, theta, phi)
    s = Integer(2)*r
    assert isinstance(s, Mul)
    assert isinstance(s.args[1]._sympy_(), sympy.Ynm)

    x = symbols("x")
    e = x + sympy.Mod(x, 2)
    assert str(e) == "x + Mod(x, 2)"
    assert isinstance(e, Add)
    assert e + sympy.Mod(x, 2) == x + 2*sympy.Mod(x, 2)

    f = e.subs({x : 10})
    assert f == 10

    f = e.subs({x : 2})
    assert f == 2

    f = e.subs({x : 100});
    v = f.n(53, real=True);
    assert abs(float(v) - 100.00000000) < 1e-7


def test_log():
    x = Symbol("x")
    y = Symbol("y")
    assert log(E) == 1
    assert log(x, x) == 1
    assert log(x, y) == log(x) / log(y)

def test_lambertw():
    assert LambertW(0) == 0
    assert LambertW(E) == 1

def test_zeta():
    x = Symbol("x")
    assert zeta(1) == zoo
    assert zeta(1, x) == zoo

def test_dirichlet_eta():
    assert dirichlet_eta(1) == log(2)
    assert dirichlet_eta(2) == pi**2/12

def test_kronecker_delta():
    x = Symbol("x")
    assert KroneckerDelta(1, 1) == 1
    assert KroneckerDelta(1, 2) == 0
    assert KroneckerDelta(x, x) == 1

def test_levi_civita():
    assert LeviCivita(1, 2, 3) == 1
    assert LeviCivita(1, 3, 2) == -1
    assert LeviCivita(1, 2, 2) == 0

def test_erf():
    assert erf(0) == 0
    assert erf(oo) == 1

def test_erfc():
    assert erfc(0) == 1
    assert erfc(oo) == 0

def test_lowergamma():
    assert lowergamma(1, 2) == 1 - exp(-2)

def test_uppergamma():
    assert uppergamma(1, 2) == exp(-2)
    assert uppergamma(4, 0) == 6

def test_loggamma():
    assert loggamma(-1) == oo
    assert loggamma(0) == oo
    assert loggamma(1) == 0
    assert loggamma(3) == log(2)

def test_beta():
    assert beta(3, 2) == beta(2, 3)

def test_polygamma():
    assert polygamma(0, 0) == zoo

def test_digamma():
    x = Symbol("x")
    assert digamma(x) == polygamma(0, x)
    assert digamma(0) == zoo
    assert digamma(1) == -EulerGamma

def test_trigamma():
    x = Symbol("x")
    assert trigamma(-2) == zoo
    assert trigamma(x) == polygamma(1, x)
