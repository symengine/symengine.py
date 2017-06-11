from symengine import Symbol, sin, cos, sqrt, Add, Mul, function_symbol, Integer, log, E, symbols
from symengine.lib.symengine_wrapper import Subs, Derivative


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
    e = x + sympy.loggamma(x)
    assert str(e) == "x + loggamma(x)"
    assert isinstance(e, Add)
    assert e + sympy.loggamma(x) == x + 2*sympy.loggamma(x)

    f = e.subs({x : 10})
    assert f == 10 + log(362880)

    f = e.subs({x : 2})
    assert f == 2

    f = e.subs({x : 100});
    v = f.n(53, real=True);
    assert abs(float(v) - 459.13420537) < 1e-7

    f = e.diff(x)
    assert f == 1 + sympy.polygamma(0, x)


def test_log():
    x = Symbol("x")
    y = Symbol("y")
    assert log(E) == 1
    assert log(x, x) == 1
    assert log(x, y) == log(x) / log(y)
