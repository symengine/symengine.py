from symengine.utilities import raises
from symengine.lib.symengine_wrapper import (Interval, EmptySet, FiniteSet,
	I, oo, solve, Eq, Symbol)

def test_solve():
	x = Symbol("x")
	reals = Interval(-oo, oo)

	assert solve(1, x, reals) == EmptySet()
	assert solve(0, x, reals) == reals
	assert solve(x + 3, x, reals) == FiniteSet(-3)
	assert solve(x + 3, x, Interval(0, oo)) == EmptySet()
	assert solve(x, x, reals) == FiniteSet(0)
	assert solve(x**2 + 1, x) == FiniteSet(-I, I)
	assert solve(x**2 - 2*x + 1, x) == FiniteSet(1)
	assert solve(Eq(x**3 + 3*x**2 + 3*x, -1), x, reals) == FiniteSet(-1)
	assert solve(x**3 - x, x) == FiniteSet(0, 1, -1)
