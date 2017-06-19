from symengine.utilities import raises
from symengine.lib.symengine_wrapper import (true, false, Eq, Ne,
    Ge, Gt, Le, Lt, Symbol, I)

x = Symbol("x")
y = Symbol("y")

def test_relationals():
    assert Eq(0) == true
    assert Eq(1) == false
    assert Eq(x, x) == true
    assert Eq(0, 0) == true
    assert Eq(1, 0) == false
    assert Ne(0, 0) == false
    assert Ne(1, 0) == true
    assert Lt(0, 1) == true
    assert Lt(1, 0) == false
    assert Le(0, 1) == true
    assert Le(1, 0) == false
    assert Le(0, 0) == true
    assert Gt(1, 0) == true
    assert Gt(0, 1) == false
    assert Ge(1, 0) == true
    assert Ge(0, 1) == false
    assert Ge(1, 1) == true
    assert Eq(I, 2) == false
    assert Ne(I, 2) == true

def test_rich_cmp():
    assert (x < y) == Lt(x, y)
    assert (x <= y) == Le(x, y)
    assert (x > y) == Gt(x, y)
    assert (x >= y) == Ge(x, y)
