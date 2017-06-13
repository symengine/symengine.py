from symengine.utilities import raises

from symengine import Symbol, Integer, sympify, SympifyError
from symengine.lib.symengine_wrapper import _sympify


def test_sympify1():
    assert sympify(1) == Integer(1)
    assert sympify(2) != Integer(1)
    assert sympify(-5) == Integer(-5)
    assert sympify(Integer(3)) == Integer(3)
    assert sympify("3+5") == Integer(8)


def test_sympify_error1a():
    class Test(object):
        pass
    raises(SympifyError, lambda: sympify(Test()))


def test_sympify_error1b():
    assert not _sympify("1***2", raise_error=False)


def test_error1():
    # _sympify doesn't parse strings
    raises(SympifyError, lambda: _sympify("x"))
