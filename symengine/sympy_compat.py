from .lib import symengine_wrapper as symengine
from .utilities import var, symbols
from .compatibility import with_metaclass
from .lib.symengine_wrapper import (Symbol, sympify, sympify as S,
        SympifyError, sqrt, I, E, pi, Matrix, Derivative, exp,
        Lambdify as lambdify, symarray, diff, zeros, eye, diag, ones, zeros,
        expand, FunctionSymbol as AppliedUndef)


class BasicMeta(type):
    def __instancecheck__(self, instance):
        return isinstance(instance, self._classes)


class Basic(with_metaclass(BasicMeta, object)):
    _classes = (symengine.Basic,)
    pass


class Number(Basic):
    _classes = (symengine.Number,) + Basic._classes
    pass


class Rational(Number):
    _classes = (symengine.Rational,) + Number._classes

    def __new__(cls, num, den = 1):
        return symengine.Integer(num) / den


class Integer(Rational):
    _classes = (symengine.Integer,) + Rational._classes

    def __new__(cls, i):
        return symengine.Integer(i)


class Add(Basic):
    _classes = (symengine.Add,)

    def __new__(cls, *args):
        return symengine.add(*args)


class Mul(Basic):
    _classes = (symengine.Mul,)

    def __new__(cls, *args):
        return symengine.mul(*args)


class Pow(Basic):
    _classes = (symengine.Pow,)

    def __new__(cls, a, b):
        return symengine.sympify(a) ** b


class Function(Basic):
    _classes = (symengine.Function,)

    def __new__(cls, name):
        return symengine.UndefFunction(name)


class log(Function):
    _classes = (symengine.Log,)

    def __new__(cls, a, b = E):
        return symengine.log(a, b)


class sin(Function):
    _classes = (symengine.Sin,)

    def __new__(cls, a):
        return symengine.sin(a)


class cos(Function):
    _classes = (symengine.Cos,)

    def __new__(cls, a):
        return symengine.cos(a)


class tan(Function):
    _classes = (symengine.Tan,)

    def __new__(cls, a):
        return symengine.tan(a)

class gamma(Function):
    _classes = (symengine.Gamma,)

    def __new__(cls, a):
        return symengine.gamma(a)


class cot(Function):
    _classes = (symengine.Cot,)

    def __new__(cls, a):
        return symengine.cot(a)


class csc(Function):
    _classes = (symengine.Csc,)

    def __new__(cls, a):
        return symengine.csc(a)


class sec(Function):
    _classes = (symengine.Sec,)

    def __new__(cls, a):
        return symengine.sec(a)


class asin(Function):
    _classes = (symengine.ASin,)

    def __new__(cls, a):
        return symengine.asin(a)


class acos(Function):
    _classes = (symengine.ACos,)

    def __new__(cls, a):
        return symengine.acos(a)


class atan(Function):
    _classes = (symengine.ATan,)

    def __new__(cls, a):
        return symengine.atan(a)


class acot(Function):
    _classes = (symengine.ACot,)

    def __new__(cls, a):
        return symengine.acot(a)


class acsc(Function):
    _classes = (symengine.ACsc,)

    def __new__(cls, a):
        return symengine.acsc(a)


class asec(Function):
    _classes = (symengine.ASec,)

    def __new__(cls, a):
        return symengine.asec(a)


class sinh(Function):
    _classes = (symengine.Sinh,)

    def __new__(cls, a):
        return symengine.sinh(a)


class cosh(Function):
    _classes = (symengine.Cosh,)

    def __new__(cls, a):
        return symengine.cosh(a)


class tanh(Function):
    _classes = (symengine.Tanh,)

    def __new__(cls, a):
        return symengine.tanh(a)


class coth(Function):
    _classes = (symengine.Coth,)

    def __new__(cls, a):
        return symengine.coth(a)


class asinh(Function):
    _classes = (symengine.ASinh,)

    def __new__(cls, a):
        return symengine.asinh(a)


class acosh(Function):
    _classes = (symengine.ACosh,)

    def __new__(cls, a):
        return symengine.acosh(a)


class atanh(Function):
    _classes = (symengine.ATanh,)

    def __new__(cls, a):
        return symengine.atanh(a)


class acoth(Function):
    _classes = (symengine.ACoth,)

    def __new__(cls, a):
        return symengine.acoth(a)

'''
for i in ("""Sin Cos Tan Gamma Cot Csc Sec ASin ACos ATan
          ACot ACsc ASec Sinh Cosh Tanh Coth ASinh ACosh ATanh
          ACoth""").split():
    print("""
class %s(Function):
    _classes = (symengine.%s,)

    def __new__(cls, a):
        return symengine.%s(a)
""" % (i.lower(), i, i.lower())
)
'''
