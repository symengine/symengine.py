from .lib import symengine_wrapper as symengine
from .utilities import var, symbols
from .compatibility import with_metaclass
from .lib.symengine_wrapper import (sympify, sympify as S,
        SympifyError, sqrt, I, E, pi, Matrix, Derivative, exp,
        factorial, gcd, lcm, factor, nextprime, mod_inverse,
        totient, primitive_root, Lambdify as lambdify, symarray, 
        diff, eye, diag, ones, zeros, expand, Subs, 
        FunctionSymbol as AppliedUndef)
from types import ModuleType
import sys


class BasicMeta(type):
    def __instancecheck__(self, instance):
        return isinstance(instance, self._classes)


class Basic(with_metaclass(BasicMeta, object)):
    _classes = (symengine.Basic,)
    pass


class Number(Basic):
    _classes = (symengine.Number,)
    pass


class Symbol(symengine.PySymbol, Basic):
    _classes = (symengine.Symbol,)
    pass


class Rational(Number):
    _classes = (symengine.Rational, symengine.Integer)

    def __new__(cls, num, den=1):
        return symengine.Integer(num) / den


class Integer(Rational):
    _classes = (symengine.Integer,)

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


functions = ModuleType(__name__ + ".functions")
sys.modules[functions.__name__] = functions

functions.sqrt = sqrt
functions.exp = exp


class _FunctionRegistrarMeta(BasicMeta):

    def __new__(mcls, name, bases, dict):
        cls = BasicMeta.__new__(mcls, name, bases, dict)
        if not name.startswith("_"):
            setattr(functions, name, cls)
        return cls


class _RegisteredFunction(with_metaclass(_FunctionRegistrarMeta, Function)):
    pass


class log(_RegisteredFunction):
    _classes = (symengine.Log,)

    def __new__(cls, a, b=E):
        return symengine.log(a, b)


class sin(_RegisteredFunction):
    _classes = (symengine.Sin,)

    def __new__(cls, a):
        return symengine.sin(a)


class cos(_RegisteredFunction):
    _classes = (symengine.Cos,)

    def __new__(cls, a):
        return symengine.cos(a)


class tan(_RegisteredFunction):
    _classes = (symengine.Tan,)

    def __new__(cls, a):
        return symengine.tan(a)


class gamma(_RegisteredFunction):
    _classes = (symengine.Gamma,)

    def __new__(cls, a):
        return symengine.gamma(a)


class cot(_RegisteredFunction):
    _classes = (symengine.Cot,)

    def __new__(cls, a):
        return symengine.cot(a)


class csc(_RegisteredFunction):
    _classes = (symengine.Csc,)

    def __new__(cls, a):
        return symengine.csc(a)


class sec(_RegisteredFunction):
    _classes = (symengine.Sec,)

    def __new__(cls, a):
        return symengine.sec(a)


class asin(_RegisteredFunction):
    _classes = (symengine.ASin,)

    def __new__(cls, a):
        return symengine.asin(a)


class acos(_RegisteredFunction):
    _classes = (symengine.ACos,)

    def __new__(cls, a):
        return symengine.acos(a)


class atan(_RegisteredFunction):
    _classes = (symengine.ATan,)

    def __new__(cls, a):
        return symengine.atan(a)


class acot(_RegisteredFunction):
    _classes = (symengine.ACot,)

    def __new__(cls, a):
        return symengine.acot(a)


class acsc(_RegisteredFunction):
    _classes = (symengine.ACsc,)

    def __new__(cls, a):
        return symengine.acsc(a)


class asec(_RegisteredFunction):
    _classes = (symengine.ASec,)

    def __new__(cls, a):
        return symengine.asec(a)


class sinh(_RegisteredFunction):
    _classes = (symengine.Sinh,)

    def __new__(cls, a):
        return symengine.sinh(a)


class cosh(_RegisteredFunction):
    _classes = (symengine.Cosh,)

    def __new__(cls, a):
        return symengine.cosh(a)


class tanh(_RegisteredFunction):
    _classes = (symengine.Tanh,)

    def __new__(cls, a):
        return symengine.tanh(a)


class coth(_RegisteredFunction):
    _classes = (symengine.Coth,)

    def __new__(cls, a):
        return symengine.coth(a)

class csch(_RegisteredFunction):
    _classes = (symengine.Csch,)

    def __new__(cls, a):
        return symengine.csch(a)


class sech(_RegisteredFunction):
    _classes = (symengine.Sech,)

    def __new__(cls, a):
        return symengine.sech(a)


class asinh(_RegisteredFunction):
    _classes = (symengine.ASinh,)

    def __new__(cls, a):
        return symengine.asinh(a)


class acosh(_RegisteredFunction):
    _classes = (symengine.ACosh,)

    def __new__(cls, a):
        return symengine.acosh(a)


class atanh(_RegisteredFunction):
    _classes = (symengine.ATanh,)

    def __new__(cls, a):
        return symengine.atanh(a)


class acoth(_RegisteredFunction):
    _classes = (symengine.ACoth,)

    def __new__(cls, a):
        return symengine.acoth(a)

class acsch(_RegisteredFunction):
    _classes = (symengine.ACsch,)

    def __new__(cls, a):
        return symengine.acsch(a)


class asech(_RegisteredFunction):
    _classes = (symengine.ASech,)

    def __new__(cls, a):
        return symengine.asech(a)

class atan2(_RegisteredFunction):
    _classes = (symengine.ATan2,)

    def __new__(cls, a, b):
        return symengine.atan2(a, b)


'''
for i in ("""Sin Cos Tan Gamma Cot Csc Sec ASin ACos ATan
          ACot ACsc ASec Sinh Cosh Tanh Coth Sech Csch ASinh ACosh ATanh
          ACoth ASech ACsch""").split():
    print("""
class %s(Function):
    _classes = (symengine.%s,)

    def __new__(cls, a):
        return symengine.%s(a)
""" % (i.lower(), i, i.lower())
)
'''
