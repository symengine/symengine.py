from .lib import symengine_wrapper as symengine
from .utilities import var, symbols
from .compatibility import with_metaclass
from .lib.symengine_wrapper import (sympify, sympify as S,
        SympifyError, sqrt, I, E, pi, MutableDenseMatrix, 
        ImmutableDenseMatrix, DenseMatrix, Matrix, Derivative, exp,
        nextprime, mod_inverse, primitive_root, Lambdify as lambdify, 
        symarray, diff, eye, diag, ones, zeros, expand, Subs, 
        FunctionSymbol as AppliedUndef, Max, Min, Integer, Rational,
        Float, Number, Add, Mul, Pow, sin, cos, tan, cot, csc, sec,
        asin, acos, atan, acot, acsc, asec, sinh, cosh, tanh, coth, sech, csch,
        asinh, acosh, atanh, acoth, asech, acsch, gamma, log, atan2)
from types import ModuleType
import sys


class BasicMeta(type):
    def __instancecheck__(self, instance):
        return isinstance(instance, self._classes)


class Basic(with_metaclass(BasicMeta, object)):
    _classes = (symengine.Basic,)
    pass


class Symbol(symengine.PySymbol, Basic):
    _classes = (symengine.Symbol,)
    pass


functions = ModuleType(__name__ + ".functions")
sys.modules[functions.__name__] = functions

functions.sqrt = sqrt
functions.exp = exp

for name in ("""sin cos tan cot csc sec
                asin acos atan acot acsc asec
                sinh cosh tanh coth sech csch
                asinh acosh atanh acoth asech acsch
                gamma log atan2""").split():
    setattr(functions, name, getattr(symengine, name))

