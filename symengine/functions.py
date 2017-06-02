import symengine
from types import ModuleType
import sys

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

