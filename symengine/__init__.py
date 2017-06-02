from .lib.symengine_wrapper import (Symbol, sympify as S, sympify,
        SympifyError, Add, Mul, Pow, function_symbol, I, E, pi,
        have_mpfr, have_mpc, have_flint, have_piranha, have_llvm, 
        Integer, Rational, Float, Number, RealNumber, RealDouble,
        ComplexDouble, Max, Min, DenseMatrix, Matrix, ImmutableMatrix,
        ImmutableDenseMatrix, MutableDenseMatrix, MutableMatrix,
        sin, cos, tan, cot, csc, sec, asin, acos, atan, acot, acsc, asec,
        sinh, cosh, tanh, coth, sech, csch, asinh, acosh, atanh, acoth,
        asech, acsch, atan2, exp, log, gamma, sqrt,
        Lambdify, LambdifyCSE, Lambdify as lambdify, DictBasic, symarray,
        series, diff, zeros, eye, diag,ones, zeros,
        add, expand, has_symbol, UndefFunction)
from .utilities import var, symbols

if have_mpfr:
    from .lib.symengine_wrapper import RealMPFR

if have_mpc:
    from .lib.symengine_wrapper import ComplexMPC

__version__ = "0.2.1.dev"


def test():
    import pytest
    import os
    return not pytest.cmdline.main(
        [os.path.dirname(os.path.abspath(__file__))])
