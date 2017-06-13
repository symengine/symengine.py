from .lib.symengine_wrapper import (Symbol, sympify as S, sympify,
        SympifyError, Add, Mul, Pow, function_symbol, I, E, pi, oo,
        zoo, nan, have_mpfr, have_mpc, have_flint, have_piranha,
        have_llvm, Integer, Rational, Float, Number, RealNumber,
        RealDouble, ComplexDouble, Max, Min, DenseMatrix, Matrix,
        ImmutableMatrix, ImmutableDenseMatrix, MutableDenseMatrix,
        MatrixBase, Basic, Lambdify, LambdifyCSE, Lambdify as lambdify,
        DictBasic, symarray, series, diff, zeros, eye, diag, ones,
        Derivative, Subs, add, expand, has_symbol, UndefFunction,
        Function, FunctionSymbol as AppliedUndef)
from .utilities import var, symbols
from .functions import *

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
