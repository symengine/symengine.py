from .lib.symengine_wrapper import (Symbol, sympify as S, sympify,
        SympifyError, Add, Mul, Pow, function_symbol, I, E, pi, oo,
        zoo, nan, have_mpfr, have_mpc, have_flint, have_piranha,
        have_llvm, Integer, Rational, Float, Number, RealNumber,
        RealDouble, ComplexDouble, Max, Min, DenseMatrix, Matrix,
        ImmutableMatrix, ImmutableDenseMatrix, MutableDenseMatrix,
        MatrixBase, Basic, DictBasic, symarray, series, diff, zeros,
        eye, diag, ones, Derivative, Subs, add, expand, has_symbol,
        UndefFunction, Function, FunctionSymbol as AppliedUndef,
        have_numpy)
from .utilities import var, symbols
from .functions import *

if have_mpfr:
    from .lib.symengine_wrapper import RealMPFR

if have_mpc:
    from .lib.symengine_wrapper import ComplexMPC

if have_numpy:
    from .lib.symengine_wrapper import Lambdify, LambdifyCSE

    def lambdify(args, exprs, real=True, backend=None):
        try:
            len(args)
        except TypeError:
            args = [args]
        lmb = Lambdify(args, *exprs, real=real, backend=backend)
        def f(*inner_args):
            if len(inner_args) != len(args):
                raise TypeError("Incorrect number of arguments")
            return lmb(inner_args)
        return f


__version__ = "0.3.0.rc0"


def test():
    import pytest
    import os
    return not pytest.cmdline.main(
        [os.path.dirname(os.path.abspath(__file__))])
