from .lib.symengine_wrapper import (Symbol, S, sympify,
        SympifyError, Add, Mul, Pow, function_symbol, I, E, pi, oo,
        zoo, nan, have_mpfr, have_mpc, have_flint, have_piranha,
        have_llvm, Integer, Rational, Float, Number, RealNumber,
        RealDouble, ComplexDouble, Max, Min, DenseMatrix, Matrix,
        ImmutableMatrix, ImmutableDenseMatrix, MutableDenseMatrix,
        MatrixBase, Basic, DictBasic, symarray, series, diff, zeros,
        eye, diag, ones, Derivative, Subs, add, expand, has_symbol,
        UndefFunction, Function, FunctionSymbol as AppliedUndef,
        have_numpy, true, false, Equality, Unequality, GreaterThan,
        LessThan, StrictGreaterThan, StrictLessThan, Eq, Ne, Ge, Le,
        Gt, Lt, golden_ratio as GoldenRatio, catalan as Catalan,
        eulergamma as EulerGamma, Dummy, perfect_power, integer_nthroot,
        isprime, sqrt_mod, Expr, cse)
from .utilities import var, symbols
from .functions import *

if have_mpfr:
    from .lib.symengine_wrapper import RealMPFR

if have_mpc:
    from .lib.symengine_wrapper import ComplexMPC

if have_numpy:
    from .lib.symengine_wrapper import (Lambdify, LambdifyCSE)

    def lambdify(args, exprs, **kwargs):
        return Lambdify(args, *exprs, **kwargs)


__version__ = "0.3.1.dev1"


def test():
    import pytest
    import os
    return not pytest.cmdline.main(
        [os.path.dirname(os.path.abspath(__file__))])
