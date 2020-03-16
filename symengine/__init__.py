from .lib.symengine_wrapper import (
    have_mpfr, have_mpc, have_flint, have_piranha, have_llvm, have_llvm_long_double,
    I, E, pi, oo, zoo, nan, Symbol, Dummy, S, sympify, SympifyError,
    Integer, Rational, Float, Number, RealNumber, RealDouble, ComplexDouble,
    add, Add, Mul, Pow, function_symbol,
    Max, Min, DenseMatrix, Matrix,
    ImmutableMatrix, ImmutableDenseMatrix, MutableDenseMatrix,
    MatrixBase, Basic, DictBasic, symarray, series, diff, zeros,
    eye, diag, ones, Derivative, Subs, expand, has_symbol,
    UndefFunction, Function, latex,
    have_numpy, true, false, Equality, Unequality, GreaterThan,
    LessThan, StrictGreaterThan, StrictLessThan, Eq, Ne, Ge, Le,
    Gt, Lt, And, Or, Not, Nand, Nor, Xor, Xnor, perfect_power, integer_nthroot,
    isprime, sqrt_mod, Expr, cse, count_ops, ccode, Piecewise, Contains, Interval, FiniteSet,
    EmptySet, linsolve,
    FunctionSymbol as AppliedUndef,
    golden_ratio as GoldenRatio,
    catalan as Catalan,
    eulergamma as EulerGamma
)
from .utilities import var, symbols
from .functions import *
from .printing import init_printing

if have_mpfr:
    from .lib.symengine_wrapper import RealMPFR

if have_mpc:
    from .lib.symengine_wrapper import ComplexMPC

if have_numpy:
    from .lib.symengine_wrapper import (Lambdify, LambdifyCSE)

    def lambdify(args, exprs, **kwargs):
        return Lambdify(args, *exprs, **kwargs)


__version__ = "0.6.1"


def test():
    import pytest
    import os
    return not pytest.cmdline.main(
        [os.path.dirname(os.path.abspath(__file__))])
