from .lib.symengine_wrapper import (Symbol, Integer, sympify, SympifyError,
        Add, Mul, Pow, exp, log, sqrt, function_symbol, I, E, pi,
        have_mpfr, have_mpc, RealDouble, ComplexDouble, DenseMatrix,
        sin, cos, tan, cot, csc, sec, asin, acos, atan, acot, acsc, asec,
        sinh, cosh, tanh, coth, asinh, acosh, atanh, acoth)
from .utilities import var, symbols

if have_mpfr:
    from .lib.symengine_wrapper import RealMPFR

if have_mpc:
    from .lib.symengine_wrapper import ComplexMPC

__version__ = "0.1.0.dev"

def ascii_art():
	print("""\
	 _____           _____         _
	|   __|_ _ _____|   __|___ ___|_|___ ___
	|__   | | |     |   __|   | . | |   | -_|
	|_____|_  |_|_|_|_____|_|_|_  |_|_|_|___|
	      |___|               |___|
	""")

def test():
    import pytest, os
    return not pytest.cmdline.main(
        [os.path.dirname(os.path.abspath(__file__))])
