from symengine import symbols, sin, sinh, Lambdify, have_numpy
import pickle
import unittest

@unittest.skipUnless(have_numpy, "Numpy not installed")
def test_llvm_double():
    import numpy as np
    args = x, y, z = symbols('x y z')
    expr = sin(sinh(x+y) + z)
    l = Lambdify(args, expr, cse=True, backend='llvm')
    ss = pickle.dumps(l)
    ll = pickle.loads(ss)
    inp = [1, 2, 3]
    assert np.allclose(l(inp), ll(inp))

