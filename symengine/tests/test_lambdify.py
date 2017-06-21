# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)


import array
import cmath
import itertools
import math
import sys

import symengine as se
from symengine.utilities import raises
from symengine import have_numpy
if have_numpy:
    import numpy as np

def _size(arr):
    try:
        return arr.memview.size
    except AttributeError:
        return len(arr)


def isclose(a, b, rtol=1e-13, atol=1e-13):
    discr = a - b
    toler = (atol + rtol*abs(a))
    return abs(discr) < toler


def allclose(vec1, vec2, rtol=1e-13, atol=1e-13):
    n1, n2 = _size(vec1), _size(vec2)
    if n1 != n2:
        return False

    for idx in range(n1):
        if not isclose(vec1[idx], vec2[idx], rtol, atol):
            return False
    return True


def test_get_shape():
    if not have_numpy:
        return
    get_shape = se.lib.symengine_wrapper.get_shape
    assert get_shape([1]) == (1,)
    assert get_shape([1, 1, 1]) == (3,)
    assert get_shape([[1], [1], [1]]) == (3, 1)
    assert get_shape([[1, 1, 1]]) == (1, 3)

    x = se.symbols('x')
    exprs = [x+1, x+2, x+3, 1/x, 1/(x*x), 1/(x**3.0)]
    A = se.DenseMatrix(2, 3, exprs)
    assert get_shape(A) == (2, 3)


def test_ravel():
    if not have_numpy:
        return
    x = se.symbols('x')
    ravel = se.lib.symengine_wrapper.ravel
    exprs = [x+1, x+2, x+3, 1/x, 1/(x*x), 1/(x**3.0)]
    A = se.DenseMatrix(2, 3, exprs)
    assert ravel(A) == exprs


def test_Lambdify():
    if not have_numpy:
        return
    n = 7
    args = x, y, z = se.symbols('x y z')
    L = se.Lambdify(args, [x+y+z, x**2, (x-y)/z, x*y*z], backend='lambda')
    assert allclose(L(range(n, n+len(args))),
                    [3*n+3, n**2, -1/(n+2), n*(n+1)*(n+2)])


def test_Lambdify_LLVM():
    if not have_numpy:
        return
    n = 7
    args = x, y, z = se.symbols('x y z')
    if not se.have_llvm:
        raises(ValueError, lambda: se.Lambdify(args, [x+y+z, x**2,
                                                      (x-y)/z, x*y*z],
                                               backend='llvm'))
        return
    L = se.Lambdify(args, [x+y+z, x**2, (x-y)/z, x*y*z], backend='llvm')
    assert allclose(L(range(n, n+len(args))),
                    [3*n+3, n**2, -1/(n+2), n*(n+1)*(n+2)])


def _get_2_to_2by2():
    args = x, y = se.symbols('x y')
    exprs = np.array([[x+y+1.0, x*y],
                      [x/y, x**y]])
    L = se.Lambdify(args, exprs)

    def check(A, inp):
        X, Y = inp
        assert abs(A[0, 0] - (X+Y+1.0)) < 1e-15
        assert abs(A[0, 1] - (X*Y)) < 1e-15
        assert abs(A[1, 0] - (X/Y)) < 1e-15
        assert abs(A[1, 1] - (X**Y)) < 1e-13
    return L, check


def test_Lambdify_2dim():
    if not have_numpy:
        return
    lmb, check = _get_2_to_2by2()
    for inp in [(5, 7), np.array([5, 7]), [5.0, 7.0]]:
        A = lmb(inp)
        assert A.shape == (2, 2)
        check(A, inp)


def _get_array():
    X, Y, Z = inp = array.array('d', [1, 2, 3])
    args = x, y, z = se.symbols('x y z')
    exprs = [x+y+z, se.sin(x)*se.log(y)*se.exp(z)]
    ref = [X+Y+Z, math.sin(X)*math.log(Y)*math.exp(Z)]

    def check(arr):
        assert all([abs(x1-x2) < 1e-13 for x1, x2 in zip(ref, arr)])
    return args, exprs, inp, check


def test_array():
    if not have_numpy:
        return
    args, exprs, inp, check = _get_array()
    lmb = se.Lambdify(args, exprs)
    out = lmb(inp)
    check(out)


def test_numpy_array_out_exceptions():
    if not have_numpy:
        return
    args, exprs, inp, check = _get_array()
    lmb = se.Lambdify(args, exprs)

    all_right = np.empty(len(exprs))
    lmb(inp, all_right)

    too_short = np.empty(len(exprs) - 1)
    raises(ValueError, lambda: (lmb(inp, too_short)))

    wrong_dtype = np.empty(len(exprs), dtype=int)
    raises(ValueError, lambda: (lmb(inp, wrong_dtype)))

    read_only = np.empty(len(exprs))
    read_only.flags['WRITEABLE'] = False
    raises(ValueError, lambda: (lmb(inp, read_only)))

    all_right_broadcast = np.empty((4, len(exprs)))
    inp_bcast = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    lmb(np.array(inp_bcast), all_right_broadcast)

    noncontig_broadcast = np.empty((4, len(exprs), 3)).transpose((1, 2, 0))
    raises(ValueError, lambda: (lmb(inp_bcast, noncontig_broadcast)))


def test_broadcast():
    if not have_numpy:
        return
    a = np.linspace(-np.pi, np.pi)
    inp = np.vstack((np.cos(a), np.sin(a))).T  # 50 rows 2 cols
    x, y = se.symbols('x y')
    distance = se.Lambdify([x, y], [se.sqrt(x**2 + y**2)])
    assert np.allclose(distance([inp[0, 0], inp[0, 1]]), [1])
    dists = distance(inp)
    assert dists.shape == (50, 1)
    assert np.allclose(dists, 1)


def test_broadcast_multiple_extra_dimensions():
    if not have_numpy:
        return
    inp = np.arange(12.).reshape((4, 3, 1))
    x = se.symbols('x')
    cb = se.Lambdify([x], [x**2, x**3])
    assert np.allclose(cb([inp[0, 2]]), [4, 8])
    out = cb(inp)
    assert out.shape == (4, 3, 2)
    assert abs(out[2, 1, 0] - 7**2) < 1e-14
    assert abs(out[2, 1, 1] - 7**3) < 1e-14
    assert abs(out[-1, -1, 0] - 11**2) < 1e-14
    assert abs(out[-1, -1, 1] - 11**3) < 1e-14


def _get_cse_exprs():
    import sympy as sp
    args = x, y = sp.symbols('x y')
    exprs = [x*x + y, y/(x*x), y*x*x+x]
    inp = [11, 13]
    ref = [121+13, 13/121, 13*121 + 11]
    return args, exprs, inp, ref


def test_cse():
    if not have_numpy:
        return
    args, exprs, inp, ref = _get_cse_exprs()
    lmb = se.LambdifyCSE(args, exprs)
    out = lmb(inp)
    assert allclose(out, ref)

def _get_cse_exprs_big():
    # this is essentially a performance test (can be replaced by a benchmark)
    x, p = se.symarray('x', 14), se.symarray('p', 14)
    exp = se.exp
    exprs = [
        x[0] + x[1] - x[4] + 36.252574322669, x[0] - x[2] + x[3] + 21.3219379611249,
        x[3] + x[5] - x[6] + 9.9011158998744, 2*x[3] + x[5] - x[7] + 18.190422234653,
        3*x[3] + x[5] - x[8] + 24.8679190043357, 4*x[3] + x[5] - x[9] + 29.9336062089226,
        -x[10] + 5*x[3] + x[5] + 28.5520551531262, 2*x[0] + x[11] - 2*x[4] - 2*x[5] + 32.4401680272417,
        3*x[1] - x[12] + x[5] + 34.9992934135095, 4*x[1] - x[13] + x[5] + 37.0716199972041,
        (p[0] - p[1] + 2*p[10] + 2*p[11] - p[12] - 2*p[13] + p[2] + 2*p[5] + 2*p[6] + 2*p[7] +
         2*p[8] + 2*p[9] - exp(x[0]) + exp(x[1]) - 2*exp(x[10]) - 2*exp(x[11]) + exp(x[12]) +
         2*exp(x[13]) - exp(x[2]) - 2*exp(x[5]) - 2*exp(x[6]) - 2*exp(x[7]) - 2*exp(x[8]) - 2*exp(x[9])),
        (-p[0] - p[1] - 15*p[10] - 2*p[11] - 3*p[12] - 4*p[13] - 4*p[2] - 3*p[3] - 2*p[4] - 3*p[6] -
         6*p[7] - 9*p[8] - 12*p[9] + exp(x[0]) + exp(x[1]) + 15*exp(x[10]) + 2*exp(x[11]) +
         3*exp(x[12]) + 4*exp(x[13]) + 4*exp(x[2]) + 3*exp(x[3]) + 2*exp(x[4]) + 3*exp(x[6]) +
         6*exp(x[7]) + 9*exp(x[8]) + 12*exp(x[9])),
        (-5*p[10] - p[2] - p[3] - p[6] - 2*p[7] - 3*p[8] - 4*p[9] + 5*exp(x[10]) + exp(x[2]) + exp(x[3]) +
         exp(x[6]) + 2*exp(x[7]) + 3*exp(x[8]) + 4*exp(x[9])),
        -p[1] - 2*p[11] - 3*p[12] - 4*p[13] - p[4] + exp(x[1]) + 2*exp(x[11]) + 3*exp(x[12]) + 4*exp(x[13]) + exp(x[4]),
        (-p[10] - 2*p[11] - p[12] - p[13] - p[5] - p[6] - p[7] - p[8] - p[9] + exp(x[10]) +
         2*exp(x[11]) + exp(x[12]) + exp(x[13]) + exp(x[5]) + exp(x[6]) + exp(x[7]) + exp(x[8]) + exp(x[9]))
    ]
    return tuple(x) + tuple(p), exprs, np.ones(len(x) + len(p))


def test_cse_big():
    if not have_numpy:
        return
    args, exprs, inp = _get_cse_exprs_big()
    lmb = se.LambdifyCSE(args, exprs)
    out = lmb(inp)
    ref = [expr.xreplace(dict(zip(args, inp))) for expr in exprs]
    assert allclose(out, ref)


def test_broadcast_c():
    if not have_numpy:
        return
    n = 3
    inp = np.arange(2*n).reshape((n, 2))
    lmb, check = _get_2_to_2by2()
    A = lmb(inp)
    assert A.shape == (3, 2, 2)
    for i in range(n):
        check(A[i, ...], inp[i, :])


def test_broadcast_fortran():
    if not have_numpy:
        return
    n = 3
    inp = np.arange(2*n).reshape((n, 2), order='F')
    lmb, check = _get_2_to_2by2()
    A = lmb(inp)
    assert A.shape == (3, 2, 2)
    for i in range(n):
        check(A[i, ...], inp[i, :])


def _get_1_to_2by3_matrix(Mtx=se.DenseMatrix):
    x = se.symbols('x')
    args = x,
    exprs = Mtx(2, 3, [x+1, x+2, x+3,
                       1/x, 1/(x*x), 1/(x**3.0)])
    L = se.Lambdify(args, exprs)

    def check(A, inp):
        X, = inp
        assert abs(A[0, 0] - (X+1)) < 1e-15
        assert abs(A[0, 1] - (X+2)) < 1e-15
        assert abs(A[0, 2] - (X+3)) < 1e-15
        assert abs(A[1, 0] - (1/X)) < 1e-15
        assert abs(A[1, 1] - (1/(X*X))) < 1e-15
        assert abs(A[1, 2] - (1/(X**3.0))) < 1e-15
    return L, check


def test_2dim_Matrix():
    if not have_numpy:
        return
    L, check = _get_1_to_2by3_matrix()
    inp = [7]
    check(L(inp), inp)


def test_2dim_Matrix__sympy():
    if not have_numpy:
        return
    import sympy as sp
    L, check = _get_1_to_2by3_matrix(sp.Matrix)
    inp = [7]
    check(L(inp), inp)



def _test_2dim_Matrix_broadcast():
    L, check = _get_1_to_2by3_matrix()
    inp = range(1, 5)
    out = L(inp)
    for i in range(len(inp)):
        check(out[i, ...], (inp[i],))



def test_2dim_Matrix_broadcast():
    if not have_numpy:
        return
    _test_2dim_Matrix_broadcast()


def test_2dim_Matrix_broadcast_multiple_extra_dim():
    if not have_numpy:
        return
    L, check = _get_1_to_2by3_matrix()
    inp = np.arange(1, 4*5*6+1).reshape((4, 5, 6))
    out = L(inp)
    assert out.shape == (4, 5, 6, 2, 3)
    for i, j, k in itertools.product(range(4), range(5), range(6)):
        check(out[i, j, k, ...], (inp[i, j, k],))


def test_jacobian():
    if not have_numpy:
        return
    x, y = se.symbols('x, y')
    args = se.DenseMatrix(2, 1, [x, y])
    v = se.DenseMatrix(2, 1, [x**3 * y, (x+1)*(y+1)])
    jac = v.jacobian(args)
    lmb = se.Lambdify(args, jac)
    out = np.empty((2, 2))
    inp = X, Y = 7, 11
    lmb(inp, out)
    assert np.allclose(out, [[3 * X**2 * Y, X**3],
                             [Y + 1, X + 1]])


def test_jacobian__broadcast():
    if not have_numpy:
        return
    x, y = se.symbols('x, y')
    args = se.DenseMatrix(2, 1, [x, y])
    v = se.DenseMatrix(2, 1, [x**3 * y, (x+1)*(y+1)])
    jac = v.jacobian(args)
    lmb = se.Lambdify(args, jac)
    out = np.empty((3, 2, 2))
    inp0 = 7, 11
    inp1 = 8, 13
    inp2 = 5, 9
    inp = np.array([inp0, inp1, inp2])
    lmb(inp, out)
    for idx, (X, Y) in enumerate([inp0, inp1, inp2]):
        assert np.allclose(out[idx, ...], [[3 * X**2 * Y, X**3],
                                           [Y + 1, X + 1]])


def test_excessive_args():
    if not have_numpy:
        return
    x = se.symbols('x')
    lmb = se.Lambdify([x], [-x])
    inp = np.ones(2)
    out = lmb(inp)
    assert np.allclose(inp, [1, 1])
    assert len(out) == 2  # broad casting
    assert np.allclose(out, -1)


def test_excessive_out():
    if not have_numpy:
        return
    x = se.symbols('x')
    lmb = se.Lambdify([x], [-x])
    inp = np.ones(1)
    out = np.ones(2)
    out = lmb(inp, out)
    assert np.allclose(inp, [1, 1])
    assert out.shape == (2,)
    assert out[0] == -1
    assert out[1] == 1


def all_indices(shape):
    return itertools.product(*(range(dim) for dim in shape))


def ravelled(A):
    try:
        return A.ravel()
    except AttributeError:
        L = []
        for idx in all_indices(A.memview.shape):
            L.append(A[idx])
        return L


def _get_2_to_2by2_list(real=True):
    args = x, y = se.symbols('x y')
    exprs = [[x + y*y, y*y], [x*y*y, se.sqrt(x)+y*y]]
    L = se.Lambdify(args, exprs, real=real)

    def check(A, inp):
        X, Y = inp
        assert A.shape[-2:] == (2, 2)
        ref = [X + Y*Y, Y*Y, X*Y*Y, cmath.sqrt(X)+Y*Y]
        ravA = ravelled(A)
        size = _size(ravA)
        for i in range(size//4):
            for j in range(4):
                assert isclose(ravA[i*4 + j], ref[j])
    return L, check


def test_2_to_2by2():
    if not have_numpy:
        return
    L, check = _get_2_to_2by2_list()
    inp = [13, 17]
    A = L(inp)
    check(A, inp)


def test_unsafe_real():
    if not have_numpy:
        return
    L, check = _get_2_to_2by2_list()
    inp = np.array([13., 17.])
    out = np.empty(4)
    L.unsafe_real(inp, out)
    check(out.reshape((2, 2)), inp)


def test_unsafe_complex():
    if not have_numpy:
        return
    L, check = _get_2_to_2by2_list(real=False)
    assert not L.real
    inp = np.array([13+11j, 7+4j], dtype=np.complex128)
    out = np.empty(4, dtype=np.complex128)
    L.unsafe_complex(inp, out)
    check(out.reshape((2, 2)), inp)


def test_itertools_chain():
    if not have_numpy:
        return
    args, exprs, inp, check = _get_array()
    L = se.Lambdify(args, exprs)
    inp = itertools.chain([inp[0]], (inp[1],), [inp[2]])
    A = L(inp)
    check(A)


# @pytest.mark.xfail(not have_numpy, reason='array.array lacks "Zd"')
def test_complex_1():
    if not have_numpy:
        return
    x = se.Symbol('x')
    lmb = se.Lambdify([x], [1j + x], real=False)
    assert abs(lmb([11+13j])[0] -
               (11 + 14j)) < 1e-15


# @pytest.mark.xfail(not have_numpy, reason='array.array lacks "Zd"')
def test_complex_2():
    if not have_numpy:
        return
    x = se.Symbol('x')
    lmb = se.Lambdify([x], [3 + x - 1j], real=False)
    assert abs(lmb([11+13j])[0] -
               (14 + 12j)) < 1e-15


def test_more_than_255_args():
    # SymPy's lambdify can handle at most 255 arguments
    # this is a proof of concept that this limitation does
    # not affect SymEngine's Lambdify class
    if not have_numpy:
        return
    n = 257
    x = se.symarray('x', n)
    p, q, r = 17, 42, 13
    terms = [i*s for i, s in enumerate(x, p)]
    exprs = [se.add(*terms), r + x[0], -99]
    callback = se.Lambdify(x, exprs)
    input_arr = np.arange(q, q + n*n).reshape((n, n))
    out = callback(input_arr)
    ref = np.empty((n, 3))
    coeffs = np.arange(p, p + n, dtype=np.int64)
    for i in range(n):
        ref[i, 0] = coeffs.dot(np.arange(q + n*i, q + n*(i+1), dtype=np.int64))
        ref[i, 1] = q + n*i + r
    ref[:, 2] = -99
    assert np.allclose(out, ref)


def _Lambdify_heterogeneous_output(Lambdify):
    x, y = se.symbols('x, y')
    args = se.DenseMatrix(2, 1, [x, y])
    v = se.DenseMatrix(2, 1, [x**3 * y, (x+1)*(y+1)])
    jac = v.jacobian(args)
    exprs = [jac, x+y, v, (x+1)*(y+1)]
    lmb = Lambdify(args, *exprs)
    inp0 = 7, 11
    inp1 = 8, 13
    inp2 = 5, 9
    inp = np.array([inp0, inp1, inp2])
    o_j, o_xpy, o_v, o_xty = lmb(inp)
    for idx, (X, Y) in enumerate([inp0, inp1, inp2]):
        assert np.allclose(o_j[idx, ...], [[3 * X**2 * Y, X**3],
                                           [Y + 1, X + 1]])
        assert np.allclose(o_xpy[idx, ...], [X+Y])
        assert np.allclose(o_v[idx, ...], [[X**3 * Y], [(X+1)*(Y+1)]])
        assert np.allclose(o_xty[idx, ...], [(X+1)*(Y+1)])


def test_Lambdify_heterogeneous_output():
    if not have_numpy:
        return
    _Lambdify_heterogeneous_output(se.Lambdify)


def test_LambdifyCSE_heterogeneous_output():
    if not have_numpy:
        return
    _Lambdify_heterogeneous_output(se.LambdifyCSE)


def _sympy_lambdify_heterogeneous_output(cb, Mtx):
    x, y = se.symbols('x, y')
    args = Mtx(2, 1, [x, y])
    v = Mtx(2, 1, [x**3 * y, (x+1)*(y+1)])
    jac = v.jacobian(args)
    exprs = [jac, x+y, v, (x+1)*(y+1)]
    lmb = cb(args, exprs)
    inp0 = 7, 11
    inp1 = 8, 13
    inp2 = 5, 9
    for idx, (X, Y) in enumerate([inp0, inp1, inp2]):
        o_j, o_xpy, o_v, o_xty = lmb(X, Y)
        assert np.allclose(o_j, [[3 * X**2 * Y, X**3],
                                 [Y + 1, X + 1]])
        assert np.allclose(o_xpy, [X+Y])
        assert np.allclose(o_v, [[X**3 * Y], [(X+1)*(Y+1)]])
        assert np.allclose(o_xty, [(X+1)*(Y+1)])


def test_lambdify__sympy():
    if not have_numpy:
        return
    import sympy as sp
    _sympy_lambdify_heterogeneous_output(se.lambdify, se.DenseMatrix)
    _sympy_lambdify_heterogeneous_output(sp.lambdify, sp.Matrix)
