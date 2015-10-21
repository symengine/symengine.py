# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import array
import cmath
import itertools
import math
import operator
import sys

import pytest

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


import symengine as se


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


def test_Lambdify():
    n = 7
    args = x, y, z = se.symbols('x y z')
    l = se.Lambdify(args, [x+y+z, x**2, (x-y)/z, x*y*z])
    assert allclose(l(range(n, n+len(args))),
                    [3*n+3, n**2, -1/(n+2), n*(n+1)*(n+2)])


def _get_2_to_2by2_numpy():
    args = x, y = se.symbols('x y')
    exprs = np.array([[x+y+1.0, x*y],
                      [x/y, x**y]])
    l = se.Lambdify(args, exprs)

    def check(A, inp):
        X, Y = inp
        assert abs(A[0, 0] - (X+Y+1.0)) < 1e-15
        assert abs(A[0, 1] - (X*Y)) < 1e-15
        assert abs(A[1, 0] - (X/Y)) < 1e-15
        assert abs(A[1, 1] - (X**Y)) < 1e-13
    return l, check


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_Lambdify_2dim_numpy():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    lmb, check = _get_2_to_2by2_numpy()
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
    args, exprs, inp, check = _get_array()
    lmb = se.Lambdify(args, exprs)
    out = lmb(inp)
    check(out)


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires NumPy')
def test_array_out():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    if sys.version_info[0] < 3:
        return  # requires Py3
    args, exprs, inp, check = _get_array()
    lmb = se.Lambdify(args, exprs)
    out1 = array.array('d', [0]*len(exprs))
    out2 = lmb(inp, out1)
    # Ensure buffer points to still data point:
    assert out1.buffer_info()[0] == out2.__array_interface__['data'][0]
    check(out1)
    check(out2)
    out2[:] = -1
    assert np.allclose(out1[:], [-1]*len(exprs))


def test_array_out_no_numpy():
    if sys.version_info[0] < 3:
        return  # requires Py3
    args, exprs, inp, check = _get_array()
    lmb = se.Lambdify(args, exprs)
    out1 = array.array('d', [0]*len(exprs))
    out2 = lmb(inp, out1, use_numpy=False)
    # Ensure buffer points to still data point:
    assert out1.buffer_info() == out2.buffer_info()
    assert out1 is out2
    check(out1)
    check(out2)
    assert out2[0] != -1
    out1[0] = -1
    assert out2[0] == -1


def test_memview_out():
    args, exprs, inp, check = _get_array()
    lmb = se.Lambdify(args, exprs)
    cy_arr1 = lmb(inp, use_numpy=False)
    check(cy_arr1)
    cy_arr2 = lmb(inp, cy_arr1, use_numpy=False)
    check(cy_arr2)
    assert cy_arr2[0] != -1
    cy_arr1[0] = -1
    assert cy_arr2[0] == -1


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_broadcast():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    a = np.linspace(-np.pi, np.pi)
    inp = np.vstack((np.cos(a), np.sin(a))).T  # 50 rows 2 cols
    x, y = se.symbols('x y')
    distance = se.Lambdify([x, y], [se.sqrt(x**2 + y**2)])
    assert np.allclose(distance([inp[0, 0], inp[0, 1]]), [1])
    dists = distance(inp)
    assert dists.shape == (50, 1)
    assert np.allclose(dists, 1)


def _get_cse_exprs():
    import sympy as sp
    args = x, y = sp.symbols('x y')
    exprs = [x*x + y, y/(x*x), y*x*x+x]
    inp = [11, 13]
    ref = [121+13, 13/121, 13*121 + 11]
    return args, exprs, inp, ref


def test_cse_list_input():
    args, exprs, inp, ref = _get_cse_exprs()
    lmb = se.LambdifyCSE(args, exprs, concatenate=lambda tup:
                         tup[0]+list(tup[1]))
    out = lmb(inp, use_numpy=False)
    assert allclose(out, ref)


def test_cse_array_input():
    args, exprs, inp, ref = _get_cse_exprs()
    inp = array.array('d', inp)
    lmb = se.LambdifyCSE(args, exprs, concatenate=lambda tup:
                         tup[0]+array.array('d', tup[1]))
    out = lmb(inp, use_numpy=False)
    assert allclose(out, ref)


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_cse_numpy():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    args, exprs, inp, ref = _get_cse_exprs()
    lmb = se.LambdifyCSE(args, exprs)
    out = lmb(inp)
    assert allclose(out, ref)


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_broadcast_c():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    n = 3
    inp = np.arange(2*n).reshape((n, 2))
    lmb, check = _get_2_to_2by2_numpy()
    A = lmb(inp)
    assert A.shape == (3, 2, 2)
    for i in range(n):
        check(A[i, ...], inp[i, :])


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_broadcast_fortran():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    n = 3
    inp = np.arange(2*n).reshape((n, 2), order='F')
    lmb, check = _get_2_to_2by2_numpy()
    A = lmb(inp)
    assert A.shape == (3, 2, 2)
    for i in range(n):
        check(A[i, ...], inp[i, :])


def _get_1_to_2by3_matrix():
    x = se.symbols('x')
    args = x,
    exprs = se.DenseMatrix(2, 3, [x+1, x+2, x+3,
                                  1/x, 1/(x*x), 1/(x**3.0)])
    l = se.Lambdify(args, exprs)

    def check(A, inp):
        X, = inp
        assert abs(A[0, 0] - (X+1)) < 1e-15
        assert abs(A[0, 1] - (X+2)) < 1e-15
        assert abs(A[0, 2] - (X+3)) < 1e-15
        assert abs(A[1, 0] - (1/X)) < 1e-15
        assert abs(A[1, 1] - (1/(X*X))) < 1e-15
        assert abs(A[1, 2] - (1/(X**3.0))) < 1e-15
    return l, check


def _test_2dim_Matrix(use_numpy):
    l, check = _get_1_to_2by3_matrix()
    inp = [7]
    check(l(inp, use_numpy=use_numpy), inp)


def test_2dim_Matrix():
    _test_2dim_Matrix(False)


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_2dim_Matrix_numpy():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    _test_2dim_Matrix(True)


def _test_2dim_Matrix_broadcast(use_numpy):
    l, check = _get_1_to_2by3_matrix()
    inp = range(1, 5)
    out = l(inp, use_numpy=use_numpy)
    for i in range(len(inp)):
        check(out[i, ...], (inp[i],))


def test_2dim_Matrix_broadcast():
    _test_2dim_Matrix_broadcast(False)


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_2dim_Matrix_broadcast_numpy():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    _test_2dim_Matrix_broadcast(True)


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_jacobian():
    if not HAVE_NUMPY:  # nosetests work-around
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


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_excessive_args():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    x = se.symbols('x')
    lmb = se.Lambdify([x], [-x])
    inp = np.ones(2)
    out = lmb(inp)
    assert np.allclose(inp, [1, 1])
    assert len(out) == 2  # broad casting
    assert np.allclose(out, -1)


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_excessive_out():
    if not HAVE_NUMPY:  # nosetests work-around
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
        l = []
        for idx in all_indices(A.memview.shape):
            l.append(A[idx])
        return l


def _get_2_to_2by2_list(real=True):
    args = x, y = se.symbols('x y')
    exprs = [[x + y*y, y*y], [x*y*y, se.sqrt(x)+y*y]]
    l = se.Lambdify(args, exprs, real=real)

    def check(A, inp):
        X, Y = inp
        assert A.shape[-2:] == (2, 2)
        ref = [X + Y*Y, Y*Y, X*Y*Y, cmath.sqrt(X)+Y*Y]
        ravA = ravelled(A)
        size = _size(ravA)
        for i in range(size//4):
            for j in range(4):
                assert isclose(ravA[i*4 + j], ref[j])
    return l, check


def test_2_to_2by2_list():
    l, check = _get_2_to_2by2_list()
    inp = [13, 17]
    A = l(inp, use_numpy=False)
    check(A, inp)


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_2_to_2by2_numpy():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    l, check = _get_2_to_2by2_list()
    inp = [13, 17]
    A = l(inp, use_numpy=True)
    check(A, inp)


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_unsafe_real():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    l, check = _get_2_to_2by2_list()
    inp = np.array([13., 17.])
    out = np.empty(4)
    l.unsafe_real(inp, out)
    check(out.reshape((2, 2)), inp)


@pytest.mark.skipif(not HAVE_NUMPY, reason='requires numpy')
def test_unsafe_complex():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    l, check = _get_2_to_2by2_list(real=False)
    assert not l.real
    inp = np.array([13+11j, 7+4j], dtype=np.complex128)
    out = np.empty(4, dtype=np.complex128)
    l.unsafe_complex(inp, out)
    check(out.reshape((2, 2)), inp)


def test_itertools_chain():
    args, exprs, inp, check = _get_array()
    l = se.Lambdify(args, exprs)
    inp = itertools.chain([inp[0]], (inp[1],), [inp[2]])
    A = l(inp, use_numpy=False)
    check(A)


@pytest.mark.xfail(not HAVE_NUMPY, reason='array.array lacks "Zd"')
def test_complex_1():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    x = se.Symbol('x')
    lmb = se.Lambdify([x], [1j + x], real=False)
    assert abs(lmb([11+13j])[0] -
               (11 + 14j)) < 1e-15


@pytest.mark.xfail(not HAVE_NUMPY, reason='array.array lacks "Zd"')
def test_complex_2():
    if not HAVE_NUMPY:  # nosetests work-around
        return
    x = se.Symbol('x')
    lmb = se.Lambdify([x], [3 + x - 1j], real=False)
    assert abs(lmb([11+13j])[0] -
               (14 + 12j)) < 1e-15
