#!/usr/bin/env python
import os
from time import clock
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations
import symengine as se
import warnings

exprs_s = open(os.path.join(os.path.dirname(__file__), '6_links_rhs.txt'), 'tr').read()
exprs = parse_expr(exprs_s, transformations=standard_transformations)
args = sp.Matrix(1, 14, exprs).free_symbols
inp = np.ones(len(args))
assert inp.size == 26
print([expr.subs(dict(zip(args, [1]*len(args)))) for expr in exprs])

# Real-life example (ion speciation problem in water chemistry)


lmb_sp = sp.lambdify(args, exprs, modules='math')
lmb_se = se.Lambdify(args, exprs)
# lmb_se_cse = se.LambdifyCSE(args, exprs)
lmb_se_llvm = se.Lambdify(args, exprs, backend='llvm')


lmb_sp(*inp)
tim_sympy = clock()
for i in range(500):
    res_sympy = lmb_sp(*inp)
tim_sympy = clock() - tim_sympy

lmb_se(inp)
tim_se = clock()
res_se = np.empty(len(exprs))
for i in range(500):
    res_se = lmb_se(inp)
tim_se = clock() - tim_se

# lmb_se_cse(inp)
# tim_se_cse = clock()
# res_se_cse = np.empty(len(exprs))
# for i in range(500):
#     res_se_cse = lmb_se_cse(inp)
# tim_se_cse = clock() - tim_se_cse

lmb_se_llvm(inp)
tim_se_llvm = clock()
res_se_llvm = np.empty(len(exprs))
for i in range(500):
    res_se_llvm = lmb_se_llvm(inp)
tim_se_llvm = clock() - tim_se_llvm


print('SymEngine (lambda double)       speed-up factor (higher is better) vs sympy: %12.5g' %
      (tim_sympy/tim_se))

# print('symengine (lambda double + CSE) speed-up factor (higher is better) vs sympy: %12.5g' %
#       (tim_sympy/tim_se_cse))

print('symengine (LLVM)                speed-up factor (higher is better) vs sympy: %12.5g' %
      (tim_sympy/tim_se_llvm))

import itertools
from functools import reduce
from operator import mul

def ManualLLVM(inputs, *outputs):
    outputs_ravel = list(itertools.chain(*outputs))
    cb = se.Lambdify(inputs, outputs_ravel, backend="llvm")
    def func(*args):
        result = []
        n = np.empty(len(outputs_ravel))
        t = cb.unsafe_real(np.concatenate([arg.ravel() for arg in args]), n)
        start = 0
        for output in outputs:
            elems = reduce(mul, output.shape)
            result.append(n[start:start+elems].reshape(output.shape))
            start += elems
        return result
    return func

lmb_se_llvm_manual = ManualLLVM(args, np.array(exprs))
lmb_se_llvm_manual(inp)
tim_se_llvm_manual = clock()
res_se_llvm_manual = np.empty(len(exprs))
for i in range(500):
    res_se_llvm_manual = lmb_se_llvm_manual(inp)
tim_se_llvm_manual = clock() - tim_se_llvm_manual
print('symengine (ManualLLVM)          speed-up factor (higher is better) vs sympy: %12.5g' %
      (tim_sympy/tim_se_llvm_manual))

if tim_se_llvm_manual < tim_se_llvm:
    warnings.warn("Cython code for Lambdify.__call__ is slow.")

import setuptools
import pyximport
pyximport.install()
from Lambdify_6_links_reference import _benchmark_reference_for_Lambdify as lmb_ref

lmb_ref(inp)
tim_ref = clock()
for i in range(500):
    res_ref = lmb_ref(inp)
tim_ref = clock() - tim_ref
print('Hard-coded Cython code          speed-up factor (higher is better) vs sympy: %12.5g' %
      (tim_sympy/tim_ref))
