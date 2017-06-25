from cython.operator cimport dereference as deref, preincrement as inc
cimport symengine
from symengine cimport RCP, pair, map_basic_basic, umap_int_basic, umap_int_basic_iterator, umap_basic_num, umap_basic_num_iterator, rcp_const_basic, std_pair_short_rcp_const_basic, rcp_const_seriescoeffinterface
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from cpython cimport PyObject, Py_XINCREF, Py_XDECREF, \
    PyObject_CallMethodObjArgs
from libc.string cimport memcpy
import cython
import itertools
from operator import mul
from functools import reduce
import collections
import warnings
from symengine.compatibility import is_sequence
import os

try:
    import numpy as np
    # Lambdify requires NumPy (since b713a61, see gh-112)
    have_numpy = True
except ImportError:
    have_numpy = False

include "config.pxi"

class SympifyError(Exception):
    pass

cdef c2py(RCP[const symengine.Basic] o):
    cdef Basic r
    if (symengine.is_a_Add(deref(o))):
        r = Basic.__new__(Add)
    elif (symengine.is_a_Mul(deref(o))):
        r = Basic.__new__(Mul)
    elif (symengine.is_a_Pow(deref(o))):
        r = Basic.__new__(Pow)
    elif (symengine.is_a_Integer(deref(o))):
        r = Number.__new__(Integer)
    elif (symengine.is_a_Rational(deref(o))):
        r = Number.__new__(Rational)
    elif (symengine.is_a_Complex(deref(o))):
        r = Complex.__new__(Complex)
    elif (symengine.is_a_Symbol(deref(o))):
        if (symengine.is_a_PySymbol(deref(o))):
            return <object>(deref(symengine.rcp_static_cast_PySymbol(o)).get_py_object())
        r = Basic.__new__(Symbol)
    elif (symengine.is_a_Constant(deref(o))):
        r = Constant.__new__(Constant)
    elif (symengine.is_a_Infty(deref(o))):
        if (deref(symengine.rcp_static_cast_Infty(o)).is_positive()):
            r = Number.__new__(Infinity)
        elif (deref(symengine.rcp_static_cast_Infty(o)).is_negative()):
            r = Number.__new__(NegativeInfinity)
        else:
            r = Number.__new__(ComplexInfinity)
    elif (symengine.is_a_NaN(deref(o))):
        r = Number.__new__(NaN)
    elif (symengine.is_a_PyFunction(deref(o))):
        r = PyFunction.__new__(PyFunction)
    elif (symengine.is_a_FunctionSymbol(deref(o))):
        r = FunctionSymbol.__new__(FunctionSymbol)
    elif (symengine.is_a_Abs(deref(o))):
        r = Function.__new__(Abs)
    elif (symengine.is_a_Max(deref(o))):
        r = Function.__new__(Max)
    elif (symengine.is_a_Min(deref(o))):
        r = Function.__new__(Min)
    elif (symengine.is_a_Gamma(deref(o))):
        r = Function.__new__(Gamma)
    elif (symengine.is_a_Derivative(deref(o))):
        r = Basic.__new__(Derivative)
    elif (symengine.is_a_Subs(deref(o))):
        r = Basic.__new__(Subs)
    elif (symengine.is_a_RealDouble(deref(o))):
        r = Number.__new__(RealDouble)
    elif (symengine.is_a_ComplexDouble(deref(o))):
        r = ComplexDouble.__new__(ComplexDouble)
    elif (symengine.is_a_RealMPFR(deref(o))):
        r = Number.__new__(RealMPFR)
    elif (symengine.is_a_ComplexMPC(deref(o))):
        r = ComplexMPC.__new__(ComplexMPC)
    elif (symengine.is_a_Log(deref(o))):
        r = Function.__new__(Log)
    elif (symengine.is_a_Sin(deref(o))):
        r = Function.__new__(Sin)
    elif (symengine.is_a_Cos(deref(o))):
        r = Function.__new__(Cos)
    elif (symengine.is_a_Tan(deref(o))):
        r = Function.__new__(Tan)
    elif (symengine.is_a_Cot(deref(o))):
        r = Function.__new__(Cot)
    elif (symengine.is_a_Csc(deref(o))):
        r = Function.__new__(Csc)
    elif (symengine.is_a_Sec(deref(o))):
        r = Function.__new__(Sec)
    elif (symengine.is_a_ASin(deref(o))):
        r = Function.__new__(ASin)
    elif (symengine.is_a_ACos(deref(o))):
        r = Function.__new__(ACos)
    elif (symengine.is_a_ATan(deref(o))):
        r = Function.__new__(ATan)
    elif (symengine.is_a_ACot(deref(o))):
        r = Function.__new__(ACot)
    elif (symengine.is_a_ACsc(deref(o))):
        r = Function.__new__(ACsc)
    elif (symengine.is_a_ASec(deref(o))):
        r = Function.__new__(ASec)
    elif (symengine.is_a_Sinh(deref(o))):
        r = Function.__new__(Sinh)
    elif (symengine.is_a_Cosh(deref(o))):
        r = Function.__new__(Cosh)
    elif (symengine.is_a_Tanh(deref(o))):
        r = Function.__new__(Tanh)
    elif (symengine.is_a_Coth(deref(o))):
        r = Function.__new__(Coth)
    elif (symengine.is_a_Csch(deref(o))):
        r = Function.__new__(Csch)
    elif (symengine.is_a_Sech(deref(o))):
        r = Function.__new__(Sech)
    elif (symengine.is_a_ASinh(deref(o))):
        r = Function.__new__(ASinh)
    elif (symengine.is_a_ACosh(deref(o))):
        r = Function.__new__(ACosh)
    elif (symengine.is_a_ATanh(deref(o))):
        r = Function.__new__(ATanh)
    elif (symengine.is_a_ACoth(deref(o))):
        r = Function.__new__(ACoth)
    elif (symengine.is_a_ACsch(deref(o))):
        r = Function.__new__(ACsch)
    elif (symengine.is_a_ASech(deref(o))):
        r = Function.__new__(ASech)
    elif (symengine.is_a_ATan2(deref(o))):
        r = Function.__new__(ATan2)
    elif (symengine.is_a_PyNumber(deref(o))):
        r = PyNumber.__new__(PyNumber)
    else:
        raise Exception("Unsupported SymEngine class.")
    r.thisptr = o
    return r

def sympy2symengine(a, raise_error=False):
    """
    Converts 'a' from SymPy to SymEngine.

    If the expression cannot be converted, it either returns None (if
    raise_error==False) or raises a SympifyError exception (if
    raise_error==True).
    """
    import sympy
    from sympy.core.function import AppliedUndef as sympy_AppliedUndef
    if isinstance(a, sympy.Symbol):
        return Symbol(a.name)
    elif isinstance(a, sympy.Mul):
        return mul(*[sympy2symengine(x, raise_error) for x in a.args])
    elif isinstance(a, sympy.Add):
        return add(*[sympy2symengine(x, raise_error) for x in a.args])
    elif isinstance(a, (sympy.Pow, sympy.exp)):
        x, y = a.as_base_exp()
        return sympy2symengine(x, raise_error) ** sympy2symengine(y, raise_error)
    elif isinstance(a, sympy.Integer):
        return Integer(a.p)
    elif isinstance(a, sympy.Rational):
        return Integer(a.p) / Integer(a.q)
    elif isinstance(a, sympy.Float):
        IF HAVE_SYMENGINE_MPFR:
            if a._prec > 53:
                return RealMPFR(str(a), a._prec)
            else:
                return RealDouble(float(str(a)))
        ELSE:
            return RealDouble(float(str(a)))
    elif a is sympy.I:
        return I
    elif a is sympy.E:
        return E
    elif a is sympy.pi:
        return pi
    elif a is sympy.S.NegativeInfinity:
        return -oo
    elif a is sympy.S.Infinity:
        return oo
    elif a is sympy.S.ComplexInfinity:
        return zoo
    elif a is sympy.nan:
        return nan
    elif isinstance(a, sympy.functions.elementary.trigonometric.TrigonometricFunction):
        if isinstance(a, sympy.sin):
            return sin(a.args[0])
        elif isinstance(a, sympy.cos):
            return cos(a.args[0])
        elif isinstance(a, sympy.tan):
            return tan(a.args[0])
        elif isinstance(a, sympy.cot):
            return cot(a.args[0])
        elif isinstance(a, sympy.csc):
            return csc(a.args[0])
        elif isinstance(a, sympy.sec):
            return sec(a.args[0])
    elif isinstance(a, sympy.functions.elementary.trigonometric.InverseTrigonometricFunction):
        if isinstance(a, sympy.asin):
            return asin(a.args[0])
        elif isinstance(a, sympy.acos):
            return acos(a.args[0])
        elif isinstance(a, sympy.atan):
            return atan(a.args[0])
        elif isinstance(a, sympy.acot):
            return acot(a.args[0])
        elif isinstance(a, sympy.acsc):
            return acsc(a.args[0])
        elif isinstance(a, sympy.asec):
            return asec(a.args[0])
        elif isinstance(a, sympy.atan2):
            return atan2(*a.args)
    elif isinstance(a, sympy.functions.elementary.hyperbolic.HyperbolicFunction):
        if isinstance(a, sympy.sinh):
            return sinh(a.args[0])
        elif isinstance(a, sympy.cosh):
            return cosh(a.args[0])
        elif isinstance(a, sympy.tanh):
            return tanh(a.args[0])
        elif isinstance(a, sympy.coth):
            return coth(a.args[0])
        elif isinstance(a, sympy.csch):
            return csch(a.args[0])
        elif isinstance(a, sympy.sech):
            return sech(a.args[0])
    elif isinstance(a, sympy.asinh):
        return asinh(a.args[0])
    elif isinstance(a, sympy.acosh):
        return acosh(a.args[0])
    elif isinstance(a, sympy.atanh):
        return atanh(a.args[0])
    elif isinstance(a, sympy.acoth):
        return acoth(a.args[0])
    elif isinstance(a, sympy.log):
        return log(a.args[0])
    elif isinstance(a, sympy.Abs):
        return abs(sympy2symengine(a.args[0], raise_error))
    elif isinstance(a, sympy.Max):
        return _max(*a.args)
    elif isinstance(a, sympy.Min):
        return _min(*a.args)
    elif isinstance(a, sympy.gamma):
        return gamma(a.args[0])
    elif isinstance(a, sympy.Derivative):
        return Derivative(a.expr, *a.variables)
    elif isinstance(a, sympy.Subs):
        return Subs(a.expr, a.variables, a.point)
    elif isinstance(a, sympy_AppliedUndef):
        name = str(a.func)
        return function_symbol(name, *(a.args))
    elif isinstance(a, sympy.Function):
        return PyFunction(a, a.args, a.func, sympy_module)
    elif isinstance(a, sympy.MatrixBase):
        row, col = a.shape
        v = []
        for r in a.tolist():
            for e in r:
                v.append(e)
        if isinstance(a, sympy.MutableDenseMatrix):
            return MutableDenseMatrix(row, col, v)
        elif isinstance(a, sympy.ImmutableDenseMatrix):
            return ImmutableDenseMatrix(row, col, v)
        else:
            raise NotImplementedError
    elif isinstance(a, sympy.polys.domains.modularinteger.ModularInteger):
        return PyNumber(a, sympy_module)
    elif sympy.__version__ > '1.0':
        if isinstance(a, sympy.acsch):
            return acsch(a.args[0])
        elif isinstance(a, sympy.asech):
            return asech(a.args[0])

    if raise_error:
        raise SympifyError("sympy2symengine: Cannot convert '%r' to a symengine type." % a)


def sympify(a):
    """
    Converts an expression 'a' into a SymEngine type.

    Arguments
    =========

    a ............. An expression to convert.

    Examples
    ========

    >>> from symengine import sympify
    >>> sympify(1)
    1
    >>> sympify("a+b")
    a + b
    """
    if isinstance(a, str):
        return c2py(symengine.parse(a.encode("utf-8")))
    return _sympify(a, True)


def _sympify(a, raise_error=True):
    """
    Converts an expression 'a' into a SymEngine type.

    Arguments
    =========

    a ............. An expression to convert.
    raise_error ... Will raise an error on a failure (default True), otherwise
                    it returns None if 'a' cannot be converted.

    Examples
    ========

    >>> from symengine.li.symengine_wrapper import _sympify
    >>> _sympify(1)
    1
    >>> _sympify("abc", False)
    >>>

    """
    if isinstance(a, (Basic, MatrixBase)):
        return a
    elif isinstance(a, (int, long)):
        return Integer(a)
    elif isinstance(a, float):
        return RealDouble(a)
    elif isinstance(a, complex):
        return ComplexDouble(a)
    elif isinstance(a, tuple):
        v = []
        for e in a:
            v.append(_sympify(e, True))
        return tuple(v)
    elif isinstance(a, list):
        v = []
        for e in a:
            v.append(_sympify(e, True))
        return v
    elif hasattr(a, '_symengine_'):
        return _sympify(a._symengine_(), raise_error)
    elif hasattr(a, '_sympy_'):
        return _sympify(a._sympy_(), raise_error)
    elif hasattr(a, 'pyobject'):
        return _sympify(a.pyobject(), raise_error)
    return sympy2symengine(a, raise_error)

funcs = {}

def get_function_class(function, module):
    if not function in funcs:
        funcs[function] = PyFunctionClass(function, module)
    return funcs[function]


cdef class DictBasicIter(object):

    cdef init(self, map_basic_basic.iterator begin, map_basic_basic.iterator end):
        self.begin = begin
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.begin != self.end:
            obj = c2py(deref(self.begin).first)
        else:
            raise StopIteration
        inc(self.begin)
        return obj


cdef class _DictBasic(object):

    def __init__(self, tocopy = None):
        if tocopy != None:
            self.add_dict(tocopy)

    def as_dict(self):
        ret = {}
        it = self.c.begin()
        while it != self.c.end():
            ret[c2py(deref(it).first)] = c2py(deref(it).second)
            inc(it)
        return ret

    def add_dict(self, d):
        cdef _DictBasic D
        if isinstance(d, DictBasic):
            D = d
            self.c.insert(D.c.begin(), D.c.end())
        else:
            for key, value in d.iteritems():
                self.add(key, value)

    def add(self, key, value):
        cdef Basic K = sympify(key)
        cdef Basic V = sympify(value)
        cdef symengine.std_pair_rcp_const_basic_rcp_const_basic pair
        pair.first = K.thisptr
        pair.second = V.thisptr
        return self.c.insert(pair).second

    def copy(self):
        return DictBasic(self)

    __copy__ = copy

    def __len__(self):
        return self.c.size()

    def __getitem__(self, key):
        cdef Basic K = sympify(key)
        it = self.c.find(K.thisptr)
        if it == self.c.end():
            raise KeyError(key)
        else:
            return c2py(deref(it).second)

    def __setitem__(self, key, value):
        cdef Basic K = sympify(key)
        cdef Basic V = sympify(value)
        self.c[K.thisptr] = V.thisptr

    def clear(self):
        self.clear()

    def __delitem__(self, key):
        cdef Basic K = sympify(key)
        self.c.erase(K.thisptr)

    def __contains__(self, key):
        cdef Basic K = sympify(key)
        it = self.c.find(K.thisptr)
        return it != self.c.end()

    def __iter__(self):
        cdef DictBasicIter d = DictBasicIter()
        d.init(self.c.begin(), self.c.end())
        return d


class DictBasic(_DictBasic, collections.MutableMapping):

    def __str__(self):
        return "{" + ", ".join(["%s: %s" % (str(key), str(value)) for key, value in self.items()]) + "}"

    def __repr__(self):
        return self.__str__()

def get_dict(*args):
    """
    Returns a DictBasic instance from args. Inputs can be,
        1. a DictBasic
        2. a Python dictionary
        3. two args old, new
    """
    cdef _DictBasic D = DictBasic()
    if len(args) == 2:
        if is_sequence(args[0]):
            for k, v in zip(args[0], args[1]):
                D.add(k, v)
        else:
            D.add(args[0], args[1])
        return D
    elif len(args) == 1:
        arg = args[0]
    else:
        raise TypeError("subs/msubs takes one or two arguments (%d given)" % \
                len(args))
    if isinstance(arg, DictBasic):
        return arg
    for k, v in arg.items():
        D.add(k, v)
    return D


cdef tuple vec_basic_to_tuple(symengine.vec_basic& vec):
    result = []
    for i in range(vec.size()):
        result.append(c2py(<RCP[const symengine.Basic]>(vec[i])))
    return tuple(result)


cdef class Basic(object):

    def __str__(self):
        return deref(self.thisptr).__str__().decode("utf-8")

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return deref(self.thisptr).hash()

    def __dealloc__(self):
        self.thisptr.reset()

    def __add__(a, b):
        cdef Basic A = _sympify(a, False)
        B_ = _sympify(b, False)
        if A is None or B_ is None or isinstance(B_, MatrixBase): return NotImplemented
        cdef Basic B = B_
        return c2py(symengine.add(A.thisptr, B.thisptr))

    def __sub__(a, b):
        cdef Basic A = _sympify(a, False)
        B_ = _sympify(b, False)
        if A is None or B_ is None or isinstance(B_, MatrixBase): return NotImplemented
        cdef Basic B = B_
        return c2py(symengine.sub(A.thisptr, B.thisptr))

    def __mul__(a, b):
        cdef Basic A = _sympify(a, False)
        B_ = _sympify(b, False)
        if A is None or B_ is None or isinstance(B_, MatrixBase): return NotImplemented
        cdef Basic B = B_
        return c2py(symengine.mul(A.thisptr, B.thisptr))

    def __truediv__(a, b):
        cdef Basic A = _sympify(a, False)
        cdef Basic B = _sympify(b, False)
        if A is None or B is None: return NotImplemented
        return c2py(symengine.div(A.thisptr, B.thisptr))

    # This is for Python 2.7 compatibility only:
    def __div__(a, b):
        cdef Basic A = _sympify(a, False)
        cdef Basic B = _sympify(b, False)
        if A is None or B is None: return NotImplemented
        return c2py(symengine.div(A.thisptr, B.thisptr))

    def __pow__(a, b, c):
        if c is not None:
            return powermod(a, b, c)
        cdef Basic A = _sympify(a, False)
        cdef Basic B = _sympify(b, False)
        if A is None or B is None: return NotImplemented
        return c2py(symengine.pow(A.thisptr, B.thisptr))

    def __neg__(Basic self not None):
        return c2py(symengine.neg(self.thisptr))

    def __abs__(Basic self not None):
        return c2py(symengine.abs(self.thisptr))

    def __richcmp__(a, b, int op):
        A = _sympify(a, False)
        B = _sympify(b, False)
        if not (isinstance(A, Basic) and isinstance(B, Basic)):
            if (op == 2):
                return False
            elif (op == 3):
                return True
            else:
                return NotImplemented
        return Basic._richcmp_(A, B, op)

    def _richcmp_(Basic A, Basic B, int op):
        if (op == 2):
            return symengine.eq(deref(A.thisptr), deref(B.thisptr))
        elif (op == 3):
            return symengine.neq(deref(A.thisptr), deref(B.thisptr))
        from sympy import Rel
        if (op == 0):
            return Rel(A, B, '<')
        elif (op == 1):
            return Rel(A, B, '<=')
        elif (op == 4):
            return Rel(A, B, '>')
        elif (op == 5):
            return Rel(A, B, '>=')

    def expand(Basic self not None):
        return c2py(symengine.expand(self.thisptr))

    def diff(Basic self not None, x = None):
        if x is None:
            f = self.free_symbols
            if (len(f) != 1):
                raise RuntimeError("Variable w.r.t should be given")
            return self.diff(f.pop())
        cdef Basic s = sympify(x)
        return c2py(symengine.diff(self.thisptr, s.thisptr))

    def subs_dict(Basic self not None, *args):
        warnings.warn("subs_dict() is deprecated. Use subs() instead", DeprecationWarning)
        return self.subs(*args)

    def subs_oldnew(Basic self not None, old, new):
        warnings.warn("subs_oldnew() is deprecated. Use subs() instead", DeprecationWarning)
        return self.subs({old: new})

    def subs(Basic self not None, *args):
        cdef _DictBasic D = get_dict(*args)
        return c2py(symengine.ssubs(self.thisptr, D.c))

    xreplace = subs

    def msubs(Basic self not None, *args):
        cdef _DictBasic D = get_dict(*args)
        return c2py(symengine.msubs(self.thisptr, D.c))

    def n(self, prec = 53, real = False):
        if real:
            return eval_real(self, prec)
        else:
            return eval(self, prec)

    @property
    def args(self):
        cdef symengine.vec_basic args = deref(self.thisptr).get_args()
        return vec_basic_to_tuple(args)

    @property
    def free_symbols(self):
        cdef symengine.set_basic _set = symengine.free_symbols(deref(self.thisptr))
        return {c2py(<RCP[const symengine.Basic]>(elem)) for elem in _set}

    @property
    def is_Atom(self):
        return False

    @property
    def is_Symbol(self):
        return False

    @property
    def is_Function(self):
        return False

    @property
    def is_Add(self):
        return False

    @property
    def is_Mul(self):
        return False

    @property
    def is_Pow(self):
        return False

    @property
    def is_Number(self):
        return False

    @property
    def is_Float(self):
        return False

    @property
    def is_Rational(self):
        return False

    @property
    def is_Integer(self):
        return False

    @property
    def is_Derivative(self):
        return False

    @property
    def is_Matrix(self):
        return False

    def _symbolic_(self, ring):
        return ring(self._sage_())

    def atoms(self, *types):
        if types:
            s = set()
            if (isinstance(self, types)):
                s.add(self)
            for arg in self.args:
                s.update(arg.atoms(*types))
            return s
        else:
            return self.free_symbols

    def simplify(self, *args, **kwargs):
        return sympify(self._sympy_().simplify(*args, **kwargs))

    def as_coefficients_dict(self):
        d = collections.defaultdict(int)
        d[self] = 1
        return d

    def coeff(self, x, n=1):
        cdef Basic _x = sympify(x)
        require(_x, Symbol)
        cdef Basic _n = sympify(n)
        return c2py(symengine.coeff(deref(self.thisptr), deref(_x.thisptr), deref(_n.thisptr)))

    def has(self, *symbols):
        return any([has_symbol(self, symbol) for symbol in symbols])

    def args_as_sage(Basic self):
        cdef symengine.vec_basic Y = deref(self.thisptr).get_args()
        s = []
        for i in range(Y.size()):
            s.append(c2py(<RCP[const symengine.Basic]>(Y[i]))._sage_())
        return s

    def args_as_sympy(Basic self):
        cdef symengine.vec_basic Y = deref(self.thisptr).get_args()
        s = []
        for i in range(Y.size()):
            s.append(c2py(<RCP[const symengine.Basic]>(Y[i]))._sympy_())
        return s

def series(ex, x=None, x0=0, n=6, as_deg_coef_pair=False):
    # TODO: check for x0 an infinity, see sympy/core/expr.py
    # TODO: nonzero x0
    # underscored local vars are of symengine.py type
    cdef Basic _ex = sympify(ex)
    syms = _ex.free_symbols
    if not syms:
        return _ex

    cdef Basic _x
    if x is None:
        _x = list(syms)[0]
    else:
        _x = sympify(x)
    require(_x, Symbol)
    if not _x in syms:
        return _ex

    if x0 != 0:
        _ex = _ex.subs({_x: _x + x0})

    cdef RCP[const symengine.Symbol] X = symengine.rcp_static_cast_Symbol(_x.thisptr)
    cdef umap_int_basic umap
    cdef umap_int_basic_iterator iter, iterend

    if not as_deg_coef_pair:
        b = c2py(<symengine.RCP[const symengine.Basic]>deref(symengine.series(_ex.thisptr, X, n)).as_basic())
        if x0 != 0:
            b = b.subs({_x: _x - x0})
        return b

    umap = deref(symengine.series(_ex.thisptr, X, n)).as_dict()

    iter = umap.begin()
    iterend = umap.end()
    poly = 0
    l = []
    while iter != iterend:
        l.append([deref(iter).first, c2py(<symengine.RCP[const symengine.Basic]>(deref(iter).second))])
        inc(iter)
    if as_deg_coef_pair:
        return l
    return add(*l)


class Symbol(Basic):

    """
    Symbol is a class to store a symbolic variable with a given name.
    """

    def __init__(Basic self, name, *args, **kwargs):
        if type(self) == Symbol:
            self.thisptr = symengine.make_rcp_Symbol(name.encode("utf-8"))
        else:
            self.thisptr = symengine.make_rcp_PySymbol(name.encode("utf-8"), <PyObject*>self)

    def _sympy_(self):
        import sympy
        return sympy.Symbol(str(self))

    def _sage_(self):
        import sage.all as sage
        return sage.SR.symbol(str(self))

    @property
    def name(self):
        return self.__str__()

    @property
    def is_Atom(self):
        return True

    @property
    def is_Symbol(self):
        return True

    @property
    def func(self):
        return self.__class__


def symarray(prefix, shape, **kwargs):
    """ Creates an nd-array of symbols

    Parameters
    ----------
    prefix: str
    shape: tuple
    \*\*kwargs:
        Passed on to :class:`Symbol`.

    Notes
    -----
    This function requires NumPy.

    """
    import numpy as np
    arr = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        arr[index] = Symbol('%s_%s' % (prefix, '_'.join(map(str, index))), **kwargs)
    return arr


cdef class Constant(Basic):

    def __cinit__(self, name = None):
        if name is None:
            return
        self.thisptr = symengine.make_rcp_Constant(name.encode("utf-8"))

    def _sympy_(self):
        import sympy
        if self == E:
            return sympy.E
        elif self == pi:
            return sympy.pi
        else:
            raise Exception("Unknown Constant")

    def _sage_(self):
        import sage.all as sage
        if self == E:
            return sage.e
        elif self == pi:
            return sage.pi
        else:
            raise Exception("Unknown Constant")


cdef class Number(Basic):
    @property
    def is_Atom(self):
        return True

    @property
    def is_Number(self):
        return True

    @property
    def is_positive(Basic self):
        return deref(symengine.rcp_static_cast_Number(self.thisptr)).is_positive()

    @property
    def is_negative(Basic self):
        return deref(symengine.rcp_static_cast_Number(self.thisptr)).is_negative()

    @property
    def is_zero(Basic self):
        return deref(symengine.rcp_static_cast_Number(self.thisptr)).is_zero()

    @property
    def is_nonzero(self):
        return not (self.is_complex or self.is_zero)

    @property
    def is_nonnegative(self):
        return not (self.is_complex or self.is_negative)

    @property
    def is_nonpositive(self):
        return not (self.is_complex or self.is_positive)

    @property
    def is_complex(Basic self):
        return deref(symengine.rcp_static_cast_Number(self.thisptr)).is_complex()


class Rational(Number):

    def __new__(cls, p, q):
        return Integer(p)/q

    @property
    def is_Rational(self):
        return True

    @property
    def p(self):
        return self.get_num_den()[0]

    @property
    def q(self):
        return self.get_num_den()[1]

    def get_num_den(Basic self):
        cdef RCP[const symengine.Integer] _num, _den
        symengine.get_num_den(deref(symengine.rcp_static_cast_Rational(self.thisptr)),
                           symengine.outArg_Integer(_num), symengine.outArg_Integer(_den))
        return [c2py(<RCP[const symengine.Basic]>_num), c2py(<RCP[const symengine.Basic]>_den)]

    def _sympy_(self):
        rat = self.get_num_den()
        return rat[0]._sympy_() / rat[1]._sympy_()

    def _sage_(self):
        try:
            from sage.symbolic.symengine_conversions import convert_to_rational
            return convert_to_rational(self)
        except ImportError:
            rat = self.get_num_den()
            return rat[0]._sage_() / rat[1]._sage_()

    @property
    def func(self):
        return self.__class__

class Integer(Rational):

    def __new__(cls, i):
        i = int(i)
        cdef int i_
        cdef symengine.integer_class i__
        cdef string tmp
        try:
            # Try to convert "i" to int
            i_ = i
            int_ok = True
        except OverflowError:
            # Too big, need to use mpz
            int_ok = False
            tmp = str(i).encode("utf-8")
            i__ = symengine.integer_class(tmp)
        # Note: all other exceptions are left intact
        if int_ok:
            return c2py(<RCP[const symengine.Basic]>symengine.integer(i_))
        else:
            return c2py(<RCP[const symengine.Basic]>symengine.integer(i__))

    @property
    def is_Integer(self):
        return True

    def __hash__(Basic self):
        return deref(self.thisptr).hash()

    def __richcmp__(a, b, int op):
        A = _sympify(a, False)
        B = _sympify(b, False)
        if not (isinstance(A, Integer) and isinstance(B, Integer)):
            if (op == 2):
                return False
            elif (op == 3):
                return True
            return NotImplemented
        return Integer._richcmp_(A, B, op)

    def _richcmp_(Basic A, Basic B, int op):
        cdef int i = deref(symengine.rcp_static_cast_Integer(A.thisptr)).compare(deref(symengine.rcp_static_cast_Integer(B.thisptr)))
        if (op == 0):
            return i < 0
        elif (op == 1):
            return i <= 0
        elif (op == 2):
            return i == 0
        elif (op == 3):
            return i != 0
        elif (op == 4):
            return i > 0
        elif (op == 5):
            return i >= 0
        else:
            return NotImplemented

    def __floordiv__(x, y):
        return quotient(x, y)

    def __mod__(x, y):
        return mod(x, y)

    def __divmod__(x, y):
        return quotient_mod(x, y)

    def _sympy_(Basic self):
        import sympy
        return sympy.Integer(deref(self.thisptr).__str__().decode("utf-8"))

    def _sage_(Basic self):
        try:
            from sage.symbolic.symengine_conversions import convert_to_integer
            return convert_to_integer(self)
        except ImportError:
            import sage.all as sage
            return sage.Integer(str(self))

    def __int__(self):
        return int(str(self))

    def __long__(self):
        return long(str(self))

    def __float__(self):
        return float(str(self))

    @property
    def p(self):
        return int(self)

    @property
    def q(self):
        return 1

    def get_num_den(Basic self):
        return self, 1

    @property
    def func(self):
        return self.__class__


def dps_to_prec(n):
    """Return the number of bits required to represent n decimals accurately."""
    return max(1, int(round((int(n)+1)*3.3219280948873626)))


class BasicMeta(type):
    def __instancecheck__(self, instance):
        return isinstance(instance, self._classes)

class Float(Number):

    def __new__(cls, num, dps=None, precision=None):
        if cls is not Float:
            return super(Float, cls).__new__(cls)

        if dps is not None and precision is not None:
            raise ValueError('Both decimal and binary precision supplied. '
                             'Supply only one. ')
        if dps is None and precision is None:
            dps = 15
        if precision is None:
            precision = dps_to_prec(dps)

        IF HAVE_SYMENGINE_MPFR:
            if precision > 53:
                if isinstance(num, RealMPFR) and precision == num.get_prec():
                    return num
                return RealMPFR(str(num), precision)
        if precision > 53:
            raise ValueError('RealMPFR unavailable for high precision numerical values.')
        elif isinstance(num, RealDouble):
            return num
        else:
            return RealDouble(float(num))


RealNumber = Float


class RealDouble(Float):

    @property
    def is_Float(self):
        return True

    def __new__(cls, i):
        cdef double i_ = i
        return c2py(symengine.make_rcp_RealDouble(i_))

    def _sympy_(Basic self):
        import sympy
        return sympy.Float(deref(self.thisptr).__str__().decode("utf-8"))

    def _sage_(Basic self):
        import sage.all as sage
        cdef double i = deref(symengine.rcp_static_cast_RealDouble(self.thisptr)).as_double()
        return sage.RealDoubleField()(i)

    def __float__(self):
        return float(str(self))


cdef class ComplexDouble(Number):

    def __cinit__(self, i = None):
        if i is None:
            return
        cdef double complex i_ = i
        self.thisptr = symengine.make_rcp_ComplexDouble(i_)

    def real_part(Basic self):
        return c2py(<RCP[const symengine.Basic]>deref(symengine.rcp_static_cast_ComplexDouble(self.thisptr)).real_part())

    def imaginary_part(Basic self):
        return c2py(<RCP[const symengine.Basic]>deref(symengine.rcp_static_cast_ComplexDouble(self.thisptr)).imaginary_part())

    def _sympy_(self):
        import sympy
        return self.real_part()._sympy_() + sympy.I * self.imaginary_part()._sympy_()

    def _sage_(self):
        import sage.all as sage
        return self.real_part()._sage_() + sage.I * self.imaginary_part()._sage_()


class RealMPFR(Float):

    @property
    def is_Float(self):
        return True

    IF HAVE_SYMENGINE_MPFR:
        def __new__(cls, i = None, long prec = 53, unsigned base = 10):
            if i is None:
                return
            cdef string i_ = str(i).encode("utf-8")
            cdef symengine.mpfr_class m
            m = symengine.mpfr_class(i_, prec, base)
            return c2py(<RCP[const symengine.Basic]>symengine.real_mpfr(symengine.std_move_mpfr(m)))

        def get_prec(Basic self):
            return Integer(deref(symengine.rcp_static_cast_RealMPFR(self.thisptr)).get_prec())

        def _sympy_(self):
            import sympy
            cdef long prec_ = self.get_prec()
            prec = max(1, int(round(prec_/3.3219280948873626)-1))
            return sympy.Float(str(self), prec)

        def _sage_(self):
            try:
                from sage.symbolic.symengine_conversions import convert_to_real_number
                return convert_to_real_number(self)
            except ImportError:
                import sage.all as sage
                return sage.RealField(int(self.get_prec()))(str(self))

        def __float__(self):
            return float(str(self))
    ELSE:
        pass


cdef class ComplexMPC(Number):
    IF HAVE_SYMENGINE_MPC:
        def __cinit__(self, i = None, j = 0, long prec = 53, unsigned base = 10):
            if i is None:
                return
            cdef string i_ = ("(" + str(i) + " " + str(j) + ")").encode("utf-8")
            cdef symengine.mpc_class m = symengine.mpc_class(i_, prec, base)
            self.thisptr = <RCP[const symengine.Basic]>symengine.complex_mpc(symengine.std_move_mpc(m))

        def real_part(self):
            return c2py(<RCP[const symengine.Basic]>deref(symengine.rcp_static_cast_ComplexMPC(self.thisptr)).real_part())

        def imaginary_part(self):
            return c2py(<RCP[const symengine.Basic]>deref(symengine.rcp_static_cast_ComplexMPC(self.thisptr)).imaginary_part())

        def _sympy_(self):
            import sympy
            return self.real_part()._sympy_() + sympy.I * self.imaginary_part()._sympy_()

        def _sage_(self):
            try:
                from sage.symbolic.symengine_conversions import convert_to_mpcomplex_number
                return convert_to_mpcomplex_number(self)
            except ImportError:
                import sage.all as sage
                return sage.MPComplexField(int(self.get_prec()))(str(self.real_part()), str(self.imaginary_part()))
    ELSE:
        pass


cdef class Complex(Number):

    def real_part(self):
        return c2py(<RCP[const symengine.Basic]>deref(symengine.rcp_static_cast_Complex(self.thisptr)).real_part())

    def imaginary_part(self):
        return c2py(<RCP[const symengine.Basic]>deref(symengine.rcp_static_cast_Complex(self.thisptr)).imaginary_part())

    def _sympy_(self):
        import sympy
        return self.real_part()._sympy_() + sympy.I * self.imaginary_part()._sympy_()

    def _sage_(self):
        import sage.all as sage
        return self.real_part()._sage_() + sage.I * self.imaginary_part()._sage_()

class Infinity(Number):

    def __new__(cls):
        return oo

    def _sympy_(self):
        import sympy
        return sympy.oo

    def _sage_(self):
        import sage.all as sage
        return sage.oo

class NegativeInfinity(Number):

    def __new__(cls):
        return -oo

    def _sympy_(self):
        import sympy
        return -sympy.oo

    def _sage_(self):
        import sage.all as sage
        return -sage.oo

class ComplexInfinity(Number):

    def __new__(cls):
        return zoo

    def _sympy_(self):
        import sympy
        return sympy.zoo

    def _sage_(self):
        import sage.all as sage
        return sage.unsigned_infinity

class NaN(Number):

    def __new__(cls):
        return nan

    def _sympy_(self):
        import sympy
        return sympy.nan

    def _sage_(self):
        import sage.all as sage
        return sage.NaN

class Add(Basic):

    def __new__(cls, *args, **kwargs):
        cdef symengine.vec_basic v_
        cdef Basic e
        for e_ in args:
            e = _sympify(e_)
            v_.push_back(e.thisptr)
        return c2py(symengine.add(v_))

    @property
    def is_Add(self):
        return True

    def _sympy_(self):
        from sympy import Add
        return Add(*self.args)

    def _sage_(Basic self):
        cdef RCP[const symengine.Add] X = symengine.rcp_static_cast_Add(self.thisptr)
        cdef RCP[const symengine.Basic] a, b
        deref(X).as_two_terms(symengine.outArg(a), symengine.outArg(b))
        return c2py(a)._sage_() + c2py(b)._sage_()

    @property
    def func(self):
        return self.__class__

    def as_coefficients_dict(Basic self):
        cdef RCP[const symengine.Add] X = symengine.rcp_static_cast_Add(self.thisptr)
        cdef umap_basic_num umap
        cdef umap_basic_num_iterator iter, iterend
        d = collections.defaultdict(int)
        d[Integer(1)] = c2py(<RCP[const symengine.Basic]>(deref(X).get_coef()))
        umap = deref(X).get_dict()
        iter = umap.begin()
        iterend = umap.end()
        while iter != iterend:
            d[c2py(<RCP[const symengine.Basic]>(deref(iter).first))] =\
                    c2py(<RCP[const symengine.Basic]>(deref(iter).second))
            inc(iter)
        return d

class Mul(Basic):

    def __new__(cls, *args, **kwargs):
        cdef symengine.vec_basic v_
        cdef Basic e
        for e_ in args:
            e = _sympify(e_)
            v_.push_back(e.thisptr)
        return c2py(symengine.mul(v_))

    @property
    def is_Mul(self):
        return True

    def _sympy_(self):
        from sympy import Mul
        return Mul(*self.args)

    def _sage_(Basic self):
        cdef RCP[const symengine.Mul] X = symengine.rcp_static_cast_Mul(self.thisptr)
        cdef RCP[const symengine.Basic] a, b
        deref(X).as_two_terms(symengine.outArg(a), symengine.outArg(b))
        return c2py(a)._sage_() * c2py(b)._sage_()

    @property
    def func(self):
        return self.__class__

    def as_coefficients_dict(Basic self):
        cdef RCP[const symengine.Mul] X = symengine.rcp_static_cast_Mul(self.thisptr)
        cdef RCP[const symengine.Integer] one = symengine.integer(1)
        cdef map_basic_basic dict = deref(X).get_dict()
        d = collections.defaultdict(int)
        d[c2py(<RCP[const symengine.Basic]>symengine.mul_from_dict(\
                <RCP[const symengine.Number]>(one),
                symengine.std_move_map_basic_basic(dict)))] =\
                c2py(<RCP[const symengine.Basic]>deref(X).get_coef())
        return d


class Pow(Basic):

    def __new__(cls, a, b):
        return _sympify(a) ** b

    @property
    def is_Pow(self):
        return True

    def _sympy_(Basic self):
        cdef RCP[const symengine.Pow] X = symengine.rcp_static_cast_Pow(self.thisptr)
        base = c2py(deref(X).get_base())
        exp = c2py(deref(X).get_exp())
        return base._sympy_() ** exp._sympy_()

    def _sage_(Basic self):
        cdef RCP[const symengine.Pow] X = symengine.rcp_static_cast_Pow(self.thisptr)
        base = c2py(deref(X).get_base())
        exp = c2py(deref(X).get_exp())
        return base._sage_() ** exp._sage_()

    @property
    def func(self):
        return self.__class__


class Function(Basic):

    def __new__(cls, *args, **kwargs):
        if cls == Function and len(args) == 1:
            return UndefFunction(args[0])
        return super(Function, cls).__new__(cls)

    @property
    def is_Function(self):
        return True

    def func(self, *values):
        import sys
        return getattr(sys.modules[__name__], self.__class__.__name__.lower())(*values)


class OneArgFunction(Function):

    def get_arg(Basic self):
        cdef RCP[const symengine.OneArgFunction] X = symengine.rcp_static_cast_OneArgFunction(self.thisptr)
        return c2py(deref(X).get_arg())

    def _sympy_(self):
        import sympy
        return getattr(sympy, self.__class__.__name__.lower())(self.get_arg()._sympy_())

    def _sage_(self):
        import sage.all as sage
        return getattr(sage, self.__class__.__name__.lower())(self.get_arg()._sage_())


class HyperbolicFunction(OneArgFunction):
    pass

class TrigFunction(OneArgFunction):
    pass

class gamma(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.gamma(X.thisptr))

class log(OneArgFunction):
    def __new__(cls, x, y=None):
        cdef Basic X = sympify(x)
        if y == None:
            return c2py(symengine.log(X.thisptr))
        cdef Basic Y = sympify(y)
        return c2py(symengine.log(X.thisptr, Y.thisptr))

class sin(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sin(X.thisptr))

class cos(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.cos(X.thisptr))

class tan(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.tan(X.thisptr))

class cot(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.cot(X.thisptr))

class sec(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sec(X.thisptr))

class csc(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.csc(X.thisptr))

class asin(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.asin(X.thisptr))

class acos(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acos(X.thisptr))

class atan(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.atan(X.thisptr))

class acot(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acot(X.thisptr))

class asec(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.asec(X.thisptr))

class acsc(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acsc(X.thisptr))

class sinh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sinh(X.thisptr))

class cosh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.cosh(X.thisptr))

class tanh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.tanh(X.thisptr))

class coth(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.coth(X.thisptr))

class sech(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sech(X.thisptr))

class csch(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.csch(X.thisptr))

class asinh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.asinh(X.thisptr))

class acosh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acosh(X.thisptr))

class atanh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.atanh(X.thisptr))

class acoth(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acoth(X.thisptr))

class asech(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.asech(X.thisptr))

class acsch(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acsch(X.thisptr))

class atan2(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.atan2(X.thisptr, Y.thisptr))

# For backwards compatibility

Sin = sin
Cos = cos
Tan = tan
Cot = cot
Sec = sec
Csc = csc
ASin = asin
ACos = acos
ATan = atan
ACot = acot
ASec = asec
ACsc = acsc
Sinh = sinh
Cosh = cosh
Tanh = tanh
Coth = coth
Sech = sech
Csch = csch
ASinh = asinh
ACosh = acosh
ATanh = atanh
ACoth = acoth
ASech = asech
ACsch = acsch
ATan2 = atan2
Log = log
Gamma = gamma

add = Add
mul = Mul


class Abs(OneArgFunction):

    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.abs(X.thisptr))

    def _sympy_(Basic self):
        cdef RCP[const symengine.Abs] X = symengine.rcp_static_cast_Abs(self.thisptr)
        arg = c2py(deref(X).get_arg())._sympy_()
        return abs(arg)

    def _sage_(Basic self):
        cdef RCP[const symengine.Abs] X = symengine.rcp_static_cast_Abs(self.thisptr)
        arg = c2py(deref(X).get_arg())._sage_()
        return abs(arg)

    @property
    def func(self):
        return self.__class__

class FunctionSymbol(Function):

    def get_name(Basic self):
        cdef RCP[const symengine.FunctionSymbol] X = \
            symengine.rcp_static_cast_FunctionSymbol(self.thisptr)
        name = deref(X).get_name().decode("utf-8")
        # In Python 2.7, function names cannot be unicode:
        return str(name)

    def _sympy_(self):
        import sympy
        name = self.get_name()
        return sympy.Function(name)(*self.args_as_sympy())

    def _sage_(self):
        import sage.all as sage
        name = self.get_name()
        return sage.function(name, *self.args_as_sage())

    def func(self, *values):
        name = self.get_name()
        return function_symbol(name, *values)


class UndefFunction(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, *values):
        return function_symbol(self.name, *values)


cdef RCP[const symengine.Basic] pynumber_to_symengine(PyObject* o1):
    cdef Basic X = sympify(<object>o1)
    return X.thisptr

cdef PyObject* symengine_to_sage(RCP[const symengine.Basic] o1):
    import sage.all as sage
    t = sage.SR(c2py(o1)._sage_())
    Py_XINCREF(<PyObject*>t)
    return <PyObject*>(t)

cdef PyObject* symengine_to_sympy(RCP[const symengine.Basic] o1):
    t = c2py(o1)._sympy_()
    Py_XINCREF(<PyObject*>t)
    return <PyObject*>(t)

cdef RCP[const symengine.Number] sympy_eval(PyObject* o1, long bits):
    prec = max(1, int(round(bits/3.3219280948873626)-1))
    cdef Number X = sympify((<object>o1).n(prec))
    return symengine.rcp_static_cast_Number(X.thisptr)

cdef RCP[const symengine.Number] sage_eval(PyObject* o1, long bits):
    cdef Number X = sympify((<object>o1).n(bits))
    return symengine.rcp_static_cast_Number(X.thisptr)

cdef RCP[const symengine.Basic] sage_diff(PyObject* o1, RCP[const symengine.Basic] symbol):
    cdef Basic X = sympify((<object>o1).diff(c2py(symbol)._sage_()))
    return X.thisptr

cdef RCP[const symengine.Basic] sympy_diff(PyObject* o1, RCP[const symengine.Basic] symbol):
    cdef Basic X = sympify((<object>o1).diff(c2py(symbol)._sympy_()))
    return X.thisptr

def create_sympy_module():
    cdef PyModule s = PyModule.__new__(PyModule)
    s.thisptr = symengine.make_rcp_PyModule(&symengine_to_sympy, &pynumber_to_symengine, &sympy_eval,
                                    &sympy_diff)
    return s

def create_sage_module():
    cdef PyModule s = PyModule.__new__(PyModule)
    s.thisptr = symengine.make_rcp_PyModule(&symengine_to_sage, &pynumber_to_symengine, &sage_eval,
                                    &sage_diff)
    return s

sympy_module = create_sympy_module()
sage_module = create_sage_module()

cdef class PyNumber(Number):
    def __cinit__(self, obj = None, PyModule module = None):
        if obj is None:
            return
        Py_XINCREF(<PyObject*>(obj))
        self.thisptr = symengine.make_rcp_PyNumber(<PyObject*>(obj), module.thisptr)

    def _sympy_(self):
        import sympy
        return sympy.sympify(self.pyobject())

    def _sage_(self):
        import sage.all as sage
        return sage.SR(self.pyobject())

    def pyobject(self):
        return <object>deref(symengine.rcp_static_cast_PyNumber(self.thisptr)).get_py_object()


class PyFunction(FunctionSymbol):

    def __init__(Basic self, pyfunction = None, args = None, pyfunction_class=None, module=None):
        if pyfunction is None:
            return
        cdef symengine.vec_basic v
        cdef Basic arg_
        for arg in args:
            arg_ = sympify(arg)
            v.push_back(arg_.thisptr)
        cdef PyFunctionClass _pyfunction_class = get_function_class(pyfunction_class, module)
        cdef PyObject* _pyfunction = <PyObject*>pyfunction
        Py_XINCREF(_pyfunction)
        self.thisptr = symengine.make_rcp_PyFunction(v, _pyfunction_class.thisptr, _pyfunction)

    def _sympy_(self):
        import sympy
        return sympy.sympify(self.pyobject())

    def _sage_(self):
        import sage.all as sage
        return sage.SR(self.pyobject())

    def pyobject(Basic self):
        return <object>deref(symengine.rcp_static_cast_PyFunction(self.thisptr)).get_py_object()

cdef class PyFunctionClass(object):

    def __cinit__(self, function, PyModule module not None):
        self.thisptr = symengine.make_rcp_PyFunctionClass(<PyObject*>(function), str(function).encode("utf-8"),
                                module.thisptr)

# TODO: remove this once SymEngine conversions are available in Sage.
def wrap_sage_function(func):
    return PyFunction(func, func.operands(), func.operator(), sage_module)


class Max(Function):

    def __new__(cls, *args):
        return _max(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Max(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.max(*s)

    @property
    def func(self):
        return self.__class__

class Min(Function):

    def __new__(cls, *args):
        return _min(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Min(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.min(*s)

    @property
    def func(self):
        return self.__class__


class Derivative(Basic):

    def __new__(self, expr, *variables):
        if len(variables) == 1 and is_sequence(variables[0]):
            return diff(expr, *variables[0])
        return diff(expr, *variables)

    @property
    def is_Derivative(self):
        return True

    @property
    def expr(Basic self):
        cdef RCP[const symengine.Derivative] X = symengine.rcp_static_cast_Derivative(self.thisptr)
        return c2py(deref(X).get_arg())

    @property
    def variables(self):
        return self.args[1:]

    def _sympy_(Basic self):
        cdef RCP[const symengine.Derivative] X = \
            symengine.rcp_static_cast_Derivative(self.thisptr)
        arg = c2py(deref(X).get_arg())._sympy_()
        cdef symengine.multiset_basic Y = deref(X).get_symbols()
        s = []
        for i in Y:
            s.append(c2py(<RCP[const symengine.Basic]>(i))._sympy_())
        import sympy
        return sympy.Derivative(arg, *s)

    def _sage_(Basic self):
        cdef RCP[const symengine.Derivative] X = \
            symengine.rcp_static_cast_Derivative(self.thisptr)
        arg = c2py(deref(X).get_arg())._sage_()
        cdef symengine.multiset_basic Y = deref(X).get_symbols()
        s = []
        for i in Y:
            s.append(c2py(<RCP[const symengine.Basic]>(i))._sage_())
        return arg.diff(*s)

    @property
    def func(self):
        return self.__class__


class Subs(Basic):

    def __new__(self, expr, variables, point):
        return sympify(expr).subs(variables, point)

    @property
    def expr(Basic self):
        cdef RCP[const symengine.Subs] me = symengine.rcp_static_cast_Subs(self.thisptr)
        return c2py(deref(me).get_arg())

    @property
    def variables(Basic self):
        cdef RCP[const symengine.Subs] me = symengine.rcp_static_cast_Subs(self.thisptr)
        cdef symengine.vec_basic variables = deref(me).get_variables()
        return vec_basic_to_tuple(variables)

    @property
    def point(Basic self):
        cdef RCP[const symengine.Subs] me = symengine.rcp_static_cast_Subs(self.thisptr)
        cdef symengine.vec_basic point = deref(me).get_point()
        return vec_basic_to_tuple(point)

    def _sympy_(Basic self):
        cdef RCP[const symengine.Subs] X = symengine.rcp_static_cast_Subs(self.thisptr)
        arg = c2py(deref(X).get_arg())._sympy_()
        cdef symengine.vec_basic V = deref(X).get_variables()
        cdef symengine.vec_basic P = deref(X).get_point()
        v = []
        p = []
        for i in range(V.size()):
            v.append(c2py(<RCP[const symengine.Basic]>(V[i]))._sympy_())
            p.append(c2py(<RCP[const symengine.Basic]>(P[i]))._sympy_())
        import sympy
        return sympy.Subs(arg, v, p)

    def _sage_(Basic self):
        cdef RCP[const symengine.Subs] X = symengine.rcp_static_cast_Subs(self.thisptr)
        arg = c2py(deref(X).get_arg())._sage_()
        cdef symengine.vec_basic V = deref(X).get_variables()
        cdef symengine.vec_basic P = deref(X).get_point()
        v = {}
        for i in range(V.size()):
            v[c2py(<RCP[const symengine.Basic]>(V[i]))._sage_()] = \
                c2py(<RCP[const symengine.Basic]>(P[i]))._sage_()
        return arg.subs(v)

    @property
    def func(self):
        return self.__class__


cdef class MatrixBase:

    @property
    def is_Matrix(self):
        return True

    def __richcmp__(a, b, int op):
        A = _sympify(a, False)
        B = _sympify(b, False)
        if not (isinstance(A, MatrixBase) and isinstance(B, MatrixBase)):
            if (op == 2):
                return False
            elif (op == 3):
                return True
            return NotImplemented
        return A._richcmp_(B, op)

    def _richcmp_(MatrixBase A, MatrixBase B, int op):
        if (op == 2):
            return deref(A.thisptr).eq(deref(B.thisptr))
        elif (op == 3):
            return not deref(A.thisptr).eq(deref(B.thisptr))
        else:
            return NotImplemented

    def __dealloc__(self):
        del self.thisptr

    def _symbolic_(self, ring):
        return ring(self._sage_())

    # TODO: fix this
    def __hash__(self):
        return 0


class MatrixError(Exception):
    pass


class ShapeError(ValueError, MatrixError):
    """Wrong matrix shape"""
    pass


class NonSquareMatrixError(ShapeError):
    pass

cdef class DenseMatrixBase(MatrixBase):
    """
    Represents a two-dimensional dense matrix.

    Examples
    ========

    Empty matrix:

    >>> DenseMatrix(3, 2)

    2D Matrix:

    >>> DenseMatrix(3, 2, [1, 2, 3, 4, 5, 6])
    [1, 2]
    [3, 4]
    [5, 6]

    >>> DenseMatrix([[1, 2], [3, 4], [5, 6]])
    [1, 2]
    [3, 4]
    [5, 6]

    """

    def __cinit__(self, row=None, col=None, v=None):
        if row is None:
            self.thisptr = new symengine.DenseMatrix(0, 0)
            return
        if v is None and col is not None:
            self.thisptr = new symengine.DenseMatrix(row, col)
            return
        if col is None:
            v = row
            row = 0
        cdef symengine.vec_basic v_
        cdef DenseMatrixBase A
        cdef Basic e_
        #TODO: Add a constructor to DenseMatrix in C++
        if (isinstance(v, DenseMatrixBase)):
            matrix_to_vec(v, v_)
            if col is None:
                row = v.nrows()
                col = v.ncols()
            self.thisptr = new symengine.DenseMatrix(row, col, v_)
            return
        for e in v:
            f = sympify(e)
            if isinstance(f, DenseMatrixBase):
                matrix_to_vec(f, v_)
                if col is None:
                    row = row + f.nrows()
                continue
            try:
                for e_ in f:
                    v_.push_back(e_.thisptr)
                if col is None:
                    row = row + 1
            except TypeError:
                e_ = f
                v_.push_back(e_.thisptr)
                if col is None:
                    row = row + 1
        if (row == 0):
            if (v_.size() != 0):
                self.thisptr = new symengine.DenseMatrix(0, 0, v_)
                raise ValueError("sizes don't match.")
            else:
                self.thisptr = new symengine.DenseMatrix(0, 0, v_)
        else:
            self.thisptr = new symengine.DenseMatrix(row, v_.size() / row, v_)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return deref(self.thisptr).__str__().decode("utf-8")

    def __add__(a, b):
        a = _sympify(a, False)
        b = _sympify(b, False)
        if not isinstance(a, MatrixBase) or not isinstance(b, MatrixBase):
            return NotImplemented
        cdef MatrixBase a_ = a
        cdef MatrixBase b_ = b
        if (a_.shape == (0, 0)):
            return b_
        if (b_.shape == (0, 0)):
            return a_
        if (a_.shape != b_.shape):
            raise ShapeError("Invalid shapes for matrix addition. Got %s %s" % (a_.shape, b_.shape))
        return a_.add_matrix(b_)

    def __mul__(a, b):
        a = _sympify(a, False)
        b = _sympify(b, False)
        if isinstance(a, MatrixBase):
            if isinstance(b, MatrixBase):
                if (a.ncols() != b.nrows()):
                    raise ShapeError("Invalid shapes for matrix multiplication. Got %s %s" % (a.shape, b.shape))
                return a.mul_matrix(b)
            elif isinstance(b, Basic):
                return a.mul_scalar(b)
            else:
                return NotImplemented
        elif isinstance(a, Basic):
            return b.mul_scalar(a)
        else:
            return NotImplemented

    def __sub__(a, b):
        a = _sympify(a, False)
        b = _sympify(b, False)
        if not isinstance(a, MatrixBase) or not isinstance(b, MatrixBase):
            return NotImplemented
        cdef MatrixBase a_ = a
        cdef MatrixBase b_ = b
        if (a_.shape != b_.shape):
            raise ShapeError("Invalid shapes for matrix subtraction. Got %s %s" % (a.shape, b.shape))
        return a_.add_matrix(-b_)

    def __neg__(self):
        return self.mul_scalar(-1)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if (self.ncols() == 0 or self.nrows() == 0):
                return []
            return [self.get(i // self.ncols(), i % self.ncols()) for i in range(*item.indices(len(self)))]
        elif isinstance(item, int):
            return self.get(item // self.ncols(), item % self.ncols())
        elif isinstance(item, tuple):
            if isinstance(item[0], int) and isinstance(item[1], int):
                return self.get(item[0], item[1])
            else:
                s = [0, 0, 0, 0, 0, 0]
                for i in (0, 1):
                    if isinstance(item[i], slice):
                        s[i], s[i+2], s[i+4] = item[i].indices(self.nrows() if i == 0 else self.ncols())
                    else:
                        s[i], s[i+2], s[i+4] = item[i], item[i] + 1, 1
                if (s[0] < 0 or s[0] > self.rows or s[0] >= s[2] or s[2] < 0 or s[2] > self.rows):
                    raise IndexError
                if (s[1] < 0 or s[1] > self.cols or s[1] >= s[3] or s[3] < 0 or s[3] > self.cols):
                    raise IndexError
                return self._submatrix(*s)
        else:
            raise NotImplementedError

    def __setitem__(self, key, value):
        cdef unsigned k, l
        if isinstance(key, int):
            self.set(key // self.ncols(), key % self.ncols(), value)
        elif isinstance(key, slice):
            k = 0
            for i in range(*key.indices(len(self))):
                self.set(i // self.ncols(), i % self.ncols(), value[k])
                k = k + 1
        elif isinstance(key, tuple):
            if isinstance(key[0], int):
                if isinstance(key[1], int):
                    self.set(key[0], key[1], value)
                else:
                    k = 0
                    for i in range(*key[1].indices(self.cols)):
                        self.set(key[0], i, value[k])
                        k = k + 1
            else:
                if isinstance(key[1], int):
                    k = 0
                    for i in range(*key[0].indices(self.rows)):
                        self.set(i, key[1], value[k])
                        k = k + 1
                else:
                    k = 0
                    for i in range(*key[0].indices(self.rows)):
                        l = 0
                        for j in range(*key[1].indices(self.cols)):
                            try:
                                self.set(i, j, value[k, l])
                            except TypeError:
                                self.set(i, j, value[k][l])
                            l = l + 1
                        k = k + 1
        else:
            raise NotImplementedError

    def row_join(self, rhs):
        cdef DenseMatrixBase o = sympify(rhs)
        if self.rows != o.rows:
            raise ShapeError("`self` and `rhs` must have the same number of rows.")
        cdef DenseMatrixBase result = self.__class__(self.rows, self.cols + o.cols)
        cdef Basic e_
        for i in range(self.rows):
            for j in range(self.cols):
                e_ = self._get(i, j)
                deref(result.thisptr).set(i, j, e_.thisptr)
        for i in range(o.rows):
            for j in range(o.cols):
                e_ = sympify(o._get(i, j))
                deref(result.thisptr).set(i, j + self.cols, e_.thisptr)
        return result

    def col_join(self, bott):
        cdef DenseMatrixBase o = sympify(bott)
        if self.cols != o.cols:
            raise ShapeError("`self` and `rhs` must have the same number of columns.")
        cdef DenseMatrixBase result = self.__class__(self.rows + o.rows, self.cols)
        cdef Basic e_
        for i in range(self.rows):
            for j in range(self.cols):
                e_ = self._get(i, j)
                deref(result.thisptr).set(i, j, e_.thisptr)
        for i in range(o.rows):
            for j in range(o.cols):
                e_ = sympify(o._get(i, j))
                deref(result.thisptr).set(i + self.rows, j, e_.thisptr)
        return result

    @property
    def rows(self):
        return self.nrows()

    @property
    def cols(self):
        return self.ncols()

    def nrows(self):
        return deref(self.thisptr).nrows()

    def ncols(self):
        return deref(self.thisptr).ncols()

    def __len__(self):
        return self.nrows() * self.ncols()

    property shape:
        def __get__(self):
            return (self.nrows(), self.ncols())

    property size:
        def __get__(self):
            return self.nrows()*self.ncols()

    def ravel(self):
        return [self._get(i, j) for i in range(self.nrows()) for j in range(self.ncols())]

    def reshape(self, rows, cols):
        if len(self) != rows*cols:
            raise ValueError("Invalid reshape parameters %d %d" % (rows, cols))
        cdef DenseMatrixBase r = self.__class__(self)
        deref(symengine.static_cast_DenseMatrix(r.thisptr)).resize(rows, cols)
        return r

    def _get_index(self, i, j):
        nr = self.nrows()
        nc = self.ncols()
        if i < 0:
            i += nr
        if j < 0:
            j += nc
        if i < 0 or i >= nr:
            raise IndexError("Row index out of bounds: %d" % i)
        if j < 0 or j >= nc:
            raise IndexError("Column index out of bounds: %d" % j)
        return i, j

    def get(self, i, j):
        i, j = self._get_index(i, j)
        return self._get(i, j)

    def _get(self, i, j):
        # No error checking is done
        return c2py(deref(self.thisptr).get(i, j))

    def set(self, i, j, e):
        i, j = self._get_index(i, j)
        return self._set(i, j, e)

    def _set(self, i, j, e):
        # No error checking is done
        cdef Basic e_ = sympify(e)
        if e_ is not None:
            deref(self.thisptr).set(i, j, e_.thisptr)

    def det(self):
        if self.nrows() != self.ncols():
            raise NonSquareMatrixError()
        return c2py(deref(self.thisptr).det())

    def inv(self, method='LU'):
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())

        if method.upper() == 'LU':
            ## inv() method of DenseMatrixBase uses LU factorization
            deref(self.thisptr).inv(deref(result.thisptr))
        elif method.upper() == 'FFLU':
            symengine.inverse_FFLU(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(result.thisptr)))
        elif method.upper() == 'GJ':
            symengine.inverse_GJ(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(result.thisptr)))
        else:
            raise Exception("Unsupported method.")
        return result

    def add_matrix(self, A):
        cdef MatrixBase A_ = sympify(A)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).add_matrix(deref(A_.thisptr), deref(result.thisptr))
        return result

    def mul_matrix(self, A):
        cdef MatrixBase A_ = sympify(A)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), A.ncols())
        deref(self.thisptr).mul_matrix(deref(A_.thisptr), deref(result.thisptr))
        return result

    def add_scalar(self, k):
        cdef Basic k_ = sympify(k)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).add_scalar(k_.thisptr, deref(result.thisptr))
        return result

    def mul_scalar(self, k):
        cdef Basic k_ = sympify(k)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).mul_scalar(k_.thisptr, deref(result.thisptr))
        return result

    def transpose(self):
        cdef DenseMatrixBase result = self.__class__(self.ncols(), self.nrows())
        deref(self.thisptr).transpose(deref(result.thisptr))
        return result

    @property
    def T(self):
        return self.transpose()

    def applyfunc(self, f):
        cdef DenseMatrixBase out = self.__class__(self)
        cdef int nr = self.nrows()
        cdef int nc = self.ncols()
        cdef Basic e_;
        for i in range(nr):
            for j in range(nc):
                e_ = sympify(f(self._get(i, j)))
                if e_ is not None:
                    deref(out.thisptr).set(i, j, e_.thisptr)
        return out

    def msubs(self, *args):
        cdef _DictBasic D = get_dict(*args)
        return self.applyfunc(lambda x: x.msubs(D))

    def diff(self, x):
        cdef Basic x_ = sympify(x)
        cdef DenseMatrixBase R = self.__class__(self.rows, self.cols)
        symengine.diff(<const symengine.DenseMatrix &>deref(self.thisptr),
                x_.thisptr, <symengine.DenseMatrix &>deref(R.thisptr))
        return R

    #TODO: implement this in C++
    def subs(self, *args):
        cdef _DictBasic D = get_dict(*args)
        return self.applyfunc(lambda x: x.subs(D))


    @property
    def free_symbols(self):
        s = set()
        for i in range(self.nrows()):
            for j in range(self.ncols()):
                s.update(self._get(i, j).free_symbols)
        return s

    def _submatrix(self, unsigned r_i, unsigned c_i, unsigned r_j, unsigned c_j, unsigned r_s=1, unsigned c_s=1):
        r_j, c_j = r_j - 1, c_j - 1
        cdef DenseMatrixBase result = self.__class__(((r_j - r_i) // r_s) + 1, ((c_j - c_i) // c_s) + 1)
        deref(self.thisptr).submatrix(deref(result.thisptr), r_i, c_i, r_j, c_j, r_s, c_s)
        return result

    def LU(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase U = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).LU(deref(L.thisptr), deref(U.thisptr))
        return L, U

    def LDL(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase D = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).LDL(deref(L.thisptr), deref(D.thisptr))
        return L, D

    def solve(self, b, method='LU'):
        cdef DenseMatrixBase b_ = sympify(b)
        cdef DenseMatrixBase x = self.__class__(b_.nrows(), b_.ncols())

        if method.upper() == 'LU':
            ## solve() method of DenseMatrixBase uses LU factorization
            symengine.pivoted_LU_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
                deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        elif method.upper() == 'FFLU':
            symengine.FFLU_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
                deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        elif method.upper() == 'LDL':
            symengine.LDL_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
                deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        elif method.upper() == 'FFGJ':
            symengine.FFGJ_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
                deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        else:
            raise Exception("Unsupported method.")

        return x

    def LUsolve(self, b):
        cdef DenseMatrixBase b_ = sympify(b)
        cdef DenseMatrixBase x = self.__class__(b.nrows(), b.ncols())
        symengine.pivoted_LU_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
            deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
            deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        return x

    def FFLU(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase U = self.__class__(self.nrows(), self.ncols(), [0]*self.nrows()*self.ncols())
        deref(self.thisptr).FFLU(deref(L.thisptr))

        for i in range(self.nrows()):
            for j in range(i + 1, self.ncols()):
                U.set(i, j, L.get(i, j))
                L.set(i, j, 0)
            U.set(i, i, L.get(i, i))

        return L, U

    def FFLDU(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase D = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase U = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).FFLDU(deref(L.thisptr), deref(D.thisptr), deref(U.thisptr))
        return L, D, U

    def jacobian(self, x):
        cdef DenseMatrixBase x_ = sympify(x)
        cdef DenseMatrixBase R = self.__class__(self.nrows(), x.nrows())
        symengine.jacobian(<const symengine.DenseMatrix &>deref(self.thisptr),
                <const symengine.DenseMatrix &>deref(x_.thisptr),
                <symengine.DenseMatrix &>deref(R.thisptr))
        return R

    def _sympy_(self):
        s = []
        cdef symengine.DenseMatrix A = deref(symengine.static_cast_DenseMatrix(self.thisptr))
        for i in range(A.nrows()):
            l = []
            for j in range(A.ncols()):
                l.append(c2py(A.get(i, j))._sympy_())
            s.append(l)
        import sympy
        return sympy.Matrix(s)

    def _sage_(self):
        s = []
        cdef symengine.DenseMatrix A = deref(symengine.static_cast_DenseMatrix(self.thisptr))
        for i in range(A.nrows()):
            l = []
            for j in range(A.ncols()):
                l.append(c2py(A.get(i, j))._sage_())
            s.append(l)
        import sage.all as sage
        return sage.Matrix(s)

    def dump_real(self, double[::1] out):
        cdef size_t ri, ci, nr, nc
        if out.size < self.size:
            raise ValueError("out parameter too short")
        nr = self.nrows()
        nc = self.ncols()
        for ri in range(nr):
            for ci in range(nc):
                out[ri*nc + ci] = symengine.eval_double(deref(
                    <symengine.RCP[const symengine.Basic]>(deref(self.thisptr).get(ri, ci))))

    def dump_complex(self, double complex[::1] out):
        cdef size_t ri, ci, nr, nc
        if out.size < self.size:
            raise ValueError("out parameter too short")
        nr = self.nrows()
        nc = self.ncols()
        for ri in range(nr):
            for ci in range(nc):
                out[ri*nc + ci] = symengine.eval_complex_double(deref(
                    <symengine.RCP[const symengine.Basic]>(deref(self.thisptr).get(ri, ci))))

    def __iter__(self):
        return DenseMatrixBaseIter(self)

    def as_mutable(self):
        return MutableDenseMatrix(self)

    def as_immutable(self):
        return ImmutableDenseMatrix(self)

    def tolist(self):
        return self[:]

    def atoms(self, *types):
        if types:
            s = set()
            if (isinstance(self, types)):
                s.add(self)
            for arg in self.tolist():
                s.update(arg.atoms(*types))
            return s
        else:
           return self.free_symbols

    def simplify(self, *args, **kwargs):
        return self._applyfunc(lambda x : x.simplify(*args, **kwargs))

    def expand(self, *args, **kwargs):
        return self.applyfunc(lambda x : x.expand())

class DenseMatrixBaseIter(object):

    def __init__(self, d):
        self.curr = -1
        self.d = d

    def __iter__(self):
        return self

    def __next__(self):
        self.curr = self.curr + 1
        if (self.curr < self.d.rows * self.d.cols):
            return self.d._get(self.curr // self.d.cols, self.curr % self.d.cols)
        else:
            raise StopIteration

    next = __next__

cdef class MutableDenseMatrix(DenseMatrixBase):

    def col_swap(self, i, j):
        for k in range(0, self.rows):
            self[k, i], self[k, j] = self[k, j], self[k, i]

    def fill(self, value):
        for i in range(self.rows):
            for j in range(self.cols):
                self[i, j] = value

    def row_swap(self, i, j):
        for k in range(0, self.cols):
            self[i, k], self[j, k] = self[j, k], self[i, k]

    def _applyfunc(self, f):
        cdef int nr = self.nrows()
        cdef int nc = self.ncols()
        for i in range(nr):
            for j in range(nc):
                self._set(i, j, f(self._get(i, j)))

Matrix = DenseMatrix = MutableDenseMatrix

cdef class ImmutableDenseMatrix(DenseMatrixBase):

    def __setitem__(self, key, value):
        raise TypeError("Cannot set values of {}".format(self.__class__))

    def set(self, i, j, e):
        raise TypeError("Cannot set values of {}".format(self.__class__))

    def _set(self, i, j, e):
        raise TypeError("Cannot set values of {}".format(self.__class__))

ImmutableMatrix = ImmutableDenseMatrix

cdef matrix_to_vec(DenseMatrixBase d, symengine.vec_basic& v):
    cdef Basic e_
    for i in range(d.nrows()):
        for j in range(d.ncols()):
            e_ = d._get(i, j)
            v.push_back(e_.thisptr)

def eye(n):
    cdef DenseMatrixBase d = DenseMatrix(n, n)
    symengine.eye(deref(symengine.static_cast_DenseMatrix(d.thisptr)), 0)
    return d

def diag(*values):
    cdef DenseMatrixBase d = DenseMatrix(len(values), len(values))
    cdef symengine.vec_basic V
    cdef Basic B
    for b in values:
        B = sympify(b)
        V.push_back(B.thisptr)
    symengine.diag(deref(symengine.static_cast_DenseMatrix(d.thisptr)), V, 0)
    return d

def ones(r, c = None):
    if c is None:
        c = r
    cdef DenseMatrixBase d = DenseMatrix(r, c)
    symengine.ones(deref(symengine.static_cast_DenseMatrix(d.thisptr)))
    return d

def zeros(r, c = None):
    if c is None:
        c = r
    cdef DenseMatrixBase d = DenseMatrix(r, c)
    symengine.zeros(deref(symengine.static_cast_DenseMatrix(d.thisptr)))
    return d

cdef class Sieve:
    @staticmethod
    def generate_primes(n):
        cdef symengine.vector[unsigned] primes
        symengine.sieve_generate_primes(primes, n)
        s = []
        for i in range(primes.size()):
            s.append(primes[i])
        return s

cdef class Sieve_iterator:
    cdef symengine.sieve_iterator *thisptr
    cdef unsigned limit
    def __cinit__(self):
        self.thisptr = new symengine.sieve_iterator()
        self.limit = 0

    def __cinit__(self, n):
        self.thisptr = new symengine.sieve_iterator(n)
        self.limit = n

    def __iter__(self):
        return self

    def __next__(self):
        n = deref(self.thisptr).next_prime()
        if self.limit > 0 and n > self.limit:
            raise StopIteration
        else:
            return n


I = c2py(symengine.I)
E = c2py(symengine.E)
pi = c2py(symengine.pi)
oo = c2py(symengine.Inf)
zoo = c2py(symengine.ComplexInf)
nan = c2py(symengine.Nan)

def module_cleanup():
    global I, E, pi, oo, zoo, nan, sympy_module, sage_module
    del I, E, pi, oo, zoo, nan, sympy_module, sage_module

import atexit
atexit.register(module_cleanup)

def diff(ex, *x):
    ex = sympify(ex)
    for i in x:
        ex = ex.diff(i)
    return ex

def expand(x):
    return sympify(x).expand()

def function_symbol(name, *args):
    cdef symengine.vec_basic v
    cdef Basic e_
    for e in args:
        e_ = sympify(e)
        if e_ is not None:
            v.push_back(e_.thisptr)
    return c2py(symengine.function_symbol(name.encode("utf-8"), v))

def sqrt(x):
    cdef Basic X = sympify(x)
    return c2py(symengine.sqrt(X.thisptr))

def exp(x):
    cdef Basic X = sympify(x)
    return c2py(symengine.exp(X.thisptr))

def _max(*args):
    cdef symengine.vec_basic v
    cdef Basic e_
    for e in args:
        e_ = sympify(e)
        v.push_back(e_.thisptr)
    return c2py(symengine.max(v))

def _min(*args):
    cdef symengine.vec_basic v
    cdef Basic e_
    for e in args:
        e_ = sympify(e)
        v.push_back(e_.thisptr)
    return c2py(symengine.min(v))

def gamma(x):
    cdef Basic X = sympify(x)
    return c2py(symengine.gamma(X.thisptr))

def eval_double(x):
    cdef Basic X = sympify(x)
    return c2py(<RCP[const symengine.Basic]>(symengine.real_double(symengine.eval_double(deref(X.thisptr)))))

def eval_complex_double(x):
    cdef Basic X = sympify(x)
    return c2py(<RCP[const symengine.Basic]>(symengine.complex_double(symengine.eval_complex_double(deref(X.thisptr)))))

have_mpfr = False
have_mpc = False
have_piranha = False
have_flint = False
have_llvm = False

IF HAVE_SYMENGINE_MPFR:
    have_mpfr = True
    def eval_mpfr(x, long prec):
        cdef Basic X = sympify(x)
        cdef symengine.mpfr_class a = symengine.mpfr_class(prec)
        symengine.eval_mpfr(a.get_mpfr_t(), deref(X.thisptr), symengine.MPFR_RNDN)
        return c2py(<RCP[const symengine.Basic]>(symengine.real_mpfr(symengine.std_move_mpfr(a))))

IF HAVE_SYMENGINE_MPC:
    have_mpc = True
    def eval_mpc(x, long prec):
        cdef Basic X = sympify(x)
        cdef symengine.mpc_class a = symengine.mpc_class(prec)
        symengine.eval_mpc(a.get_mpc_t(), deref(X.thisptr), symengine.MPFR_RNDN)
        return c2py(<RCP[const symengine.Basic]>(symengine.complex_mpc(symengine.std_move_mpc(a))))

IF HAVE_SYMENGINE_PIRANHA:
    have_piranha = True

IF HAVE_SYMENGINE_FLINT:
    have_flint = True

IF HAVE_SYMENGINE_LLVM:
    have_llvm = True

def require(obj, t):
    if not isinstance(obj, t):
        raise TypeError("{} required. {} is of type {}".format(t, obj, type(obj)))

def eval(x, long prec):
    if prec <= 53:
        return eval_complex_double(x)
    else:
        IF HAVE_SYMENGINE_MPC:
            return eval_mpc(x, prec)
        ELSE:
            raise ValueError("Precision %s is only supported with MPC" % prec)

def eval_real(x, long prec):
    if prec <= 53:
        return eval_double(x)
    else:
        IF HAVE_SYMENGINE_MPFR:
            return eval_mpfr(x, prec)
        ELSE:
            raise ValueError("Precision %s is only supported with MPFR" % prec)

def probab_prime_p(n, reps = 25):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return symengine.probab_prime_p(deref(symengine.rcp_static_cast_Integer(_n.thisptr)), reps) >= 1

def nextprime(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return c2py(<RCP[const symengine.Basic]>(symengine.nextprime(deref(symengine.rcp_static_cast_Integer(_n.thisptr)))))

def gcd(a, b):
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return c2py(<RCP[const symengine.Basic]>(symengine.gcd(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))))

def lcm(a, b):
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return c2py(<RCP[const symengine.Basic]>(symengine.lcm(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))))

def gcd_ext(a, b):
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    cdef RCP[const symengine.Integer] g, s, t
    symengine.gcd_ext(symengine.outArg_Integer(g), symengine.outArg_Integer(s), symengine.outArg_Integer(t),
        deref(symengine.rcp_static_cast_Integer(_a.thisptr)), deref(symengine.rcp_static_cast_Integer(_b.thisptr)))
    return [c2py(<RCP[const symengine.Basic]>g), c2py(<RCP[const symengine.Basic]>s), c2py(<RCP[const symengine.Basic]>t)]

def mod(a, b):
    if b == 0:
        raise ZeroDivisionError
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return c2py(<RCP[const symengine.Basic]>(symengine.mod(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))))

def quotient(a, b):
    if b == 0:
        raise ZeroDivisionError
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return c2py(<RCP[const symengine.Basic]>(symengine.quotient(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))))

def quotient_mod(a, b):
    if b == 0:
        raise ZeroDivisionError
    cdef RCP[const symengine.Integer] q, r
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    symengine.quotient_mod(symengine.outArg_Integer(q), symengine.outArg_Integer(r),
        deref(symengine.rcp_static_cast_Integer(_a.thisptr)), deref(symengine.rcp_static_cast_Integer(_b.thisptr)))
    return (c2py(<RCP[const symengine.Basic]>q), c2py(<RCP[const symengine.Basic]>r))

def mod_inverse(a, b):
    cdef RCP[const symengine.Integer] inv
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    cdef int ret_val = symengine.mod_inverse(symengine.outArg_Integer(inv),
        deref(symengine.rcp_static_cast_Integer(_a.thisptr)), deref(symengine.rcp_static_cast_Integer(_b.thisptr)))
    if ret_val == 0:
        return None
    return c2py(<RCP[const symengine.Basic]>inv)

def crt(rem, mod):
    cdef symengine.vec_integer _rem, _mod
    cdef Basic _a
    cdef bool ret_val
    for i in range(len(rem)):
        _a = sympify(rem[i])
        require(_a, Integer)
        _rem.push_back(symengine.rcp_static_cast_Integer(_a.thisptr))
        _a = sympify(mod[i])
        require(_a, Integer)
        _mod.push_back(symengine.rcp_static_cast_Integer(_a.thisptr))

    cdef RCP[const symengine.Integer] c
    ret_val = symengine.crt(symengine.outArg_Integer(c), _rem, _mod)
    if not ret_val:
        return None
    return c2py(<RCP[const symengine.Basic]>c)

def fibonacci(n):
    if n < 0 :
        raise NotImplementedError
    return c2py(<RCP[const symengine.Basic]>(symengine.fibonacci(n)))

def fibonacci2(n):
    if n < 0 :
        raise NotImplementedError
    cdef RCP[const symengine.Integer] f1, f2
    symengine.fibonacci2(symengine.outArg_Integer(f1), symengine.outArg_Integer(f2), n)
    return [c2py(<RCP[const symengine.Basic]>f1), c2py(<RCP[const symengine.Basic]>f2)]

def lucas(n):
    if n < 0 :
        raise NotImplementedError
    return c2py(<RCP[const symengine.Basic]>(symengine.lucas(n)))

def lucas2(n):
    if n < 0 :
        raise NotImplementedError
    cdef RCP[const symengine.Integer] f1, f2
    symengine.lucas2(symengine.outArg_Integer(f1), symengine.outArg_Integer(f2), n)
    return [c2py(<RCP[const symengine.Basic]>f1), c2py(<RCP[const symengine.Basic]>f2)]

def binomial(n, k):
    if k < 0:
        raise ArithmeticError
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return c2py(<RCP[const symengine.Basic]>symengine.binomial(deref(symengine.rcp_static_cast_Integer(_n.thisptr)), k))

def factorial(n):
    if n < 0:
        raise ArithmeticError
    return c2py(<RCP[const symengine.Basic]>(symengine.factorial(n)))

def divides(a, b):
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return symengine.divides(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))

def factor(n, B1 = 1.0):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] f
    cdef int ret_val = symengine.factor(symengine.outArg_Integer(f),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)), B1)
    if (ret_val == 1):
        return c2py(<RCP[const symengine.Basic]>f)
    else:
        return None

def factor_lehman_method(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] f
    cdef int ret_val = symengine.factor_lehman_method(symengine.outArg_Integer(f),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    if (ret_val == 1):
        return c2py(<RCP[const symengine.Basic]>f)
    else:
        return None

def factor_pollard_pm1_method(n, B = 10, retries = 5):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] f
    cdef int ret_val = symengine.factor_pollard_pm1_method(symengine.outArg_Integer(f),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)), B, retries)
    if (ret_val == 1):
        return c2py(<RCP[const symengine.Basic]>f)
    else:
        return None

def factor_pollard_rho_method(n, retries = 5):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] f
    cdef int ret_val = symengine.factor_pollard_rho_method(symengine.outArg_Integer(f),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)), retries)
    if (ret_val == 1):
        return c2py(<RCP[const symengine.Basic]>f)
    else:
        return None

def prime_factors(n):
    cdef symengine.vec_integer factors
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    symengine.prime_factors(factors, deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    s = []
    for i in range(factors.size()):
        s.append(c2py(<RCP[const symengine.Basic]>(factors[i])))
    return s

def prime_factor_multiplicities(n):
    cdef symengine.vec_integer factors
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    symengine.prime_factors(factors, deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    cdef Basic r
    dict = {}
    for i in range(factors.size()):
        r = c2py(<RCP[const symengine.Basic]>(factors[i]))
        if (r not in dict):
            dict[r] = 1
        else:
            dict[r] += 1
    return dict

def bernoulli(n):
    if n < 0:
        raise ArithmeticError
    return c2py(<RCP[const symengine.Basic]>(symengine.bernoulli(n)))

def primitive_root(n):
    cdef RCP[const symengine.Integer] g
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef bool ret_val = symengine.primitive_root(symengine.outArg_Integer(g),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    if ret_val == 0:
        return None
    return c2py(<RCP[const symengine.Basic]>g)

def primitive_root_list(n):
    cdef symengine.vec_integer root_list
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    symengine.primitive_root_list(root_list,
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    s = []
    for i in range(root_list.size()):
        s.append(c2py(<RCP[const symengine.Basic]>(root_list[i])))
    return s

def totient(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] m = symengine.rcp_static_cast_Integer(_n.thisptr)
    return c2py(<RCP[const symengine.Basic]>symengine.totient(m))

def carmichael(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] m = symengine.rcp_static_cast_Integer(_n.thisptr)
    return c2py(<RCP[const symengine.Basic]>symengine.carmichael(m))

def multiplicative_order(a, n):
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    require(_n, Integer)
    require(_a, Integer)
    cdef RCP[const symengine.Integer] n1 = symengine.rcp_static_cast_Integer(_n.thisptr)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] o
    cdef bool c = symengine.multiplicative_order(symengine.outArg_Integer(o),
        a1, n1)
    if not c:
        return None
    return c2py(<RCP[const symengine.Basic]>o)

def legendre(a, n):
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    require(_n, Integer)
    require(_a, Integer)
    return symengine.legendre(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def jacobi(a, n):
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    require(_n, Integer)
    require(_a, Integer)
    return symengine.jacobi(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def kronecker(a, n):
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    require(_n, Integer)
    require(_a, Integer)
    return symengine.kronecker(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def nthroot_mod(a, n, m):
    cdef RCP[const symengine.Integer] root
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    cdef Basic _m = sympify(m)
    require(_n, Integer)
    require(_a, Integer)
    require(_m, Integer)
    cdef RCP[const symengine.Integer] n1 = symengine.rcp_static_cast_Integer(_n.thisptr)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] m1 = symengine.rcp_static_cast_Integer(_m.thisptr)
    cdef bool ret_val = symengine.nthroot_mod(symengine.outArg_Integer(root), a1, n1, m1)
    if not ret_val:
        return None
    return c2py(<RCP[const symengine.Basic]>root)

def nthroot_mod_list(a, n, m):
    cdef symengine.vec_integer root_list
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    cdef Basic _m = sympify(m)
    require(_n, Integer)
    require(_a, Integer)
    require(_m, Integer)
    cdef RCP[const symengine.Integer] n1 = symengine.rcp_static_cast_Integer(_n.thisptr)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] m1 = symengine.rcp_static_cast_Integer(_m.thisptr)
    symengine.nthroot_mod_list(root_list, a1, n1, m1)
    s = []
    for i in range(root_list.size()):
        s.append(c2py(<RCP[const symengine.Basic]>(root_list[i])))
    return s

def powermod(a, b, m):
    cdef Basic _a = sympify(a)
    cdef Basic _m = sympify(m)
    cdef Number _b = sympify(b)
    require(_a, Integer)
    require(_m, Integer)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] m1 = symengine.rcp_static_cast_Integer(_m.thisptr)
    cdef RCP[const symengine.Number] b1 = symengine.rcp_static_cast_Number(_b.thisptr)
    cdef RCP[const symengine.Integer] root

    cdef bool ret_val = symengine.powermod(symengine.outArg_Integer(root), a1, b1, m1)
    if ret_val == 0:
        return None
    return c2py(<RCP[const symengine.Basic]>root)

def powermod_list(a, b, m):
    cdef Basic _a = sympify(a)
    cdef Basic _m = sympify(m)
    cdef Number _b = sympify(b)
    require(_a, Integer)
    require(_m, Integer)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] m1 = symengine.rcp_static_cast_Integer(_m.thisptr)
    cdef RCP[const symengine.Number] b1 = symengine.rcp_static_cast_Number(_b.thisptr)
    cdef symengine.vec_integer v

    symengine.powermod_list(v, a1, b1, m1)
    s = []
    for i in range(v.size()):
        s.append(c2py(<RCP[const symengine.Basic]>(v[i])))
    return s

def has_symbol(obj, symbol=None):
    cdef Basic b = _sympify(obj)
    cdef Basic s = _sympify(symbol)
    require(s, Symbol)
    if (not symbol):
        return not b.free_symbols.empty()
    else:
        return symengine.has_symbol(deref(b.thisptr),
                deref(symengine.rcp_static_cast_Symbol(s.thisptr)))


cdef size_t _size(n):
    try:
        return n.size
    except AttributeError:
        return len(n)  # e.g. array.array


def _get_shape_nested(ndarr):
    # no checking of shape consistency is done
    if isinstance(ndarr, (list, tuple)):
        return (len(ndarr),) + _get_shape_nested(ndarr[0])
    else:
        return ()


def get_shape(ndarr):
    try:
        return ndarr.shape
    except AttributeError:
        return _get_shape_nested(ndarr)


def _nested_getitem(ndarr, indices):
    if len(indices) == 0:
        return ndarr
    else:
        return _nested_getitem(ndarr[indices[0]], indices[1:])


def all_indices_from_shape(shape):
    return itertools.product(*(range(dim) for dim in shape))


def _ravel_nested(ndarr):
    return [_nested_getitem(ndarr, indices) for indices in
            all_indices_from_shape(get_shape(ndarr))]


def ravel(ndarr):
    try:
        return ndarr.ravel()
    except AttributeError:
        try:
            return _ravel_nested(ndarr.tolist())
        except AttributeError:
            return _ravel_nested(ndarr)


cdef class _Lambdify(object):
    """
    Lambdify instances are callbacks that numerically evaluate their symbolic
    expressions from user provided input (real or complex) into (possibly user
    provided) output buffers (real or complex). Multidimensional data are
    processed in their most cache-friendly way (i.e. "ravelled").

    Parameters
    ----------
    args: iterable of Symbols
    \*exprs: array_like of expressions
        the shape of exprs is preserved
    real : bool
        Whether datatype is ``double`` (``double complex`` otherwise).

    Returns
    -------
    callback instance with signature f(inp, out=None)

    Examples
    --------
    >>> from symengine import var, Lambdify
    >>> var('x y z')
    >>> f = Lambdify([x, y, z], [x+y+z, x*y*z])
    >>> f([2, 3, 4])
    [ 9., 24.]
    >>> out = np.array(2)
    >>> f(x, out); out
    [ 9., 24.]

    """
    cdef size_t args_size, tot_out_size
    cdef list out_shapes
    cdef readonly bint real
    cdef readonly int n_exprs
    cdef vector[int] accum_out_sizes
    cdef object numpy_dtype

    def __cinit__(self, args, *exprs, bool real=True):
        cdef vector[int] out_sizes
        self.real = real
        self.numpy_dtype = np.float64 if self.real else np.complex128
        self.out_shapes = [get_shape(expr) for expr in exprs]
        self.n_exprs = len(exprs)
        self.args_size = _size(args)
        self.tot_out_size = 0
        for idx, shape in enumerate(self.out_shapes):
            out_sizes.push_back(reduce(mul, shape or (1,)))
            self.tot_out_size += out_sizes[idx]
        for i in range(self.n_exprs + 1):
            self.accum_out_sizes.push_back(0)
            for j in range(i):
                self.accum_out_sizes[i] += out_sizes[j]

    def __init__(self, args, *exprs, bool real=True):
        cdef:
            Basic e_
            size_t ri, ci, nr, nc
            symengine.MatrixBase *mtx
            RCP[const symengine.Basic] b_
            symengine.vec_basic args_, outs_

        if isinstance(args, DenseMatrixBase):
            nr = args.nrows()
            nc = args.ncols()
            mtx = (<DenseMatrixBase>args).thisptr
            for ri in range(nr):
                for ci in range(nc):
                   args_.push_back(deref(mtx).get(ri, ci))
        else:
            for arg in args:
                e_ = _sympify(arg)
                args_.push_back(e_.thisptr)


        for curr_expr in exprs:
            if isinstance(curr_expr, DenseMatrixBase):
                nr = curr_expr.nrows()
                nc = curr_expr.ncols()
                mtx = (<DenseMatrixBase>curr_expr).thisptr
                for ri in range(nr):
                    for ci in range(nc):
                        b_ = deref(mtx).get(ri, ci)
                        outs_.push_back(b_)
            else:
                for e in ravel(curr_expr):
                    e_ = _sympify(e)
                    outs_.push_back(e_.thisptr)
        self._init(args_, outs_)

    cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_):
        raise ValueError("Not supported")

    cpdef unsafe_real(self,
                      double[::1] inp, double[::1] out,
                      int inp_offset=0, int out_offset=0):
        raise ValueError("Not supported")

    cpdef unsafe_complex(self, double complex[::1] inp, double complex[::1] out,
                         int inp_offset=0, int out_offset=0):
        raise ValueError("Not supported")

    cpdef eval_real(self,
                    inp,
                    out):
        if inp.size != self.args_size:
            raise ValueError("Size of inp incompatible with number of args.")
        if out.size != self.tot_out_size:
            raise ValueError("Size of out incompatible with number of exprs.")
        self.unsafe_real(inp, out)

    cpdef eval_complex(self,
                       inp,
                       out):
        if inp.size != self.args_size:
            raise ValueError("Size of inp incompatible with number of args.")
        if out.size != self.tot_out_size:
            raise ValueError("Size of out incompatible with number of exprs.")
        self.unsafe_complex(inp, out)

    def __call__(self, inp, out=None):
        """
        Parameters
        ----------
        inp: array_like
            last dimension must be equal to number of arguments.
        out: array_like or None (default)
            Allows for for low-overhead use (output argument, must be contiguous).
            If ``None``: an output container will be allocated (NumPy ndarray).
            If ``len(exprs) > 0`` output is found in the corresponding
            order. Note that ``out`` is not reshaped.

        Returns
        -------
        If ``len(exprs) == 1``: ``numpy.ndarray``, otherwise a tuple of such.

        """
        cdef:
            bint reshape_outs
            size_t idx, new_tot_out_size, nbroadcast = 1
            long inp_size
            tuple inp_shape
            double[::1] real_out, real_inp
            double complex[::1] cmplx_out, cmplx_inp
        try:
            inp = np.ascontiguousarray(inp, dtype=self.numpy_dtype)
        except TypeError:
            inp = np.fromiter(inp, dtype=self.numpy_dtype)
        inp_shape = inp.shape

        if self.real:
            real_inp = inp.ravel()
        else:
            cmplx_inp = inp.ravel()

        if inp.size % self.args_size != 0:
            raise ValueError("Broadcasting failed")
        nbroadcast = inp.size // self.args_size
        if nbroadcast > 1 and self.args_size == 1 and inp.shape[-1] != 1:  # Implicit reshape
            inp_shape = inp.shape + (1,)
        else:
            inp_shape = inp.shape
        new_tot_out_size = nbroadcast * self.tot_out_size
        if out is None:
            new_out_shapes = [inp_shape[:-1] + out_shape for out_shape in self.out_shapes]
            reshape_outs = len(new_out_shapes[0]) > 1
            out = np.empty(new_tot_out_size, dtype=self.numpy_dtype)
        else:
            reshape_outs = False
            if out.size < new_tot_out_size:
                raise ValueError("Incompatible size of output argument")
            if not (out.flags['C_CONTIGUOUS'] or out.flags['F_CONTIGUOUS']):
                raise ValueError("Output argument needs to be C-contiguous")
            if not out.flags['WRITEABLE']:
                raise ValueError("Output argument needs to be writeable")
            out = out.ravel()

        if self.real:
            real_out = out
        else:
            cmplx_out = out

        if self.real:
            for idx in range(nbroadcast):
                self.unsafe_real(real_inp, real_out,
                                 idx*self.args_size, idx*self.tot_out_size)
        else:
            for idx in range(nbroadcast):
                self.unsafe_complex(cmplx_inp, cmplx_out,
                                    idx*self.args_size, idx*self.tot_out_size)

        if reshape_outs:
            out = out.reshape((nbroadcast, self.tot_out_size))
            result = [out[:, self.accum_out_sizes[idx]:self.accum_out_sizes[idx+1]].reshape(
                new_out_shapes[idx]) for idx in range(self.n_exprs)]
        else:
            result = [out]

        if self.n_exprs == 1:
            return result[0]
        else:
            return result


cdef class LambdaDouble(_Lambdify):

    cdef vector[symengine.LambdaRealDoubleVisitor] lambda_double
    cdef vector[symengine.LambdaComplexDoubleVisitor] lambda_double_complex

    cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_):
        if self.real:
            self.lambda_double.resize(1)
            self.lambda_double[0].init(args_, outs_)
        else:
            self.lambda_double_complex.resize(1)
            self.lambda_double_complex[0].init(args_, outs_)

    cpdef unsafe_real(self, double[::1] inp, double[::1] out, int inp_offset=0, int out_offset=0):
        self.lambda_double[0].call(&out[out_offset], &inp[inp_offset])

    cpdef unsafe_complex(self, double complex[::1] inp, double complex[::1] out, int inp_offset=0, int out_offset=0):
        self.lambda_double_complex[0].call(&out[out_offset], &inp[inp_offset])


IF HAVE_SYMENGINE_LLVM:
    cdef class LLVMDouble(_Lambdify):

        cdef vector[symengine.LLVMDoubleVisitor] lambda_double

        cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_):
            self.lambda_double.resize(1)
            self.lambda_double[0].init(args_, outs_)

        cpdef unsafe_real(self, double[::1] inp, double[::1] out, int inp_offset=0, int out_offset=0):
            self.lambda_double[0].call(&out[out_offset], &inp[inp_offset])


def Lambdify(args, *exprs, bool real=True, backend=None):
    if backend is None:
        backend = os.getenv('SYMENGINE_LAMBDIFY_BACKEND', "lambda")
    if backend == "llvm":
        IF HAVE_SYMENGINE_LLVM:
            return LLVMDouble(args, *exprs, real=real)
        ELSE:
            raise ValueError("""llvm backend is chosen, but symengine is not compiled
                                with llvm support.""")
    elif backend == "lambda":
        pass
    else:
        warnings.warn("Unknown SymEngine backend: %s\nUsing backend='lambda'" % backend)
    return LambdaDouble(args, *exprs, real=real)


def LambdifyCSE(args, *exprs, cse=None, concatenate=None, **kwargs):
    """
    Analogous with Lambdify but performs common subexpression elimination
    internally. See docstring of Lambdify.

    Parameters
    ----------
    args: iterable of symbols
    exprs: iterable of expressions (with symbols from args)
    cse: callback (default: None)
        defaults to sympy.cse (see SymPy documentation)
    concatenate: callback (default: numpy.concatenate)
        Examples when not using numpy:
        ``lambda tup: tup[0]+list(tup[1])``
        ``lambda tup: tup[0]+array.array('d', tup[1])``
    \*\*kwargs: Keyword arguments passed onto Lambdify

    """
    if cse is None:
        from sympy import cse
    if concatenate is None:
        from numpy import concatenate
    from sympy import sympify as s_sympify
    flat_exprs = list(itertools.chain(*map(ravel, exprs)))
    subs, flat_new_exprs = cse([s_sympify(expr) for expr in flat_exprs])

    explicit_subs = {}
    for k, v in subs:
        explicit_subs[k] = v.xreplace(explicit_subs)

    if subs:
        cse_symbs, cse_exprs = zip(*subs)
        new_exprs = []
        n_taken = 0
        for expr in exprs:
            shape = get_shape(expr) or (1,)
            size = long(reduce(mul, shape))
            if len(shape) == 1:
                new_exprs.append(flat_new_exprs[n_taken:n_taken+size])
            elif len(shape) == 2:
                new_exprs.append(DenseMatrix(
                    shape[0], shape[1], flat_new_exprs[n_taken:n_taken+size]))
            else:
                raise NotImplementedError("n-dimensional output not yet supported.")
            n_taken += size
        lmb = Lambdify(tuple(args) + cse_symbs, *new_exprs, **kwargs)
        cse_lambda = Lambdify(args, [expr.xreplace(explicit_subs) for expr in cse_exprs], **kwargs)
        def cb(inp, out=None, **kw):
            cse_vals = cse_lambda(inp, **kw)
            print(inp, cse_vals) # DO-NOT-MERGE!
            new_inp = concatenate((inp, cse_vals), axis=-1)
            return lmb(new_inp, out, **kw)

        return cb
    else:
        return Lambdify(args, *exprs, **kwargs)


def ccode(expr):
    cdef Basic expr_ = sympify(expr)
    return symengine.ccode(deref(expr_.thisptr)).decode("utf-8")

# Turn on nice stacktraces:
symengine.print_stack_on_segfault()
