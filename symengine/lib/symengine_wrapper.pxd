cimport symengine
from symengine cimport RCP, map_basic_basic

cdef class Basic(object):
    cdef RCP[const symengine.Basic] thisptr

cdef class Number(Basic):
    pass

cdef class Integer(Number):
    pass

cdef class Rational(Number):
    pass

cdef class Complex(Number):
    pass

cdef class RealDouble(Number):
    pass

cdef class ComplexDouble(Number):
    pass

cdef class RealMPFR(Number):
    pass

cdef class ComplexMPC(Number):
    pass

cdef class PyNumber(Number):
    pass

cdef class Add(Basic):
    pass

cdef class Mul(Basic):
    pass

cdef class Pow(Basic):
    pass

cdef class Function(Basic):
    pass

cdef class TrigFunction(Function):
    pass

cdef class Sin(TrigFunction):
    pass

cdef class Cos(TrigFunction):
    pass

cdef class Tan(TrigFunction):
    pass

cdef class Cot(TrigFunction):
    pass

cdef class Csc(TrigFunction):
    pass

cdef class Sec(TrigFunction):
    pass

cdef class ASin(TrigFunction):
    pass

cdef class ACos(TrigFunction):
    pass

cdef class ATan(TrigFunction):
    pass

cdef class ACot(TrigFunction):
    pass

cdef class ACsc(TrigFunction):
    pass

cdef class ASec(TrigFunction):
    pass

cdef class HyperbolicFunction(Function):
    pass

cdef class Sinh(HyperbolicFunction):
    pass

cdef class Cosh(HyperbolicFunction):
    pass

cdef class Tanh(HyperbolicFunction):
    pass

cdef class Coth(HyperbolicFunction):
    pass

cdef class ASinh(HyperbolicFunction):
    pass

cdef class ACosh(HyperbolicFunction):
    pass

cdef class ATanh(HyperbolicFunction):
    pass

cdef class ACoth(HyperbolicFunction):
    pass

cdef class FunctionSymbol(Function):
    pass

cdef class PyFunction(FunctionSymbol):
    pass

cdef class Abs(Function):
    pass

cdef class Gamma(Function):
    pass

cdef class Derivative(Basic):
    pass

cdef class Subs(Basic):
    pass

cdef class MatrixBase(object):
    cdef symengine.MatrixBase* thisptr

cdef class DenseMatrix(MatrixBase):
    pass

cdef class Log(Function):
    pass

cdef class Infinity(Number):
    pass

cdef class Piecewise(Basic):
    pass

cdef class Boolean(Basic):
    pass

cdef class BooleanAtom(Boolean):
    pass

cdef class Contains(Boolean):
    pass

cdef class Set(Basic):
    pass

cdef class Interval(Set):
    pass


cdef class PyFunctionClass(object):
    cdef RCP[const symengine.PyFunctionClass] thisptr

cdef class PyModule(object):
    cdef RCP[const symengine.PyModule] thisptr

cdef class _DictBasic(object):
    cdef map_basic_basic c

cdef class DictBasicIter(object):
    cdef map_basic_basic.iterator begin
    cdef map_basic_basic.iterator end
    cdef init(self, map_basic_basic.iterator begin, map_basic_basic.iterator end)

