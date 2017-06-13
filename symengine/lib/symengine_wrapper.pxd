cimport symengine
from symengine cimport RCP, map_basic_basic

cdef class Basic(object):
    cdef RCP[const symengine.Basic] thisptr

cdef class MatrixBase(object):
    cdef symengine.MatrixBase* thisptr

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

