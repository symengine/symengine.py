#include "pynumber.h"
#include <symengine/number.h>

namespace SymEngine {

PyNumber::PyNumber(PyObject* pyobject, PyModule* pymodule) : pyobject_(pyobject), pymodule_(pymodule) {

}

PyModule::PyModule(PyObject* pyobject, PyObject* (*to_py)(const RCP<const Basic>),
            RCP<const Basic> (*from_py)(PyObject*), RCP<const Basic> (*eval)(PyObject*, long)) :
            pyobject_(pyobject), to_py_(to_py), from_py_(from_py), eval_(eval) {

}


std::size_t PyNumber::__hash__() const {
    return PyObject_Hash(pyobject_);
}

bool PyNumber::__eq__(const Basic &o) const {
    return PyObject_RichCompareBool(pyobject_, pymodule_->to_py_(o.rcp_from_this_cast<const Basic>()), Py_EQ) == 1;
}

int PyNumber::compare(const Basic &o) const {
    PyObject* o1 = pymodule_->to_py_(o.rcp_from_this_cast<const Basic>());
    if (PyObject_RichCompareBool(pyobject_, o1, Py_EQ) == 1)
        return 0;
    return PyObject_RichCompareBool(pyobject_, o1, Py_LT) == 1 ? -1 : 1;
}

#define xstr(s) str(s)
#define str(s) #s
#define SYMENGINE_EVALUATE_PY_MACRO(function) \
    virtual RCP<const Basic> function(const Basic &x) const { \
        SYMENGINE_ASSERT(is_a<PyNumber>(x)); \
        static PyObject* name = PyString_FromString(xstr(function)); \
        PyModule* module = static_cast<const PyNumber &>(x).get_py_module(); \
        return module->from_py_(PyObject_CallMethodObjArgs(module->get_py_object(), name, static_cast<const PyNumber &>(x).get_py_object(), NULL));\
    } \

//! A class that will evaluate functions numerically.
class EvaluatePyNumber : public Evaluate {
public:
    SYMENGINE_EVALUATE_PY_MACRO(sin);
    SYMENGINE_EVALUATE_PY_MACRO(cos);
    SYMENGINE_EVALUATE_PY_MACRO(tan);
    SYMENGINE_EVALUATE_PY_MACRO(cot);
    SYMENGINE_EVALUATE_PY_MACRO(sec);
    SYMENGINE_EVALUATE_PY_MACRO(csc);
    SYMENGINE_EVALUATE_PY_MACRO(asin);
    SYMENGINE_EVALUATE_PY_MACRO(acos);
    SYMENGINE_EVALUATE_PY_MACRO(atan);
    SYMENGINE_EVALUATE_PY_MACRO(acot);
    SYMENGINE_EVALUATE_PY_MACRO(asec);
    SYMENGINE_EVALUATE_PY_MACRO(acsc);
    SYMENGINE_EVALUATE_PY_MACRO(sinh);
    SYMENGINE_EVALUATE_PY_MACRO(cosh);
    SYMENGINE_EVALUATE_PY_MACRO(tanh);
    SYMENGINE_EVALUATE_PY_MACRO(coth);
    SYMENGINE_EVALUATE_PY_MACRO(asinh);
    SYMENGINE_EVALUATE_PY_MACRO(acosh);
    SYMENGINE_EVALUATE_PY_MACRO(atanh);
    SYMENGINE_EVALUATE_PY_MACRO(acoth);
    SYMENGINE_EVALUATE_PY_MACRO(log);
    SYMENGINE_EVALUATE_PY_MACRO(gamma);
    SYMENGINE_EVALUATE_PY_MACRO(abs);
};

Evaluate& PyNumber::get_eval() const {
    static EvaluatePyNumber evaluatePyNumber;
    return evaluatePyNumber;
}

}