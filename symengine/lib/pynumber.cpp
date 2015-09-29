#include "pynumber.h"
#include "pyfunction.h"
#include <symengine/number.h>

namespace SymEngine {

PyNumber::PyNumber(PyObject* pyobject, const RCP<const PyModule> &pymodule) :
        pyobject_(pyobject), pymodule_(pymodule) {
}

PyModule::PyModule(PyObject* (*to_py)(const RCP<const Basic>), RCP<const Basic> (*from_py)(PyObject*),
        RCP<const Basic> (*eval)(PyObject*, long)) :
        to_py_(to_py), from_py_(from_py), eval_(eval) {
    zero = PyInt_FromLong(0);
    one = PyInt_FromLong(1);
    minus_one = PyInt_FromLong(-1);
}

PyModule::~PyModule(){
    Py_DecRef(zero);
    Py_DecRef(one);
    Py_DecRef(minus_one);
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

}