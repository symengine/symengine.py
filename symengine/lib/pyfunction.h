#ifndef SYMENGINE_PYFUNCTION_H
#define SYMENGINE_PYFUNCTION_H

namespace SymEngine {

class PyFunctionClass : public EnableRCPFromThis<PyFunctionClass> {
private:
    PyObject *pyobject_;
    std::string name_;
    mutable size_t hash_;
    RCP<const PyModule> pymodule_;
public:
    PyFunctionClass(PyObject *pyobject, std::string name, const RCP<const PyModule> &pymodule) :
            pyobject_{pyobject}, name_{name}, pymodule_{pymodule} {

    }

    PyObject *get_py_object() const { return pyobject_; }
    RCP<const PyModule> get_py_module() const { return pymodule_; }
    std::string get_name() const { return name_; }

    PyObject *call(const vec_basic &vec) const {
        PyObject *tuple = PyTuple_New(vec.size());
        PyObject *temp;
        for (unsigned i = 0; i < vec.size(); i++) {
            temp = pymodule_->to_py_(vec[i]);
            PyTuple_SetItem(tuple, i, temp);
            Py_DecRef(temp);
        }
        temp = PyObject_CallObject(pyobject_, tuple);
        Py_DecRef(tuple);
        return temp;
    }

    bool __eq__(const PyFunctionClass &x) const {
        return PyObject_RichCompareBool(pyobject_, x.pyobject_, Py_EQ) != 1;
    }

    int compare(const PyFunctionClass &x) const {
        if (__eq__(x)) return 0;
        return PyObject_RichCompareBool(pyobject_, x.pyobject_, Py_LT) == 1 ? 1 : -1;
    }

    std::size_t hash() const {
        if (hash_ == 0)
            hash_ = PyObject_Hash(pyobject_);
        return hash_;
    }
};

class PyFunction : public FunctionSymbol {
private:
    RCP<const PyFunctionClass> pyfunction_class_;
    PyObject *pyobject_;
public:
    PyFunction(const vec_basic &vec, const RCP<const PyFunctionClass> &pyfunc_class,
               PyObject *pyobject):
            FunctionSymbol(pyfunc_class->get_name(), std::move(vec)), pyfunction_class_{pyfunc_class}, pyobject_{pyobject} {

    }

    ~PyFunction() {
        Py_DecRef(pyobject_);
    }

    PyObject *get_py_object() const {
        return pyobject_;
    }

    virtual RCP<const Basic> create(const vec_basic &x) const {
        PyObject* pyobj = pyfunction_class_->call(x);
        RCP<const Basic> result = pyfunction_class_->get_py_module()->from_py_(pyobj);
        Py_DecRef(pyobj);
        return result;
    }
};

} // SymEngine
#endif //SYMENGINE_PYFUNCTION_H
