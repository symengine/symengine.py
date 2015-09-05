#ifndef SYMENGINE_PYNUMBER_H
#define SYMENGINE_PYNUMBER_H

#include <Python.h>
#include <symengine/number.h>
#include <symengine/constants.h>
#include <symengine/functions.h>

namespace SymEngine {

class PyModule : public EnableRCPFromThis<PyModule> {
public:
    PyObject* (*to_py_)(const RCP<const Basic> x);
    RCP<const Basic> (*from_py_)(PyObject*);
    RCP<const Basic> (*eval_)(PyObject*, long bits);
    PyObject *one, *zero, *minus_one;
public:
    PyModule(PyObject* (*)(const RCP<const Basic> x), RCP<const Basic> (*)(PyObject*),
             RCP<const Basic> (*)(PyObject*, long bits));
    ~PyModule();
    PyObject* get_zero() const {
        return zero;
    }
    PyObject* get_one() const {
        return one;
    }
    PyObject* get_minus_one() const {
        return minus_one;
    }
};

class PyFunctionClass : public EnableRCPFromThis<PyFunctionClass> {
private:
    PyObject* pyobject_;
    std::string name_;
    mutable size_t hash_;
    RCP<const PyModule> pymodule_;
public:
    PyFunctionClass(PyObject* pyobject, std::string name, RCP<const PyModule> &pymodule) :
        pyobject_{pyobject}, name_{name}, pymodule_{pymodule} {

    }
    PyObject* get_py_object() const {
        return pyobject_;
    }
    RCP<const PyModule> get_py_module() const {
        return pymodule_;
    }
    std::string get_name() const {
        return name_;
    }
    PyObject* call(const vec_basic &vec) const {
        PyObject* tuple = PyTuple_New(vec.size());
        for (unsigned i = 0; i < vec.size(); i++) {
            PyTuple_SetItem(tuple, i, pymodule_->to_py_(vec[i]));
        }
        return PyObject_CallObject(pyobject_, tuple);
    }
    bool __eq__(const PyFunctionClass &x) const {
        return PyObject_RichCompareBool(pyobject_, x.pyobject_, Py_EQ) != 1;
    }
    int compare(const PyFunctionClass &x) const {
        if (__eq__(x)) return 0;
        return PyObject_RichCompareBool(pyobject_, x.pyobject_, Py_LT) == 1 ? 1 : -1;
    }
    std::size_t __hash__() const {
        return PyObject_Hash(pyobject_);
    }
    std::size_t hash() const {
        if (hash_ == 0)
            hash_ = __hash__();
        return hash_;
    }
};

class PyFunction : public FunctionSymbol {
private:
    RCP<const PyFunctionClass> pyfunction_class_;
    mutable PyObject* pyobject_;
public:
    PyFunction(const vec_basic &vec, const RCP<const PyFunctionClass> &pyfunc_class, const PyObject* pyobject = nullptr) :
            FunctionSymbol(pyfunc_class->get_name(), vec), pyfunction_class_{pyfunc_class}, pyobject_{pyobject_} {

    }
    PyObject* get_py_object() const {
        if (pyobject_ != nullptr) {
            return pyobject_;
        }
        pyobject_ = pyfunction_class_->call(arg_);
        return pyobject_;
    }
    virtual RCP<const FunctionSymbol> create(const vec_basic &x) const {
        return make_rcp<const PyFunction>(x, pyfunction_class_);
    }
};

class PyNumber : public NumberWrapper {
private:
    PyObject* pyobject_;
    RCP<const PyModule> pymodule_;

public:
    IMPLEMENT_TYPEID(NUMBER_WRAPPER)
    PyNumber(PyObject* pyobject, const RCP<const PyModule> &pymodule);
    ~PyNumber() {
        Py_DecRef(pyobject_);
    }
    PyObject* get_py_object() const {
        return pyobject_;
    }
    RCP<const PyModule> get_py_module() const {
        return pymodule_;
    }
    //! \return true if `0`
    virtual bool is_zero() const {
        return PyObject_RichCompareBool(pyobject_, pymodule_->get_zero(), Py_EQ) == 1;
    }
    //! \return true if `1`
    virtual bool is_one() const {
        return PyObject_RichCompareBool(pyobject_, pymodule_->get_one(), Py_EQ) == 1;
    }
    //! \return true if `-1`
    virtual bool is_minus_one() const {
        return PyObject_RichCompareBool(pyobject_, pymodule_->get_minus_one(), Py_EQ) == 1;
    }
    //! \return true if negative
    virtual bool is_negative() const {
        return PyObject_RichCompareBool(pyobject_, pymodule_->get_zero(), Py_LT) == 1;
    }
    //! \return true if positive
    virtual bool is_positive() const {
        return PyObject_RichCompareBool(pyobject_, pymodule_->get_zero(), Py_GT) == 1;
    }
    //! return true if the number is an exact representation
    //  false if the number is an approximation
    virtual bool is_exact() const { return true; };

    //! Addition
    virtual RCP<const Number> add(const Number &other) const {
        PyObject *other_p, *result;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
            result = PyNumber_Add(pyobject_, other_p);
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
            result = PyNumber_Add(pyobject_, other_p);
            Py_DecRef(other_p);
        }
        return make_rcp<PyNumber>(result, pymodule_);
    }
    //! Subtraction
    virtual RCP<const Number> sub(const Number &other) const {
        PyObject *other_p, *result;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
            result = PyNumber_Subtract(pyobject_, other_p);
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
            result = PyNumber_Subtract(pyobject_, other_p);
            Py_DecRef(other_p);
        }
        return make_rcp<PyNumber>(result, pymodule_);
    }
    virtual RCP<const Number> rsub(const Number &other) const {
        PyObject *other_p, *result;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
            result = PyNumber_Subtract(other_p, pyobject_);
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
            result = PyNumber_Subtract(other_p, pyobject_);
            Py_DecRef(other_p);
        }
        return make_rcp<PyNumber>(result, pymodule_);
    }
    //! Multiplication
    virtual RCP<const Number> mul(const Number &other) const {
        PyObject *other_p, *result;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
            result = PyNumber_Multiply(pyobject_, other_p);
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
            result = PyNumber_Multiply(pyobject_, other_p);
            Py_DecRef(other_p);
        }
        return make_rcp<PyNumber>(result, pymodule_);
    }
    //! Division
    virtual RCP<const Number> div(const Number &other) const {
        PyObject *other_p, *result;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
            result = PyNumber_Divide(pyobject_, other_p);
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
            result = PyNumber_Divide(pyobject_, other_p);
            Py_DecRef(other_p);
        }
        return make_rcp<PyNumber>(result, pymodule_);
    }
    virtual RCP<const Number> rdiv(const Number &other) const {
        PyObject *other_p, *result;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
            result = PyNumber_Divide(pyobject_, other_p);
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
            result = PyNumber_Divide(pyobject_, other_p);
            Py_DecRef(other_p);
        }
        return make_rcp<PyNumber>(result, pymodule_);
    }
    //! Power
    virtual RCP<const Number> pow(const Number &other) const {
        PyObject *other_p, *result;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
            result = PyNumber_Power(pyobject_, other_p, Py_None);
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
            result = PyNumber_Power(pyobject_, other_p, Py_None);
            Py_DecRef(other_p);
        }
        return make_rcp<PyNumber>(result, pymodule_);
    }
    virtual RCP<const Number> rpow(const Number &other) const {
        PyObject *other_p, *result;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
            result = PyNumber_Power(other_p, pyobject_, Py_None);
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
            result = PyNumber_Power(other_p, pyobject_, Py_None);
            Py_DecRef(other_p);
        }
        return make_rcp<PyNumber>(result, pymodule_);
    }
    //! Differentiation w.r.t Symbol `x`
    virtual RCP<const Basic> diff(const RCP<const Symbol> &x) const {
        return zero;
    }

    virtual RCP<const Basic> eval(long bits) const {
        return pymodule_->eval_(pyobject_, bits);
    }

    virtual std::string __str__() const {
        return std::string(PyString_AsString(PyObject_Str(pyobject_)));
    }

    virtual int compare(const Basic &o) const;

    virtual bool __eq__(const Basic &o) const;

    virtual std::size_t __hash__() const;
};

}

#endif //SYMENGINE_PYNUMBER_H
