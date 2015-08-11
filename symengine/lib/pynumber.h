#ifndef SYMENGINE_PYNUMBER_H
#define SYMENGINE_PYNUMBER_H

#include <Python.h>
#include <symengine/number.h>
#include <symengine/constants.h>

namespace SymEngine {

class PyModule {
public:
    PyObject* pyobject_;
    PyObject* (*to_py_)(const RCP<const Basic> x);
    RCP<const Basic> (*from_py_)(PyObject*);
    RCP<const Basic> (*eval_)(PyObject*, long bits);
public:
    PyModule(PyObject*, PyObject* (*)(const RCP<const Basic> x),
             RCP<const Basic> (*)(PyObject*), RCP<const Basic> (*)(PyObject*, long bits));
    PyObject* get_py_object() const {

        return pyobject_;
    }
};

class PyNumber : public NumberWrapper {
private:
    PyObject* pyobject_;
    PyModule* pymodule_;

public:
    IMPLEMENT_TYPEID(NUMBER_WRAPPER)
    PyNumber(PyObject* pyobject, PyModule* pymodule);
    ~PyNumber() {
        Py_DecRef(pyobject_);
    }
    PyObject* get_py_object() const {
        return pyobject_;
    }
    PyModule* get_py_module() const {
        return pymodule_;
    }
    static PyObject* get_zero() {
        static PyObject* t = PyInt_FromLong(0);
        return t;
    }
    static PyObject* get_one() {
        static PyObject* t = PyInt_FromLong(1);
        return t;
    }
    static PyObject* get_minus_one() {
        static PyObject* t = PyInt_FromLong(-1);
        return t;
    }
    //! \return true if `0`
    virtual bool is_zero() const {
        return PyObject_RichCompareBool(pyobject_, get_zero(), Py_EQ) == 1;
    }
    //! \return true if `1`
    virtual bool is_one() const {
        return PyObject_RichCompareBool(pyobject_, get_one(), Py_EQ) == 1;
    }
    //! \return true if `-1`
    virtual bool is_minus_one() const {
        return PyObject_RichCompareBool(pyobject_, get_minus_one(), Py_EQ) == 1;
    }
    //! \return true if negative
    virtual bool is_negative() const {
        return PyObject_RichCompareBool(pyobject_, get_zero(), Py_LT) == 1;
    }
    //! \return true if positive
    virtual bool is_positive() const {
        return PyObject_RichCompareBool(pyobject_, get_zero(), Py_GT) == 1;
    }
    //! return true if the number is an exact representation
    //  false if the number is an approximation
    virtual bool is_exact() const { return false; };
    //! Get `Evaluate` singleton to evaluate numerically
    virtual Evaluate& get_eval() const;

    //! Addition
    virtual RCP<const Number> add(const Number &other) const {
        PyObject* other_p;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast< ..Number &>(other).pyobject_;
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
        }
        return make_rcp<PyNumber>(PyNumber_Add(pyobject_, other_p), pymodule_);
    }
    //! Subtraction
    virtual RCP<const Number> sub(const Number &other) const {
        PyObject* other_p;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
        }
        return make_rcp<PyNumber>(PyNumber_Subtract(pyobject_, other_p), pymodule_);
    }
    virtual RCP<const Number> rsub(const Number &other) const {
        PyObject* other_p;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
        }
        return make_rcp<PyNumber>(PyNumber_Subtract(other_p, pyobject_), pymodule_);
    }
    //! Multiplication
    virtual RCP<const Number> mul(const Number &other) const {
        PyObject* other_p;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
        }
        return make_rcp<PyNumber>(PyNumber_Multiply(pyobject_, other_p), pymodule_);
    }
    //! Division
    virtual RCP<const Number> div(const Number &other) const {
        PyObject* other_p;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
        }
        return make_rcp<PyNumber>(PyNumber_Divide(pyobject_, other_p), pymodule_);
    }
    virtual RCP<const Number> rdiv(const Number &other) const {
        PyObject* other_p;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
        }
        return make_rcp<PyNumber>(PyNumber_Divide(other_p, pyobject_), pymodule_);
    }
    //! Power
    virtual RCP<const Number> pow(const Number &other) const {
        PyObject* other_p;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
        }
        return make_rcp<PyNumber>(PyNumber_Power(pyobject_, other_p, Py_None), pymodule_);
    }
    virtual RCP<const Number> rpow(const Number &other) const {
        PyObject* other_p;
        if (is_a<PyNumber>(other)) {
            other_p = static_cast<const PyNumber &>(other).pyobject_;
        } else {
            other_p = pymodule_->to_py_(other.rcp_from_this_cast<const Basic>());
        }
        return make_rcp<PyNumber>(PyNumber_Power(other_p, pyobject_, Py_None), pymodule_);
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
