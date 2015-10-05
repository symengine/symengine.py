#ifndef SYMENGINE_PYWRAPPER_H
#define SYMENGINE_PYWRAPPER_H

#include <Python.h>
#include <symengine/number.h>
#include <symengine/constants.h>
#include <symengine/functions.h>

namespace SymEngine {

class PyModule : public EnableRCPFromThis<PyModule> {
public:
    PyObject* (*to_py_)(const RCP<const Basic> x);
    RCP<const Basic> (*from_py_)(PyObject*);
    RCP<const Number> (*eval_)(PyObject*, long bits);
    PyObject *one, *zero, *minus_one;
public:
    PyModule(PyObject* (*)(const RCP<const Basic> x), RCP<const Basic> (*)(PyObject*),
             RCP<const Number> (*)(PyObject*, long bits));
    ~PyModule();
    PyObject* get_zero() const { return zero; }
    PyObject* get_one() const { return one; }
    PyObject* get_minus_one() const { return minus_one; }
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
    PyObject* get_py_object() const { return pyobject_; }
    RCP<const PyModule> get_py_module() const { return pymodule_; }
    //! \return true if `0`
    virtual bool is_zero() const;
    //! \return true if `1`
    virtual bool is_one() const;
    //! \return true if `-1`
    virtual bool is_minus_one() const;
    //! \return true if negative
    virtual bool is_negative() const;
    //! \return true if positive
    virtual bool is_positive() const;
    //! return true if the number is an exact representation
    //  false if the number is an approximation
    virtual bool is_exact() const { return true; };

    //! Addition
    virtual RCP<const Number> add(const Number &other) const;
    //! Subtraction
    virtual RCP<const Number> sub(const Number &other) const;
    virtual RCP<const Number> rsub(const Number &other) const;
    //! Multiplication
    virtual RCP<const Number> mul(const Number &other) const;
    //! Division
    virtual RCP<const Number> div(const Number &other) const;
    virtual RCP<const Number> rdiv(const Number &other) const;
    //! Power
    virtual RCP<const Number> pow(const Number &other) const;
    virtual RCP<const Number> rpow(const Number &other) const;

    //! Differentiation w.r.t Symbol `x`
    virtual RCP<const Basic> diff(const RCP<const Symbol> &x) const;
    virtual RCP<const Number> eval(long bits) const;
    virtual std::string __str__() const;
    virtual int compare(const Basic &o) const;
    virtual bool __eq__(const Basic &o) const;
    virtual std::size_t __hash__() const;
};


class PyFunctionClass : public EnableRCPFromThis<PyFunctionClass> {
private:
    PyObject *pyobject_;
    std::string name_;
    mutable size_t hash_;
    RCP<const PyModule> pymodule_;
public:
    PyFunctionClass(PyObject *pyobject, std::string name, const RCP<const PyModule> &pymodule);
    PyObject* get_py_object() const { return pyobject_; }
    RCP<const PyModule> get_py_module() const { return pymodule_; }
    std::string get_name() const { return name_; }

    PyObject* call(const vec_basic &vec) const;
    bool __eq__(const PyFunctionClass &x) const;
    int compare(const PyFunctionClass &x) const;
    std::size_t hash() const;
};

class PyFunction : public FunctionSymbol {
private:
    RCP<const PyFunctionClass> pyfunction_class_;
    PyObject *pyobject_;
public:
    PyFunction(const vec_basic &vec, const RCP<const PyFunctionClass> &pyfunc_class,
               PyObject *pyobject);
    ~PyFunction();

    PyObject *get_py_object() const;
    virtual RCP<const Basic> create(const vec_basic &x) const;
    virtual RCP<const Number> eval(long bits) const;
};

}

#endif //SYMENGINE_PYWRAPPER_H
