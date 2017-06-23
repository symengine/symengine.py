from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from cpython.ref cimport PyObject
from libcpp.pair cimport pair

include "config.pxi"

cdef extern from 'symengine/mp_class.h' namespace "SymEngine":
    ctypedef unsigned long mp_limb_t
    ctypedef struct __mpz_struct:
        pass
    ctypedef struct __mpq_struct:
        pass
    ctypedef __mpz_struct mpz_t[1]
    ctypedef __mpq_struct mpq_t[1]

    cdef cppclass integer_class:
        integer_class()
        integer_class(int i)
        integer_class(integer_class)
        integer_class(mpz_t)
        integer_class(const string &s) except +
    mpz_t get_mpz_t(integer_class &a)
    const mpz_t get_mpz_t(const integer_class &a)
    cdef cppclass rational_class:
        rational_class()
        rational_class(mpq_t)
    const mpq_t get_mpq_t(const rational_class &a)

cdef extern from "<set>" namespace "std":
# Cython's libcpp.set does not support two template arguments to set.
# Methods to declare and iterate a set with a custom compare are given here
    cdef cppclass set[T, U]:
        cppclass iterator:
            T& operator*()
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        iterator begin() nogil
        iterator end() nogil

    cdef cppclass multiset[T, U]:
         cppclass iterator:
             T& operator*()
             iterator operator++() nogil
             iterator operator--() nogil
             bint operator==(iterator) nogil
             bint operator!=(iterator) nogil
         iterator begin() nogil
         iterator end() nogil
         iterator insert(T&) nogil

cdef extern from "<unordered_map>" namespace "std" nogil:
    cdef cppclass unordered_map[T, U]:
        cppclass iterator:
            pair[T, U]& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
        cppclass reverse_iterator:
            pair[T, U]& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
        cppclass const_iterator(iterator):
            pass
        cppclass const_reverse_iterator(reverse_iterator):
            pass
        unordered_map() except +
        unordered_map(unordered_map&) except +
        #unordered_map(key_compare&)
        U& operator[](T&)
        #unordered_map& operator=(unordered_map&)
        bint operator==(unordered_map&, unordered_map&)
        bint operator!=(unordered_map&, unordered_map&)
        bint operator<(unordered_map&, unordered_map&)
        bint operator>(unordered_map&, unordered_map&)
        bint operator<=(unordered_map&, unordered_map&)
        bint operator>=(unordered_map&, unordered_map&)
        U& at(T&)
        iterator begin()
        const_iterator const_begin "begin"()
        void clear()
        size_t count(T&)
        bint empty()
        iterator end()
        const_iterator const_end "end"()
        pair[iterator, iterator] equal_range(T&)
        #pair[const_iterator, const_iterator] equal_range(key_type&)
        void erase(iterator)
        void erase(iterator, iterator)
        size_t erase(T&)
        iterator find(T&)
        const_iterator const_find "find"(T&)
        pair[iterator, bint] insert(pair[T, U]) # XXX pair[T,U]&
        iterator insert(iterator, pair[T, U]) # XXX pair[T,U]&
        #void insert(input_iterator, input_iterator)
        #key_compare key_comp()
        iterator lower_bound(T&)
        const_iterator const_lower_bound "lower_bound"(T&)
        size_t max_size()
        reverse_iterator rbegin()
        const_reverse_iterator const_rbegin "rbegin"()
        reverse_iterator rend()
        const_reverse_iterator const_rend "rend"()
        size_t size()
        void swap(unordered_map&)
        iterator upper_bound(T&)
        const_iterator const_upper_bound "upper_bound"(T&)
        #value_compare value_comp()
        void max_load_factor(float)
        float max_load_factor()

cdef extern from "<symengine/symengine_rcp.h>" namespace "SymEngine":
    cdef enum ENull:
        null

    cdef cppclass RCP[T]:
        T& operator*() nogil
        # Not yet supported in Cython:
#        RCP[T]& operator=(RCP[T] &r_ptr) nogil except +
        void reset() nogil except +

    cdef cppclass Ptr[T]:
        T& operator*() nogil except +

    RCP[const Symbol] rcp_static_cast_Symbol "SymEngine::rcp_static_cast<const SymEngine::Symbol>"(RCP[const Basic] &b) nogil
    RCP[const PySymbol] rcp_static_cast_PySymbol "SymEngine::rcp_static_cast<const SymEngine::PySymbol>"(RCP[const Basic] &b) nogil
    RCP[const Integer] rcp_static_cast_Integer "SymEngine::rcp_static_cast<const SymEngine::Integer>"(RCP[const Basic] &b) nogil
    RCP[const Rational] rcp_static_cast_Rational "SymEngine::rcp_static_cast<const SymEngine::Rational>"(RCP[const Basic] &b) nogil
    RCP[const Complex] rcp_static_cast_Complex "SymEngine::rcp_static_cast<const SymEngine::Complex>"(RCP[const Basic] &b) nogil
    RCP[const Number] rcp_static_cast_Number "SymEngine::rcp_static_cast<const SymEngine::Number>"(RCP[const Basic] &b) nogil
    RCP[const Add] rcp_static_cast_Add "SymEngine::rcp_static_cast<const SymEngine::Add>"(RCP[const Basic] &b) nogil
    RCP[const Mul] rcp_static_cast_Mul "SymEngine::rcp_static_cast<const SymEngine::Mul>"(RCP[const Basic] &b) nogil
    RCP[const Pow] rcp_static_cast_Pow "SymEngine::rcp_static_cast<const SymEngine::Pow>"(RCP[const Basic] &b) nogil
    RCP[const OneArgFunction] rcp_static_cast_OneArgFunction "SymEngine::rcp_static_cast<const SymEngine::OneArgFunction>"(RCP[const Basic] &b) nogil
    RCP[const FunctionSymbol] rcp_static_cast_FunctionSymbol "SymEngine::rcp_static_cast<const SymEngine::FunctionSymbol>"(RCP[const Basic] &b) nogil
    RCP[const FunctionWrapper] rcp_static_cast_FunctionWrapper "SymEngine::rcp_static_cast<const SymEngine::FunctionWrapper>"(RCP[const Basic] &b) nogil
    RCP[const Abs] rcp_static_cast_Abs "SymEngine::rcp_static_cast<const SymEngine::Abs>"(RCP[const Basic] &b) nogil
    RCP[const Max] rcp_static_cast_Max "SymEngine::rcp_static_cast<const SymEngine::Max>"(RCP[const Basic] &b) nogil
    RCP[const Min] rcp_static_cast_Min "SymEngine::rcp_static_cast<const SymEngine::Min>"(RCP[const Basic] &b) nogil
    RCP[const Infty] rcp_static_cast_Infty "SymEngine::rcp_static_cast<const SymEngine::Infty>"(RCP[const Basic] &b) nogil
    RCP[const Gamma] rcp_static_cast_Gamma "SymEngine::rcp_static_cast<const SymEngine::Gamma>"(RCP[const Basic] &b) nogil
    RCP[const Derivative] rcp_static_cast_Derivative "SymEngine::rcp_static_cast<const SymEngine::Derivative>"(RCP[const Basic] &b) nogil
    RCP[const Subs] rcp_static_cast_Subs "SymEngine::rcp_static_cast<const SymEngine::Subs>"(RCP[const Basic] &b) nogil
    RCP[const RealDouble] rcp_static_cast_RealDouble "SymEngine::rcp_static_cast<const SymEngine::RealDouble>"(RCP[const Basic] &b) nogil
    RCP[const ComplexDouble] rcp_static_cast_ComplexDouble "SymEngine::rcp_static_cast<const SymEngine::ComplexDouble>"(RCP[const Basic] &b) nogil
    RCP[const RealMPFR] rcp_static_cast_RealMPFR "SymEngine::rcp_static_cast<const SymEngine::RealMPFR>"(RCP[const Basic] &b) nogil
    RCP[const ComplexMPC] rcp_static_cast_ComplexMPC "SymEngine::rcp_static_cast<const SymEngine::ComplexMPC>"(RCP[const Basic] &b) nogil
    RCP[const Log] rcp_static_cast_Log "SymEngine::rcp_static_cast<const SymEngine::Log>"(RCP[const Basic] &b) nogil
    RCP[const PyNumber] rcp_static_cast_PyNumber "SymEngine::rcp_static_cast<const SymEngine::PyNumber>"(RCP[const Basic] &b) nogil
    RCP[const PyFunction] rcp_static_cast_PyFunction "SymEngine::rcp_static_cast<const SymEngine::PyFunction>"(RCP[const Basic] &b) nogil
    Ptr[RCP[Basic]] outArg(RCP[const Basic] &arg) nogil
    Ptr[RCP[Integer]] outArg_Integer "SymEngine::outArg<SymEngine::RCP<const SymEngine::Integer>>"(RCP[const Integer] &arg) nogil

    void print_stack_on_segfault() nogil

cdef extern from "<symengine/basic.h>" namespace "SymEngine":
    ctypedef Basic const_Basic "const SymEngine::Basic"
    # Cython has broken support for the following:
    # ctypedef map[RCP[const Basic], RCP[const Basic]] map_basic_basic
    # So instead we replicate the map features we need here
    cdef cppclass std_pair_short_rcp_const_basic "std::pair<short, SymEngine::RCP<const SymEngine::Basic>>":
        short first
        RCP[const Basic] second

    cdef cppclass std_pair_rcp_const_basic_rcp_const_basic "std::pair<SymEngine::RCP<const SymEngine::Basic>, SymEngine::RCP<const SymEngine::Basic>>":
        RCP[const Basic] first
        RCP[const Basic] second

    cdef cppclass map_basic_basic:
        map_basic_basic() except +
        map_basic_basic(map_basic_basic&) except +
        cppclass iterator:
            std_pair_rcp_const_basic_rcp_const_basic& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
        RCP[const Basic]& operator[](RCP[const Basic]&)
        void clear()
        bint empty()
        size_t size()
        void swap(map_basic_basic&)
        iterator begin()
        iterator end()
        iterator find(RCP[const Basic]&)
        void erase(iterator, iterator)
        void erase_it(iterator)
        size_t erase(RCP[const Basic]&)
        pair[iterator, bint] insert(std_pair_rcp_const_basic_rcp_const_basic) except +
        iterator insert(iterator, std_pair_rcp_const_basic_rcp_const_basic) except +
        void insert(iterator, iterator) except +

    ctypedef vector[RCP[Basic]] vec_basic "SymEngine::vec_basic"
    ctypedef vector[RCP[Integer]] vec_integer "SymEngine::vec_integer"
    ctypedef map[RCP[Integer], unsigned] map_integer_uint "SymEngine::map_integer_uint"
    cdef struct RCPIntegerKeyLess
    cdef struct RCPBasicKeyLess
    ctypedef set[RCP[const_Basic], RCPBasicKeyLess] set_basic "SymEngine::set_basic"
    ctypedef multiset[RCP[const_Basic], RCPBasicKeyLess] multiset_basic "SymEngine::multiset_basic"
    cdef cppclass Basic:
        string __str__() nogil except +
        unsigned int hash() nogil except +
        vec_basic get_args() nogil
        int __cmp__(const Basic &o) nogil
    ctypedef RCP[const Basic] rcp_const_basic "SymEngine::RCP<const SymEngine::Basic>"
    ctypedef RCP[const Number] rcp_const_number "SymEngine::RCP<const SymEngine::Number>"
    ctypedef unordered_map[int, rcp_const_basic] umap_int_basic "SymEngine::umap_int_basic"
    ctypedef unordered_map[int, rcp_const_basic].iterator umap_int_basic_iterator "SymEngine::umap_int_basic::iterator"
    ctypedef unordered_map[rcp_const_basic, rcp_const_number] umap_basic_num "SymEngine::umap_basic_num"
    ctypedef unordered_map[rcp_const_basic, rcp_const_number].iterator umap_basic_num_iterator "SymEngine::umap_basic_num::iterator"

    bool eq(const Basic &a, const Basic &b) nogil except +
    bool neq(const Basic &a, const Basic &b) nogil except +

    bool is_a_Add "SymEngine::is_a<SymEngine::Add>"(const Basic &b) nogil
    bool is_a_Mul "SymEngine::is_a<SymEngine::Mul>"(const Basic &b) nogil
    bool is_a_Pow "SymEngine::is_a<SymEngine::Pow>"(const Basic &b) nogil
    bool is_a_Integer "SymEngine::is_a<SymEngine::Integer>"(const Basic &b) nogil
    bool is_a_Rational "SymEngine::is_a<SymEngine::Rational>"(const Basic &b) nogil
    bool is_a_Complex "SymEngine::is_a<SymEngine::Complex>"(const Basic &b) nogil
    bool is_a_Symbol "SymEngine::is_a<SymEngine::Symbol>"(const Basic &b) nogil
    bool is_a_Constant "SymEngine::is_a<SymEngine::Constant>"(const Basic &b) nogil
    bool is_a_Infty "SymEngine::is_a<SymEngine::Infty>"(const Basic &b) nogil
    bool is_a_NaN "SymEngine::is_a<SymEngine::NaN>"(const Basic &b) nogil
    bool is_a_Sin "SymEngine::is_a<SymEngine::Sin>"(const Basic &b) nogil
    bool is_a_Cos "SymEngine::is_a<SymEngine::Cos>"(const Basic &b) nogil
    bool is_a_Tan "SymEngine::is_a<SymEngine::Tan>"(const Basic &b) nogil
    bool is_a_Cot "SymEngine::is_a<SymEngine::Cot>"(const Basic &b) nogil
    bool is_a_Csc "SymEngine::is_a<SymEngine::Csc>"(const Basic &b) nogil
    bool is_a_Sec "SymEngine::is_a<SymEngine::Sec>"(const Basic &b) nogil
    bool is_a_ASin "SymEngine::is_a<SymEngine::ASin>"(const Basic &b) nogil
    bool is_a_ACos "SymEngine::is_a<SymEngine::ACos>"(const Basic &b) nogil
    bool is_a_ATan "SymEngine::is_a<SymEngine::ATan>"(const Basic &b) nogil
    bool is_a_ACot "SymEngine::is_a<SymEngine::ACot>"(const Basic &b) nogil
    bool is_a_ACsc "SymEngine::is_a<SymEngine::ACsc>"(const Basic &b) nogil
    bool is_a_ASec "SymEngine::is_a<SymEngine::ASec>"(const Basic &b) nogil
    bool is_a_Sinh "SymEngine::is_a<SymEngine::Sinh>"(const Basic &b) nogil
    bool is_a_Cosh "SymEngine::is_a<SymEngine::Cosh>"(const Basic &b) nogil
    bool is_a_Tanh "SymEngine::is_a<SymEngine::Tanh>"(const Basic &b) nogil
    bool is_a_Coth "SymEngine::is_a<SymEngine::Coth>"(const Basic &b) nogil
    bool is_a_Csch "SymEngine::is_a<SymEngine::Csch>"(const Basic &b) nogil
    bool is_a_Sech "SymEngine::is_a<SymEngine::Sech>"(const Basic &b) nogil
    bool is_a_ASinh "SymEngine::is_a<SymEngine::ASinh>"(const Basic &b) nogil
    bool is_a_ACosh "SymEngine::is_a<SymEngine::ACosh>"(const Basic &b) nogil
    bool is_a_ATanh "SymEngine::is_a<SymEngine::ATanh>"(const Basic &b) nogil
    bool is_a_ACoth "SymEngine::is_a<SymEngine::ACoth>"(const Basic &b) nogil
    bool is_a_ACsch "SymEngine::is_a<SymEngine::ACsch>"(const Basic &b) nogil
    bool is_a_ASech "SymEngine::is_a<SymEngine::ASech>"(const Basic &b) nogil
    bool is_a_FunctionSymbol "SymEngine::is_a<SymEngine::FunctionSymbol>"(const Basic &b) nogil
    bool is_a_Abs "SymEngine::is_a<SymEngine::Abs>"(const Basic &b) nogil
    bool is_a_Max "SymEngine::is_a<SymEngine::Max>"(const Basic &b) nogil
    bool is_a_Min "SymEngine::is_a<SymEngine::Min>"(const Basic &b) nogil
    bool is_a_Gamma "SymEngine::is_a<SymEngine::Gamma>"(const Basic &b) nogil
    bool is_a_Derivative "SymEngine::is_a<SymEngine::Derivative>"(const Basic &b) nogil
    bool is_a_Subs "SymEngine::is_a<SymEngine::Subs>"(const Basic &b) nogil
    bool is_a_PyFunction "SymEngine::is_a<SymEngine::FunctionWrapper>"(const Basic &b) nogil
    bool is_a_RealDouble "SymEngine::is_a<SymEngine::RealDouble>"(const Basic &b) nogil
    bool is_a_ComplexDouble "SymEngine::is_a<SymEngine::ComplexDouble>"(const Basic &b) nogil
    bool is_a_RealMPFR "SymEngine::is_a<SymEngine::RealMPFR>"(const Basic &b) nogil
    bool is_a_ComplexMPC "SymEngine::is_a<SymEngine::ComplexMPC>"(const Basic &b) nogil
    bool is_a_Log "SymEngine::is_a<SymEngine::Log>"(const Basic &b) nogil
    bool is_a_PyNumber "SymEngine::is_a<SymEngine::PyNumber>"(const Basic &b) nogil
    bool is_a_ATan2 "SymEngine::is_a<SymEngine::ATan2>"(const Basic &b) nogil
    bool is_a_PySymbol "SymEngine::is_a_sub<SymEngine::PySymbol>"(const Basic &b) nogil

    RCP[const Basic] expand(RCP[const Basic] &o) nogil except +

cdef extern from "<symengine/subs.h>" namespace "SymEngine":
    RCP[const Basic] msubs (RCP[const Basic] &x, const map_basic_basic &x) nogil
    RCP[const Basic] ssubs (RCP[const Basic] &x, const map_basic_basic &x) nogil

cdef extern from "<symengine/derivative.h>" namespace "SymEngine":
    RCP[const Basic] diff "SymEngine::sdiff"(RCP[const Basic] &arg, RCP[const Basic] &x) nogil except +

cdef extern from "<symengine/symbol.h>" namespace "SymEngine":
    cdef cppclass Symbol(Basic):
        Symbol(string name) nogil
        string get_name() nogil

cdef extern from "<symengine/number.h>" namespace "SymEngine":
    cdef cppclass Number(Basic):
        bool is_positive() nogil
        bool is_negative() nogil
        bool is_zero() nogil
        bool is_complex() nogil
        pass
    cdef cppclass NumberWrapper(Basic):
        pass

cdef extern from "pywrapper.h" namespace "SymEngine":
    cdef cppclass PyNumber(NumberWrapper):
        PyObject* get_py_object()
    cdef cppclass PyModule:
        pass
    cdef cppclass PyFunctionClass:
        PyObject* call(const vec_basic &vec)
    cdef cppclass PyFunction:
        PyObject* get_py_object()

cdef extern from "pywrapper.h" namespace "SymEngine":
    cdef cppclass PySymbol(Symbol):
        PySymbol(string name, PyObject* pyobj)
        PyObject* get_py_object()

cdef extern from "<symengine/integer.h>" namespace "SymEngine":
    cdef cppclass Integer(Number):
        Integer(int i) nogil
        Integer(integer_class i) nogil
        int compare(const Basic &o) nogil
        integer_class as_mpz() nogil
    cdef RCP[const Integer] integer(int i) nogil
    cdef RCP[const Integer] integer(integer_class i) nogil

cdef extern from "<symengine/rational.h>" namespace "SymEngine":
    cdef cppclass Rational(Number):
        rational_class as_mpq() nogil
    cdef RCP[const Number] from_mpq "SymEngine::Rational::from_mpq"(rational_class r) nogil
    cdef void get_num_den(const Rational &rat, const Ptr[RCP[Integer]] &num,
                     const Ptr[RCP[Integer]] &den) nogil

cdef extern from "<symengine/complex.h>" namespace "SymEngine":
    cdef cppclass Complex(Number):
        RCP[const Number] real_part() nogil
        RCP[const Number] imaginary_part() nogil

cdef extern from "<symengine/real_double.h>" namespace "SymEngine":
    cdef cppclass RealDouble(Number):
        RealDouble(double x) nogil
        double as_double() nogil
    RCP[const RealDouble] real_double(double d) nogil

cdef extern from "<symengine/complex_double.h>" namespace "SymEngine":
    cdef cppclass ComplexDouble(Number):
        ComplexDouble(double complex x) nogil
        RCP[const Number] real_part() nogil
        RCP[const Number] imaginary_part() nogil
        double complex as_complex_double() nogil
    RCP[const ComplexDouble] complex_double(double complex d) nogil

cdef extern from "<symengine/constants.h>" namespace "SymEngine":
    cdef cppclass Constant(Basic):
        Constant(string name) nogil
        string get_name() nogil
    RCP[const Basic] I
    RCP[const Basic] E
    RCP[const Basic] pi
    RCP[const Basic] Inf
    RCP[const Basic] ComplexInf
    RCP[const Basic] Nan
  
cdef extern from "<symengine/infinity.h>" namespace "SymEngine":
    cdef cppclass Infty(Number):
        pass

cdef extern from "<symengine/nan.h>" namespace "SymEngine":
    cdef cppclass NaN(Number):
        pass

cdef extern from "<symengine/add.h>" namespace "SymEngine":
    cdef RCP[const Basic] add(RCP[const Basic] &a, RCP[const Basic] &b) nogil except+
    cdef RCP[const Basic] sub(RCP[const Basic] &a, RCP[const Basic] &b) nogil except+
    cdef RCP[const Basic] add(const vec_basic &a) nogil except+

    cdef cppclass Add(Basic):
        void as_two_terms(const Ptr[RCP[Basic]] &a, const Ptr[RCP[Basic]] &b)
        RCP[const Number] get_coef()
        const umap_basic_num &get_dict()

cdef extern from "<symengine/mul.h>" namespace "SymEngine":
    cdef RCP[const Basic] mul(RCP[const Basic] &a, RCP[const Basic] &b) nogil except+
    cdef RCP[const Basic] div(RCP[const Basic] &a, RCP[const Basic] &b) nogil except+
    cdef RCP[const Basic] neg(RCP[const Basic] &a) nogil except+
    cdef RCP[const Basic] mul(const vec_basic &a) nogil except+

    cdef cppclass Mul(Basic):
        void as_two_terms(const Ptr[RCP[Basic]] &a, const Ptr[RCP[Basic]] &b)
        RCP[const Number] get_coef()
        const map_basic_basic &get_dict()
    cdef RCP[const Mul] mul_from_dict "SymEngine::Mul::from_dict"(RCP[const Number] coef, map_basic_basic &&d) nogil

cdef extern from "<symengine/pow.h>" namespace "SymEngine":
    cdef RCP[const Basic] pow(RCP[const Basic] &a, RCP[const Basic] &b) nogil except+
    cdef RCP[const Basic] sqrt(RCP[const Basic] &x) nogil except+
    cdef RCP[const Basic] exp(RCP[const Basic] &x) nogil except+
    cdef RCP[const Basic] log(RCP[const Basic] &x) nogil except+
    cdef RCP[const Basic] log(RCP[const Basic] &x, RCP[const Basic] &y) nogil except+

    cdef cppclass Pow(Basic):
        RCP[const Basic] get_base() nogil
        RCP[const Basic] get_exp() nogil

    cdef cppclass Log(Basic):
        RCP[const Basic] get_arg() nogil


cdef extern from "<symengine/basic.h>" namespace "SymEngine":
    # We need to specialize these for our classes:
    RCP[const Basic] make_rcp_Symbol "SymEngine::make_rcp<const SymEngine::Symbol>"(string name) nogil
    RCP[const Basic] make_rcp_PySymbol "SymEngine::make_rcp<const SymEngine::PySymbol>"(string name, PyObject * pyobj) nogil
    RCP[const Basic] make_rcp_Constant "SymEngine::make_rcp<const SymEngine::Constant>"(string name) nogil
    RCP[const Basic] make_rcp_Infty "SymEngine::make_rcp<const SymEngine::Infty>"(RCP[const Number] i) nogil
    RCP[const Basic] make_rcp_NaN "SymEngine::make_rcp<const SymEngine::NaN>"() nogil
    RCP[const Basic] make_rcp_Integer "SymEngine::make_rcp<const SymEngine::Integer>"(int i) nogil
    RCP[const Basic] make_rcp_Integer "SymEngine::make_rcp<const SymEngine::Integer>"(integer_class i) nogil
    RCP[const Basic] make_rcp_Subs "SymEngine::make_rcp<const SymEngine::Subs>"(RCP[const Basic] arg, const map_basic_basic &x) nogil
    RCP[const Basic] make_rcp_Derivative "SymEngine::make_rcp<const SymEngine::Derivative>"(RCP[const Basic] arg, const multiset_basic &x) nogil
    RCP[const Basic] make_rcp_FunctionWrapper "SymEngine::make_rcp<const SymEngine::FunctionWrapper>"(void* obj, string name, string hash_, const vec_basic &arg, \
            void (*dec_ref)(void *), int (*comp)(void *, void *)) nogil
    RCP[const Basic] make_rcp_RealDouble "SymEngine::make_rcp<const SymEngine::RealDouble>"(double x) nogil
    RCP[const Basic] make_rcp_ComplexDouble "SymEngine::make_rcp<const SymEngine::ComplexDouble>"(double complex x) nogil
    RCP[const PyModule] make_rcp_PyModule "SymEngine::make_rcp<const SymEngine::PyModule>"(PyObject* (*) (RCP[const Basic] x), \
            RCP[const Basic] (*)(PyObject*), RCP[const Number] (*)(PyObject*, long bits),
            RCP[const Basic] (*)(PyObject*, RCP[const Basic])) nogil
    RCP[const Basic] make_rcp_PyNumber "SymEngine::make_rcp<const SymEngine::PyNumber>"(PyObject*, RCP[const PyModule] x) nogil
    RCP[const PyFunctionClass] make_rcp_PyFunctionClass "SymEngine::make_rcp<const SymEngine::PyFunctionClass>"(PyObject* pyobject,
            string name, RCP[const PyModule] pymodule) nogil
    RCP[const Basic] make_rcp_PyFunction "SymEngine::make_rcp<const SymEngine::PyFunction>" (const vec_basic &vec,
            RCP[const PyFunctionClass] pyfunc_class, const PyObject* pyobject) nogil

cdef extern from "<symengine/functions.h>" namespace "SymEngine":
    cdef RCP[const Basic] sin(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] cos(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] tan(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] cot(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] csc(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] sec(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] asin(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] acos(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] atan(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] acot(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] acsc(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] asec(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] sinh(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] cosh(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] tanh(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] coth(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] csch(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] sech(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] asinh(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] acosh(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] atanh(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] acoth(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] acsch(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] asech(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] function_symbol(string name, const vec_basic &arg) nogil except+
    cdef RCP[const Basic] abs(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] max(const vec_basic &arg) nogil except+
    cdef RCP[const Basic] min(const vec_basic &arg) nogil except+
    cdef RCP[const Basic] gamma(RCP[const Basic] &arg) nogil except+
    cdef RCP[const Basic] atan2(RCP[const Basic] &num, RCP[const Basic] &den) nogil except+

    cdef cppclass Function(Basic):
        pass

    cdef cppclass OneArgFunction(Function):
        RCP[const Basic] get_arg() nogil

    cdef cppclass TrigFunction(OneArgFunction):
        pass

    cdef cppclass Sin(TrigFunction):
        pass

    cdef cppclass Cos(TrigFunction):
        pass

    cdef cppclass Tan(TrigFunction):
        pass

    cdef cppclass Cot(TrigFunction):
        pass

    cdef cppclass Csc(TrigFunction):
        pass

    cdef cppclass Sec(TrigFunction):
        pass

    cdef cppclass ASin(TrigFunction):
        pass

    cdef cppclass ACos(TrigFunction):
        pass

    cdef cppclass ATan(TrigFunction):
        pass

    cdef cppclass ACot(TrigFunction):
        pass

    cdef cppclass ACsc(TrigFunction):
        pass

    cdef cppclass ASec(TrigFunction):
        pass

    cdef cppclass HyperbolicFunction(OneArgFunction):
        pass

    cdef cppclass Sinh(HyperbolicFunction):
        pass

    cdef cppclass Cosh(HyperbolicFunction):
        pass

    cdef cppclass Tanh(HyperbolicFunction):
        pass

    cdef cppclass Coth(HyperbolicFunction):
        pass

    cdef cppclass Csch(HyperbolicFunction):
        pass

    cdef cppclass Sech(HyperbolicFunction):
        pass

    cdef cppclass ASinh(HyperbolicFunction):
        pass

    cdef cppclass ACosh(HyperbolicFunction):
        pass

    cdef cppclass ATanh(HyperbolicFunction):
        pass

    cdef cppclass ACoth(HyperbolicFunction):
        pass

    cdef cppclass ACsch(HyperbolicFunction):
        pass

    cdef cppclass ASech(HyperbolicFunction):
        pass

    cdef cppclass FunctionSymbol(Function):
        string get_name() nogil

    cdef cppclass FunctionWrapper(FunctionSymbol):
        FunctionWrapper(void* obj, string name, string hash_, const vec_basic &arg, \
            void (*dec_ref)(void *), int (*comp)(void *, void *))
        void* get_object()

    cdef cppclass Derivative(Basic):
        Derivative(const RCP[const Basic] &arg, const vec_basic &x) nogil
        RCP[const Basic] get_arg() nogil
        multiset_basic get_symbols() nogil

    cdef cppclass Subs(Basic):
        Subs(const RCP[const Basic] &arg, const map_basic_basic &x) nogil
        RCP[const Basic] get_arg() nogil
        vec_basic get_variables() nogil
        vec_basic get_point() nogil

    cdef cppclass Abs(OneArgFunction):
        pass

    cdef cppclass Max(Function):
        pass

    cdef cppclass Min(Function):
        pass

    cdef cppclass Gamma(OneArgFunction):
        pass

    cdef cppclass ATan2(Function):
        pass

IF HAVE_SYMENGINE_MPFR:
    cdef extern from "mpfr.h":
        ctypedef struct __mpfr_struct:
            pass
        ctypedef __mpfr_struct mpfr_t[1]
        ctypedef __mpfr_struct* mpfr_ptr
        ctypedef const __mpfr_struct* mpfr_srcptr
        ctypedef long mpfr_prec_t
        ctypedef enum mpfr_rnd_t:
            MPFR_RNDN
            MPFR_RNDZ
            MPFR_RNDU
            MPFR_RNDD
            MPFR_RNDA
            MPFR_RNDF
            MPFR_RNDNA

    cdef extern from "<symengine/real_mpfr.h>" namespace "SymEngine":
        cdef cppclass mpfr_class:
            mpfr_class() nogil
            mpfr_class(mpfr_prec_t prec) nogil
            mpfr_class(string s, mpfr_prec_t prec, unsigned base) nogil
            mpfr_class(mpfr_t m) nogil
            mpfr_ptr get_mpfr_t() nogil

        cdef cppclass RealMPFR(Number):
            RealMPFR(mpfr_class) nogil
            mpfr_class as_mpfr() nogil
            mpfr_prec_t get_prec() nogil

        RCP[const RealMPFR] real_mpfr(mpfr_class t) nogil
ELSE:
    cdef extern from "<symengine/real_mpfr.h>" namespace "SymEngine":
        cdef cppclass RealMPFR(Number):
            pass

IF HAVE_SYMENGINE_MPC:
    cdef extern from "mpc.h":
        ctypedef struct __mpc_struct:
            pass
        ctypedef __mpc_struct mpc_t[1]
        ctypedef __mpc_struct* mpc_ptr
        ctypedef const __mpc_struct* mpc_srcptr

    cdef extern from "<symengine/complex_mpc.h>" namespace "SymEngine":
        cdef cppclass mpc_class:
            mpc_class() nogil
            mpc_class(mpfr_prec_t prec) nogil
            mpc_class(mpc_t m) nogil
            mpc_ptr get_mpc_t() nogil
            mpc_class(string s, mpfr_prec_t prec, unsigned base) nogil

        cdef cppclass ComplexMPC(Number):
            ComplexMPC(mpc_class) nogil
            mpc_class as_mpc() nogil
            mpfr_prec_t get_prec() nogil
            RCP[const Number] real_part() nogil
            RCP[const Number] imaginary_part() nogil

        RCP[const ComplexMPC] complex_mpc(mpc_class t) nogil
ELSE:
    cdef extern from "<symengine/complex_mpc.h>" namespace "SymEngine":
        cdef cppclass ComplexMPC(Number):
            pass

cdef extern from "<symengine/matrix.h>" namespace "SymEngine":
    cdef cppclass MatrixBase:
        const unsigned nrows() nogil
        const unsigned ncols() nogil
        RCP[const Basic] get(unsigned i, unsigned j) nogil
        RCP[const Basic] set(unsigned i, unsigned j, RCP[const Basic] e) nogil
        string __str__() nogil except+
        bool eq(const MatrixBase &) nogil
        RCP[const Basic] det() nogil
        void inv(MatrixBase &)
        void add_matrix(const MatrixBase &other, MatrixBase &result) nogil
        void mul_matrix(const MatrixBase &other, MatrixBase &result) nogil
        void add_scalar(RCP[const Basic] k, MatrixBase &result) nogil
        void mul_scalar(RCP[const Basic] k, MatrixBase &result) nogil
        void transpose(MatrixBase &result) nogil
        void submatrix(MatrixBase &result,
                       unsigned row_start, unsigned col_start,
                       unsigned row_end, unsigned col_end,
                       unsigned row_step, unsigned col_step) nogil
        void LU(MatrixBase &L, MatrixBase &U) nogil
        void LDL(MatrixBase &L, MatrixBase &D) nogil
        void LU_solve(const MatrixBase &b, MatrixBase &x) nogil
        void FFLU(MatrixBase &LU) nogil
        void FFLDU(MatrixBase&L, MatrixBase &D, MatrixBase &U) nogil

    cdef cppclass DenseMatrix(MatrixBase):
        DenseMatrix()
        DenseMatrix(unsigned i, unsigned j) nogil
        DenseMatrix(unsigned i, unsigned j, const vec_basic &v) nogil
        void resize(unsigned i, unsigned j) nogil

    bool is_a_DenseMatrix "SymEngine::is_a<SymEngine::DenseMatrix>"(const MatrixBase &b) nogil
    DenseMatrix* static_cast_DenseMatrix "static_cast<SymEngine::DenseMatrix*>"(const MatrixBase *a)
    void inverse_FFLU "SymEngine::inverse_fraction_free_LU"(const DenseMatrix &A,
        DenseMatrix &B) nogil except +
    void pivoted_LU (const DenseMatrix &A, DenseMatrix &L, DenseMatrix &U, vector[int] &P) nogil except +
    void pivoted_LU_solve (const DenseMatrix &A, const DenseMatrix &b, DenseMatrix &x) nogil except +
    void inverse_GJ "SymEngine::inverse_gauss_jordan"(const DenseMatrix &A,
        DenseMatrix &B) nogil except +
    void FFLU_solve "SymEngine::fraction_free_LU_solve"(const DenseMatrix &A,
        const DenseMatrix &b, DenseMatrix &x) nogil except +
    void FFGJ_solve "SymEngine::fraction_free_gauss_jordan_solve"(const DenseMatrix &A,
        const DenseMatrix &b, DenseMatrix &x) nogil except +
    void LDL_solve "SymEngine::LDL_solve"(const DenseMatrix &A, const DenseMatrix &b,
        DenseMatrix &x) nogil except +
    void jacobian "SymEngine::sjacobian"(const DenseMatrix &A,
            const DenseMatrix &x, DenseMatrix &result) nogil except +
    void diff "SymEngine::sdiff"(const DenseMatrix &A,
            RCP[const Basic] &x, DenseMatrix &result) nogil except +
    void eye (DenseMatrix &A, int k) nogil
    void diag(DenseMatrix &A, vec_basic &v, int k) nogil
    void ones(DenseMatrix &A) nogil
    void zeros(DenseMatrix &A) nogil

cdef extern from "<symengine/ntheory.h>" namespace "SymEngine":
    int probab_prime_p(const Integer &a, int reps)
    RCP[const Integer] nextprime (const Integer &a) nogil
    RCP[const Integer] gcd(const Integer &a, const Integer &b) nogil
    RCP[const Integer] lcm(const Integer &a, const Integer &b) nogil
    void gcd_ext(const Ptr[RCP[Integer]] &g, const Ptr[RCP[Integer]] &s,
            const Ptr[RCP[Integer]] &t, const Integer &a, const Integer &b) nogil
    RCP[const Integer] mod "SymEngine::mod_f"(const Integer &n, const Integer &d) nogil except +
    RCP[const Integer] quotient "SymEngine::quotient_f"(const Integer &n, const Integer &d) nogil except +
    void quotient_mod "SymEngine::quotient_mod_f"(const Ptr[RCP[Integer]] &q, const Ptr[RCP[Integer]] &mod,
            const Integer &n, const Integer &d) nogil except +
    int mod_inverse(const Ptr[RCP[Integer]] &b, const Integer &a,
            const Integer &m) nogil
    bool crt(const Ptr[RCP[Integer]] &R, const vec_integer &rem,
           const vec_integer &mod) nogil
    RCP[const Integer] fibonacci(unsigned long n) nogil
    void fibonacci2(const Ptr[RCP[Integer]] &g, const Ptr[RCP[Integer]] &s,
            unsigned long n) nogil
    RCP[const Integer] lucas(unsigned long n) nogil
    void lucas2(const Ptr[RCP[Integer]] &g, const Ptr[RCP[Integer]] &s,
            unsigned long n) nogil
    RCP[const Integer] binomial(const Integer &n,unsigned long k) nogil
    RCP[const Integer] factorial(unsigned long n) nogil
    bool divides(const Integer &a, const Integer &b) nogil
    int factor(const Ptr[RCP[Integer]] &f, const Integer &n, double B1) nogil
    int factor_lehman_method(const Ptr[RCP[Integer]] &f, const Integer &n) nogil
    int factor_pollard_pm1_method(const Ptr[RCP[Integer]] &f, const Integer &n,
            unsigned B, unsigned retries) nogil
    int factor_pollard_rho_method(const Ptr[RCP[Integer]] &f, const Integer &n,
            unsigned retries) nogil
    void prime_factors(vec_integer &primes, const Integer &n) nogil except +
    void prime_factor_multiplicities(map_integer_uint &primes, const Integer &n) nogil except +
    RCP[const Number] bernoulli(unsigned long n) nogil except +
    bool primitive_root(const Ptr[RCP[Integer]] &g, const Integer &n) nogil
    void primitive_root_list(vec_integer &roots, const Integer &n) nogil
    RCP[const Integer] totient(RCP[const Integer] n) nogil
    RCP[const Integer] carmichael(RCP[const Integer] n) nogil
    bool multiplicative_order(const Ptr[RCP[Integer]] &o, RCP[const Integer] a,
            RCP[const Integer] n) nogil
    int legendre(const Integer &a, const Integer &n) nogil
    int jacobi(const Integer &a, const Integer &n) nogil
    int kronecker(const Integer &a, const Integer &n) nogil
    void nthroot_mod_list(vec_integer &roots, RCP[const Integer] n,
            RCP[const Integer] a, RCP[const Integer] m) nogil
    bool nthroot_mod(const Ptr[RCP[Integer]] &root, RCP[const Integer] n,
            RCP[const Integer] a, RCP[const Integer] m) nogil
    bool powermod(const Ptr[RCP[Integer]] &powm, RCP[const Integer] a,
            RCP[const Number] b, RCP[const Integer] m) nogil
    void powermod_list(vec_integer &powm, RCP[const Integer] a,
            RCP[const Number] b, RCP[const Integer] m) nogil

    void sieve_generate_primes "SymEngine::Sieve::generate_primes"(vector[unsigned] &primes, unsigned limit) nogil

    cdef cppclass sieve_iterator "SymEngine::Sieve::iterator":
        sieve_iterator()
        sieve_iterator(unsigned limit) nogil
        unsigned next_prime() nogil

cdef extern from "<symengine/visitor.h>" namespace "SymEngine":
    bool has_symbol(const Basic &b, const Symbol &x) nogil except +
    RCP[const Basic] coeff(const Basic &b, const Basic &x, const Basic &n) nogil except +
    set_basic free_symbols(const Basic &b) nogil except +

cdef extern from "<utility>" namespace "std":
    cdef integer_class std_move_mpz "std::move" (integer_class) nogil
    IF HAVE_SYMENGINE_MPFR:
        cdef mpfr_class std_move_mpfr "std::move" (mpfr_class) nogil
    IF HAVE_SYMENGINE_MPC:
        cdef mpc_class std_move_mpc "std::move" (mpc_class) nogil
    cdef map_basic_basic std_move_map_basic_basic "std::move" (map_basic_basic) nogil

cdef extern from "<symengine/eval_double.h>" namespace "SymEngine":
    double eval_double(const Basic &b) nogil except +
    double complex eval_complex_double(const Basic &b) nogil except +

cdef extern from "<symengine/lambda_double.h>" namespace "SymEngine":
    cdef cppclass LambdaRealDoubleVisitor:
        LambdaRealDoubleVisitor() nogil
        void init(const vec_basic &x, const vec_basic &b) nogil except +
        double call(double *r, const double *x) nogil except +
    cdef cppclass LambdaComplexDoubleVisitor:
        LambdaComplexDoubleVisitor() nogil
        void init(const vec_basic &x, const vec_basic &b) nogil except +
        double complex call(double complex *r, const double complex *x) nogil except +

cdef extern from "<symengine/llvm_double.h>" namespace "SymEngine":
    cdef cppclass LLVMDoubleVisitor:
        LLVMDoubleVisitor() nogil
        void init(const vec_basic &x, const vec_basic &b) nogil except +
        double call(double *r, const double *x) nogil except +

cdef extern from "<symengine/series.h>" namespace "SymEngine":
    cdef cppclass SeriesCoeffInterface:
        rcp_const_basic as_basic() nogil except +
        umap_int_basic as_dict() nogil except +
        rcp_const_basic get_coeff(int) nogil except +
    ctypedef RCP[const SeriesCoeffInterface] rcp_const_seriescoeffinterface "SymEngine::RCP<const SymEngine::SeriesCoeffInterface>"
    rcp_const_seriescoeffinterface series "SymEngine::series"(RCP[const Basic] &ex, RCP[const Symbol] &var, unsigned int prec) nogil except +

IF HAVE_SYMENGINE_MPFR:
    cdef extern from "<symengine/eval_mpfr.h>" namespace "SymEngine":
        void eval_mpfr(mpfr_t result, const Basic &b, mpfr_rnd_t rnd) nogil except +

IF HAVE_SYMENGINE_MPC:
    cdef extern from "<symengine/eval_mpc.h>" namespace "SymEngine":
        void eval_mpc(mpc_t result, const Basic &b, mpfr_rnd_t rnd) nogil except +

cdef extern from "<symengine/parser.h>" namespace "SymEngine":
    RCP[const Basic] parse(const string &n) nogil except +

cdef extern from "<symengine/codegen.h>" namespace "SymEngine":
    string ccode(const Basic &x) nogil except +
