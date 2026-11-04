from symengine import (ccode, unicode, Symbol, sqrt, Pow, Max, sin, Integer, MutableDenseMatrix)
from symengine.test_utilities import raises
from symengine.printing import CCodePrinter, init_printing

def test_ccode():
    x = Symbol("x")
    y = Symbol("y")
    assert ccode(x) == "x"
    assert ccode(x**3) == "pow(x, 3)"
    assert ccode(x**(y**3)) == "pow(x, pow(y, 3))"
    assert ccode(x**-1.0) == "pow(x, -1.0)"
    assert ccode(Max(x, x*x)) == "fmax(x, pow(x, 2))"
    assert ccode(sin(x)) == "sin(x)"
    assert ccode(Integer(67)) == "67"
    assert ccode(Integer(-1)) == "-1"

def test_CCodePrinter():
    x = Symbol("x")
    y = Symbol("y")
    myprinter = CCodePrinter()

    assert myprinter.doprint(1+x, "bork") == "bork = 1 + x;"
    assert myprinter.doprint(1*x) == "x"
    assert myprinter.doprint(MutableDenseMatrix(1, 2, [x, y]), "larry") == "larry[0] = x;\nlarry[1] = y;"
    raises(TypeError, lambda: myprinter.doprint(sin(x), Integer))
    raises(RuntimeError, lambda: myprinter.doprint(MutableDenseMatrix(1, 2, [x, y])))

def test_init_printing():
    x = Symbol("x")
    assert x._repr_latex_() is None
    init_printing()
    assert x._repr_latex_() == '$x$'


def test_unicode():
    x = Symbol("x")
    y = Integer(2)
    assert unicode(x / 2) == "x\n―\n2"


def test_printer_dispatch():
    from symengine import Rational
    from symengine.printing import Printer

    class MyPrinter(Printer):
        def _print_Basic(self, expr):
            return "Basic"

        def _print_Integer(self, expr):
            return "Integer"

        def _print_list(self, expr):
            return "[" + ", ".join(self._print(e) for e in expr) + "]"

    p = MyPrinter()
    assert p.doprint(Integer(2)) == "Integer"
    assert p.doprint(Rational(1, 2)) == "Basic"
    assert p.doprint([Integer(2), Symbol("x")]) == "[Integer, Basic]"
    assert p.doprint("abc") == "abc"


def test_printmethod():
    from symengine.printing import Printer

    class MyPrinter(Printer):
        printmethod = "_myprint"

    class Wrapped:
        def _myprint(self, printer):
            return "W(" + printer._print(Symbol("x")) + ")"

    assert MyPrinter().doprint(Wrapped()) == "W(x)"


def test_printer_function_symbol():
    from symengine import function_symbol
    from symengine.printing import Printer

    x = Symbol("x")

    class MyPrinter(Printer):
        def _print_gamma(self, expr):
            return "gamma"

        def _print_Function(self, expr):
            return "Function"

    # Like sympy, an undefined function is not dispatched on its name
    assert MyPrinter().doprint(function_symbol("gamma", x)) == "Function"

    class MyPrinter2(MyPrinter):
        def _print_AppliedUndef(self, expr):
            return "AppliedUndef " + expr.get_name()

    assert MyPrinter2().doprint(function_symbol("f", x)) == "AppliedUndef f"


def test_applied_undef():
    from symengine import AppliedUndef, FunctionSymbol, function_symbol

    f = function_symbol("f", Symbol("x"))
    assert FunctionSymbol is AppliedUndef
    assert type(f) is AppliedUndef
    assert type(f).__name__ == "AppliedUndef"


def test_printer_pyfunction():
    try:
        import sympy
    except ImportError:
        return
    from symengine import sympify
    from symengine.printing import Printer

    class f(sympy.Function):
        pass

    class g(f):
        pass

    class fBase(sympy.Function):
        pass

    class h(fBase):
        pass

    x = sympy.Symbol("x")

    class MyPrinter(Printer):
        def _print_f(self, expr):
            return "f"

        def _print_fBase(self, expr):
            return "fBase"

        def _print_PyFunction(self, expr):
            return "PyFunction " + expr.get_name()

    p = MyPrinter()
    assert p.doprint(sympify(f(x))) == "f"
    # Like sympy, parent classes with a different name are skipped ...
    assert p.doprint(sympify(g(x))) == "PyFunction g"
    # ... unless their name ends in "Base"
    assert p.doprint(sympify(h(x))) == "fBase"
    # A wrapped sympy function is not an undefined function
    MyPrinter._print_AppliedUndef = lambda self, expr: "AppliedUndef"
    del MyPrinter._print_PyFunction
    assert p.doprint(sympify(g(x))) == "g(x)"
