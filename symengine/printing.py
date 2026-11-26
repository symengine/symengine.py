from .lib.symengine_wrapper import (ccode, sympify, Basic, PyFunction,
                                    repr_latex as _repr_latex)


def _dispatch_names(expr):
    names = [cls.__name__ for cls in type(expr).__mro__]
    if isinstance(expr, PyFunction):
        # Like sympy, a wrapped sympy Function subclass is dispatched on its
        # name, then on its parent classes ending in "Base", and is not an
        # undefined function, even though PyFunction subclasses AppliedUndef
        names.remove("AppliedUndef")
        bases = [cls.__name__ for cls in type(expr.pyobject()).__mro__
                 if cls.__name__.endswith("Base")]
        return [expr.get_name()] + bases + names
    return names


class Printer:
    """
    Base class for printers, modelled after ``sympy.printing.Printer``.

    ``doprint(expr)`` returns the printed form of ``expr``. Printing a node
    is dispatched, in order, to

    1. ``expr.<printmethod>(printer)`` if ``expr`` defines it,
    2. the first ``_print_<ClassName>`` method found along ``type(expr).__mro__``,
    3. ``emptyPrinter(expr)``.

    As in SymPy, an undefined function like ``Function("f")(x)`` is printed
    with ``_print_AppliedUndef`` or ``_print_Function``, never with a method
    named after the function.
    A function converted from a SymPy ``Function`` subclass ``f`` is printed
    with ``_print_f`` if defined, before the methods for ``PyFunction``.
    """

    printmethod = None

    def doprint(self, expr):
        return self._print(expr)

    def _print(self, expr, **kwargs):
        if (self.printmethod and not isinstance(expr, type)
                and hasattr(expr, self.printmethod)):
            return getattr(expr, self.printmethod)(self, **kwargs)
        for name in _dispatch_names(expr):
            method = getattr(self, "_print_" + name, None)
            if method is not None:
                return method(expr, **kwargs)
        return self.emptyPrinter(expr)

    def emptyPrinter(self, expr):
        return str(expr)


class CCodePrinter:

    def doprint(self, expr, assign_to=None):
        if not isinstance(assign_to, (Basic, type(None), str)):
            raise TypeError("{} cannot assign to object of type {}".format(
                    type(self).__name__, type(assign_to)))

        expr = sympify(expr)
        if not assign_to:
            if expr.is_Matrix:
                raise RuntimeError("Matrices need a assign_to parameter")
            return ccode(expr)

        assign_to = str(assign_to)
        if not expr.is_Matrix:
            return f"{assign_to} = {ccode(expr)};"

        code_lines = []
        for i, element in enumerate(expr):
            code_line = f'{assign_to}[{i}] = {element};'
            code_lines.append(code_line)
        return '\n'.join(code_lines)


def init_printing(pretty_print=True, use_latex=True):
    if pretty_print:
        if not use_latex:
            raise RuntimeError("Only latex is supported for pretty printing")
        _repr_latex[0] = True
    else:
        _repr_latex[0] = False
