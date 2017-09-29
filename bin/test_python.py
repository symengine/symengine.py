import os
TEST_SYMPY = os.getenv('TEST_SYMPY', '0')

import symengine
if not symengine.test():
    raise Exception('Tests failed')

try:
    import sympy
    from sympy.core.cache import clear_cache
    import atexit

    atexit.register(clear_cache)
    have_sympy = True
except ImportError:
    have_sympy = False

if TEST_SYMPY and have_sympy:
    print('Testing SYMPY')
    if not sympy.test('sympy/physics/mechanics'):
        raise Exception('Tests failed')
    if not sympy.test('sympy/liealgebras'):
        raise Exception('Tests failed')
