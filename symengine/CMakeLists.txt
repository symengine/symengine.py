add_subdirectory(lib)
add_subdirectory(tests)

install(FILES
        __init__.py utilities.py compatibility.py sympy_compat.py
        DESTINATION ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/symengine)
