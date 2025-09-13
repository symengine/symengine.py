add_subdirectory(lib)

if (SYMENGINE_INSTALL_PY_FILES)
  add_subdirectory(tests)
  set(PY_PATH ${PYTHON_INSTALL_PATH}/symengine)
  install(
    FILES
      __init__.py
      functions.py
      printing.py
      sympy_compat.py
      test_utilities.py
      utilities.py
    DESTINATION
      ${PY_PATH}
)
endif ()
