execute_process(
	COMMAND ${PYTHON_BIN} -c "import numpy; print(numpy.get_include())"
        RESULT_VARIABLE NUMPY_FIND_RESULT
	OUTPUT_VARIABLE NUMPY_FIND_OUTPUT
        ERROR_VARIABLE NUMPY_FIND_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
if(NOT NUMPY_FIND_RESULT MATCHES 0)
  set(NUMPY_FOUND FALSE)
  message(STATUS "NumPy import failure:\n${NUMPY_FIND_ERROR}")  
else()
  find_path(NUMPY_INCLUDE_DIR numpy/arrayobject.h
    HINTS "${__numpy_path}" "${NUMPY_FIND_OUTPUT}" NO_DEFAULT_PATH)
  if(NUMPY_INCLUDE_DIR)
    set(NUMPY_FOUND TRUE CACHE BOOL INTERNAL "NumPy found")
    INCLUDE(FindPackageHandleStandardArgs)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(NUMPY DEFAULT_MSG NUMPY_INCLUDE_PATH)
  endif()
endif()
