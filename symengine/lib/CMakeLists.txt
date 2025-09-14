add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pxd
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/symengine_wrapper.in.pxd
       ${PROJECT_SOURCE_DIR}/cmake/preprocess.py
    COMMAND Python::Interpreter ${PROJECT_SOURCE_DIR}/cmake/preprocess.py
       ${CMAKE_CURRENT_SOURCE_DIR}/symengine_wrapper.in.pxd
       ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pxd
       HAVE_SYMENGINE_MPFR=${HAVE_SYMENGINE_MPFR}
       HAVE_SYMENGINE_MPC=${HAVE_SYMENGINE_MPC}
       HAVE_SYMENGINE_PIRANHA=${HAVE_SYMENGINE_PIRANHA}
       HAVE_SYMENGINE_FLINT=${HAVE_SYMENGINE_FLINT}
       HAVE_SYMENGINE_LLVM=${HAVE_SYMENGINE_LLVM}
       HAVE_SYMENGINE_LLVM_LONG_DOUBLE=${HAVE_SYMENGINE_LLVM_LONG_DOUBLE}
    COMMENT "Preprocessing symengine_wrapper.in.pxd"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pyx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/symengine_wrapper.in.pyx
       ${CMAKE_CURRENT_SOURCE_DIR}/symengine_wrapper.in.pxd
       ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pxd
       ${PROJECT_SOURCE_DIR}/cmake/preprocess.py
    COMMAND Python::Interpreter ${PROJECT_SOURCE_DIR}/cmake/preprocess.py
       ${CMAKE_CURRENT_SOURCE_DIR}/symengine_wrapper.in.pyx
       ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pyx
       HAVE_SYMENGINE_MPFR=${HAVE_SYMENGINE_MPFR}
       HAVE_SYMENGINE_MPC=${HAVE_SYMENGINE_MPC}
       HAVE_SYMENGINE_PIRANHA=${HAVE_SYMENGINE_PIRANHA}
       HAVE_SYMENGINE_FLINT=${HAVE_SYMENGINE_FLINT}
       HAVE_SYMENGINE_LLVM=${HAVE_SYMENGINE_LLVM}
       HAVE_SYMENGINE_LLVM_LONG_DOUBLE=${HAVE_SYMENGINE_LLVM_LONG_DOUBLE}
    COMMENT "Preprocessing symengine_wrapper.in.pyx"
)

cython_transpile(
  ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pyx
  LANGUAGE CXX
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.cpp
  CYTHON_ARGS
    -I ${CMAKE_CURRENT_BINARY_DIR}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
)

if (NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  if ("${SKBUILD_SABI_VERSION}" VERSION_LESS "3.11")
    message(FATAL_ERROR "symengine.py with ABI3 requires python>=3.11")
  endif()
  set(USE_SABI USE_SABI ${SKBUILD_SABI_VERSION})
endif()

Python_add_library(
  symengine_wrapper
  MODULE
  WITH_SOABI
  ${USE_SABI}
  ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.cpp
  pywrapper.cpp
)
target_link_libraries(
  symengine_wrapper
  PRIVATE
    ${SYMENGINE_LIBRARIES}
)
target_include_directories(
  symengine_wrapper
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${SYMENGINE_INCLUDE_DIRS}
)

if (CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
  # Must suppress strict aliasing for this file
  set_source_files_properties(
    symengine_wrapper.cpp
    PROPERTIES
      COMPILE_FLAGS "-fno-strict-aliasing -Wno-unused-function"
  )
endif()

if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}"
   AND DEFINED SKBUILD)
  # Editable in-place builds. THe empty generator expression ensures
  # multi-config enerators keeps us from having to set
  # LIBRARY_OUTPUT_DIRECTORY_<CONFIG> too.
  set_target_properties(
    symengine_wrapper
    PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}$<0:>"
  )
endif()

install(
  TARGETS
    symengine_wrapper
  RUNTIME DESTINATION symengine/lib
  ARCHIVE DESTINATION symengine/lib
  LIBRARY DESTINATION symengine/lib
)

install(
  FILES
    __init__.py
    symengine.pxd
    symengine_wrapper.pxd
    pywrapper.h
  DESTINATION symengine/lib
)
