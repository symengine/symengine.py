cmake_minimum_required(VERSION 3.21...4.0.0)

if (POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW) # needed for llvm >= 16
endif ()
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)  # allow user to set *_ROOT variables
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

set(CMAKE_PREFIX_PATH ${SymEngine_DIR} ${CMAKE_PREFIX_PATH})

include(GNUInstallDirs)

find_package(SymEngine 0.14.0 REQUIRED CONFIG
             PATH_SUFFIXES lib/cmake/symengine cmake/symengine CMake/)
message("SymEngine_DIR : " ${SymEngine_DIR})
message("SymEngine Version : " ${SymEngine_VERSION})
set(CMAKE_BUILD_TYPE ${SYMENGINE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS_RELEASE ${SYMENGINE_CXX_FLAGS_RELEASE})
set(CMAKE_CXX_FLAGS_DEBUG ${SYMENGINE_CXX_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYMENGINE_CXX_FLAGS}")

find_package(
  Python
  COMPONENTS
    Interpreter
    Development.Module
    ${SKBUILD_SABI_COMPONENT}
  REQUIRED
)

if ("${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(CYTHON_MIN "0.29")
else()
  set(CYTHON_MIN "3.1")
endif()

find_package(Cython ${CYTHON_MIN} MODULE REQUIRED)
include(UseCython)

if ("${CYTHON_VERSION}" VERSION_LESS "${CYTHON_MIN}")
  message(FATAL_ERROR "Cython version found: ${CYTHON_VERSION}. Minimum required: ${CYTHON_MIN}")
endif()

if (MINGW AND ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8"))
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DMS_WIN64")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DMS_WIN64")
endif()

if (MINGW AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    try_compile(CHECK_PYTHON_HYPOT "${CMAKE_CURRENT_BINARY_DIR}/" "${CMAKE_SOURCE_DIR}/cmake/check_python_hypot.cpp")
    if (NOT ${CHECK_PYTHON_HYPOT})
        # include cmath before all headers
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -include cmath")
    endif()
endif()

include(CheckTypeSize)
check_type_size("long double" SYMENGINE_SIZEOF_LONG_DOUBLE)

if (HAVE_SYMENGINE_LLVM AND  SYMENGINE_SIZEOF_LONG_DOUBLE GREATER "8" AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set (HAVE_SYMENGINE_LLVM_LONG_DOUBLE True)
else ()
    set (HAVE_SYMENGINE_LLVM_LONG_DOUBLE False)
endif ()

foreach (PKG MPC MPFR PIRANHA FLINT LLVM)
    if ("${HAVE_SYMENGINE_${PKG}}" STREQUAL "")
        set(HAVE_SYMENGINE_${PKG} False)
    endif()
endforeach()

message("CMAKE_SYSTEM_PROCESSOR              : ${CMAKE_SYSTEM_PROCESSOR}")
message("CMAKE_BUILD_TYPE                    : ${CMAKE_BUILD_TYPE}")
message("CMAKE_CXX_FLAGS                     : ${CMAKE_CXX_FLAGS}")
message("CMAKE_CXX_FLAGS_RELEASE             : ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_CXX_FLAGS_DEBUG               : ${CMAKE_CXX_FLAGS_DEBUG}")
message("HAVE_SYMENGINE_MPFR                 : ${HAVE_SYMENGINE_MPFR}")
message("HAVE_SYMENGINE_MPC                  : ${HAVE_SYMENGINE_MPC}")
message("HAVE_SYMENGINE_PIRANHA              : ${HAVE_SYMENGINE_PIRANHA}")
message("HAVE_SYMENGINE_FLINT                : ${HAVE_SYMENGINE_FLINT}")
message("HAVE_SYMENGINE_LLVM                 : ${HAVE_SYMENGINE_LLVM}")
message("HAVE_SYMENGINE_LLVM_LONG_DOUBLE     : ${HAVE_SYMENGINE_LLVM_LONG_DOUBLE}")

add_subdirectory(symengine)
