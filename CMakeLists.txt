cmake_minimum_required(VERSION 3.6)
project(python_wrapper)

# If running directly under scikit-build, set up the superbuild.
if(SKBUILD)
    include(ExternalProject)

    file(STRINGS symengine_version.txt SYMENGINE_GIT_TAG LIMIT_COUNT 1)

    ExternalProject_add(
        symengine
        SOURCE_DIR ${CMAKE_BINARY_DIR}/symengine-src
        BINARY_DIR ${CMAKE_BINARY_DIR}/symengine-bin
        CMAKE_CACHE_ARGS
          -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/symengine-install
        GIT_REPOSITORY git://github.com/symengine/symengine.git
        GIT_TAG ${SYMENGINE_GIT_TAG})

    set(symengine_install_dir
        ${CMAKE_BINARY_DIR}/symengine-install)
    set(symengine_wrapper_install_dir
        ${CMAKE_BINARY_DIR}/symengine-wrapper-install)

    ExternalProject_add(
        symengine_wrapper
        DEPENDS symengine
        SOURCE_DIR ${CMAKE_SOURCE_DIR}
        BINARY_DIR ${CMAKE_BINARY_DIR}/symengine-wrapper-bin
        CMAKE_CACHE_ARGS
          -DCMAKE_INSTALL_PREFIX:PATH=${symengine_wrapper_install_dir}
          -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
          -DPYTHON_VERSION_STRING:STRING=${PYTHON_VERSION_STRING}
          -DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR}
          -DPYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY}
          -DSKBUILD:BOOL=NO
          -DCMAKE_MODULE_PATH:PATH=${CMAKE_MODULE_PATH}
          -DSymEngine_DIR:PATH=${symengine_install_dir}
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND "")

    install(DIRECTORY ${symengine_wrapper_install_dir}/
            DESTINATION ${CMAKE_INSTALL_PREFIX})

    return()
endif()

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

set(CMAKE_PREFIX_PATH ${SymEngine_DIR} ${CMAKE_PREFIX_PATH})

find_package(SymEngine 0.3.0
             REQUIRED CONFIG
             PATH_SUFFIXES lib/cmake/symengine cmake/symengine CMake/)

message("SymEngine_DIR : " ${SymEngine_DIR})

set(CMAKE_BUILD_TYPE ${SYMENGINE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS_RELEASE ${SYMENGINE_CXX_FLAGS_RELEASE})
set(CMAKE_CXX_FLAGS_DEBUG ${SYMENGINE_CXX_FLAGS_DEBUG})

include_directories(${SYMENGINE_INCLUDE_DIRS})

# set(CMAKE_MODULE_PATH
#     ${CMAKE_MODULE_PATH}
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")


include_directories(${PYTHON_INCLUDE_PATH})

if (MINGW AND ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8"))
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DMS_WIN64")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DMS_WIN64")
endif()

if (MINGW AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    try_compile(CHECK_PYTHON_HYPOT
                "${CMAKE_CURRENT_BINARY_DIR}/"
                "${CMAKE_SOURCE_DIR}/cmake/check_python_hypot.cpp")

    if (NOT ${CHECK_PYTHON_HYPOT})
        # include cmath before all headers
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -include cmath")
    endif()
endif()

if(SYMENGINE_MPC_LIBRARIES)
    set(HAVE_SYMENGINE_MPC True)
else()
    set(HAVE_SYMENGINE_MPC False)
endif()

if(SYMENGINE_MPFR_LIBRARIES)
    set(HAVE_SYMENGINE_MPFR True)
else()
    set(HAVE_SYMENGINE_MPFR False)
endif()

if(SYMENGINE_PIRANHA_INCLUDE_DIRS)
    set(HAVE_SYMENGINE_PIRANHA True)
else()
    set(HAVE_SYMENGINE_PIRANHA False)
endif()

if(SYMENGINE_FLINT_LIBRARIES)
    set(HAVE_SYMENGINE_FLINT True)
else()
    set(HAVE_SYMENGINE_FLINT False)
endif()

if(SYMENGINE_LLVM_COMPONENTS)
    set(HAVE_SYMENGINE_LLVM True)
else()
    set(HAVE_SYMENGINE_LLVM False)
endif()


message("CMAKE_BUILD_TYPE        : ${CMAKE_BUILD_TYPE}")
message("CMAKE_CXX_FLAGS_RELEASE : ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_CXX_FLAGS_DEBUG   : ${CMAKE_CXX_FLAGS_DEBUG}")
message("HAVE_SYMENGINE_MPFR     : ${HAVE_SYMENGINE_MPFR}")
message("HAVE_SYMENGINE_MPC      : ${HAVE_SYMENGINE_MPC}")
message("HAVE_SYMENGINE_PIRANHA  : ${HAVE_SYMENGINE_PIRANHA}")
message("HAVE_SYMENGINE_FLINT    : ${HAVE_SYMENGINE_FLINT}")
message("HAVE_SYMENGINE_LLVM     : ${HAVE_SYMENGINE_LLVM}")

message("Copying source of python wrappers into: ${CMAKE_CURRENT_BINARY_DIR}")
file(COPY symengine/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/symengine)

add_subdirectory(symengine)
