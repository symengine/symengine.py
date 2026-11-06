# Build SymEngine at the commit in symengine_version.txt with
# ExternalProject_Add, as a static library using Boost.Multiprecision integers
# and no optional dependencies. SymEngine downloads the standalone
# Boost.Multiprecision headers itself, so nothing needs to be installed.
# Sets SYMENGINE_INCLUDE_DIRS and SYMENGINE_LIBRARIES like
# find_package(SymEngine) does.

include(ExternalProject)

if (POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)  # set extracted file timestamps to now
endif()

file(STRINGS ${PROJECT_SOURCE_DIR}/symengine_version.txt SYMENGINE_COMMIT
     LIMIT_COUNT 1)
set(SYMENGINE_URL
    "https://github.com/symengine/symengine/archive/${SYMENGINE_COMMIT}.tar.gz"
    CACHE STRING "URL or path of the SymEngine source archive to build")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(SYMENGINE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/symengine-install)
set(SYMENGINE_LIBRARY
    ${SYMENGINE_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}symengine${CMAKE_STATIC_LIBRARY_SUFFIX})

ExternalProject_Add(
  symengine_external
  URL ${SYMENGINE_URL}
  INSTALL_DIR ${SYMENGINE_INSTALL_DIR}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR:STRING=lib
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_OSX_ARCHITECTURES:STRING=${CMAKE_OSX_ARCHITECTURES}
    -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DBUILD_TESTS:BOOL=OFF
    -DBUILD_BENCHMARKS:BOOL=OFF
    -DINTEGER_CLASS:STRING=boostmp
    -DUSE_SYSTEM_BOOSTMP:BOOL=OFF
    -DWITH_SYMENGINE_RCP:BOOL=ON
    -DWITH_SYMENGINE_THREAD_SAFE:BOOL=ON
  BUILD_BYPRODUCTS ${SYMENGINE_LIBRARY}
)

# The include directory of an imported target must exist at configure time
file(MAKE_DIRECTORY ${SYMENGINE_INSTALL_DIR}/include)
add_library(symengine STATIC IMPORTED)
set_target_properties(symengine PROPERTIES IMPORTED_LOCATION ${SYMENGINE_LIBRARY})
add_dependencies(symengine symengine_external)

set(SYMENGINE_INCLUDE_DIRS
    ${SYMENGINE_INSTALL_DIR}/include
    ${SYMENGINE_INSTALL_DIR}/include/symengine/utilities/cereal/include
    ${SYMENGINE_INSTALL_DIR}/include/symengine/utilities/boost_multiprecision/include)
set(SYMENGINE_LIBRARIES symengine)
