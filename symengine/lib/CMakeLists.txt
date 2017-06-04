
set(SRC pywrapper.cpp)

configure_file(config.pxi.in config.pxi)

include_directories(BEFORE ${python_wrapper_BINARY_DIR}/symengine/lib)
add_subdirectory(symengine)

add_cython_target(symengine_wrapper CXX OUTPUT_VAR symengine_wrapper_src)
add_library(symengine_wrapper MODULE ${symengine_wrapper_src} ${SRC})
target_link_libraries(symengine_wrapper ${SYMENGINE_LIBRARIES})
python_extension_module(symengine_wrapper)

if (CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
  # Must suppress strict aliasing for this file
  set_source_files_properties(
    symengine_wrapper.cpp
    PROPERTIES COMPILE_FLAGS "-fno-strict-aliasing -Wno-unused-function")
endif()

set(symengine_pkg ${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/symengine)

install(TARGETS symengine_wrapper
            RUNTIME DESTINATION ${symengine_pkg}/lib
            ARCHIVE DESTINATION ${symengine_pkg}/lib
            LIBRARY DESTINATION ${symengine_pkg}/lib)

install(FILES __init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/config.pxi
    symengine.pxd
    symengine_wrapper.pxd
    DESTINATION ${symengine_pkg}/lib)
