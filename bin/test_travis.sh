#!/usr/bin/env bash

# Exit on error
set -e
# Echo each command
set -x

# Build and install python wrappers
python setup.py install --symengine-dir=$our_install_dir

# Test python wrappers
if [[ "${WITH_SAGE}" != "yes" ]]; then
    mkdir -p empty
    cd empty
    py.test -v --pyargs symengine  # test "symengine.tests"
    python $PYTHON_SOURCE_DIR/bin/test_python.py
fi
if [[ "${WITH_SAGE}" == "yes" ]]; then
    sage -t $PYTHON_SOURCE_DIR/symengine/tests/test_sage.py
fi
