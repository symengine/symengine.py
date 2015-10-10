#!/usr/bin/env bash

# Exit on error
set -e
# Echo each command
set -x

# Build and install python wrappers
python setup.py install --symengine-dir=$our_install_dir

# Test python wrappers
if [[ "${WITH_SAGE}" != "yes" ]]; then
    nosetests -v
    mkdir -p empty
    cd empty
    python $PYTHON_SOURCE_DIR/bin/test_python.py
fi
if [[ "${WITH_SAGE}" == "yes" ]]; then
    sage -t $PYTHON_SOURCE_DIR/symengine/tests/test_sage.py
fi
