#!/usr/bin/env bash

# Exit on error
set -e
# Echo each command
set -x

source bin/test_travis.sh

# Build and install python wrappers
if [[ "${WITH_PYTHON}" == "yes" ]]; then
    cd $PYTHON_SOURCE_DIR
    python setup.py install --symengine-dir=$our_install_dir
fi
# Test python wrappers
if [[ "${WITH_PYTHON}" == "yes" ]] && [[ "${WITH_SAGE}" != "yes" ]]; then
    nosetests -v
    mkdir -p empty
    cd empty
    python $PYTHON_SOURCE_DIR/bin/test_python.py
fi
if [[ "${WITH_SAGE}" == "yes" ]]; then
    sage -t $PYTHON_SOURCE_DIR/symengine/tests/test_sage.py
fi
