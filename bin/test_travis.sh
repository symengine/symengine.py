#!/usr/bin/env bash

# Exit on error
set -e
# Echo each command
set -x

# Build inplace so that nosetests can be run inside source directory
python setup.py install build_ext --inplace --symengine-dir=$our_install_dir

# Test python wrappers
if [[ "${WITH_SAGE}" != "yes" ]]; then
    nosetests -v
    # If switching to py.test, the following collects correct tests:
    #py.test -v $PYTHON_SOURCE_DIR/symengine/tests/test_*.py
    mkdir -p empty
    cd empty
    python $PYTHON_SOURCE_DIR/bin/test_python.py
fi
if [[ "${WITH_SAGE}" == "yes" ]]; then
    sage -t $PYTHON_SOURCE_DIR/symengine/tests/test_sage.py
fi
if [[ "${WITH_COVERAGE}" == "yes" ]]; then
    bash <(curl -s https://codecov.io/bash) -x $GCOV_EXECUTABLE
    exit 0;
fi
if [[ "${TRIGGER_FEEDSTOCK}" == "yes" ]]; then
    cd $PYTHON_SOURCE_DIR
    ./bin/trigger_feedstock.sh
fi

