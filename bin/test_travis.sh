#!/usr/bin/env bash

# Exit on error
set -e
# Echo each command
set -x

# Build inplace so that nosetests can be run inside source directory
pip install -U pip
pip install -U cython scikit-build
pip install -vvv -e .

# Test python wrappers
nosetests -v
# If switching to py.test, the following collects correct tests:
#py.test -v $PYTHON_SOURCE_DIR/symengine/tests/test_*.py
mkdir -p empty && cd empty
python $PYTHON_SOURCE_DIR/bin/test_python.py
cd ..

if [[ "${TRIGGER_FEEDSTOCK}" == "yes" ]]; then
    cd $PYTHON_SOURCE_DIR
    ./bin/trigger_feedstock.sh
fi

