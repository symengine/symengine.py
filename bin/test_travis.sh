#!/usr/bin/env bash

# Exit on error
set -e
# Echo each command
set -x

python -m build . --sdist
mkdir dist-extract
cd dist-extract
tar -xvf ../dist/symengine-*.tar.gz
cd symengine-*

# Build inplace
if [[ "${SYMENGINE_PY_LIMITED_API:-}" == "" ]]; then
  python3 -m pip install -e . -vv -Ccmake.define.SymEngine_DIR=$our_install_dir
else
  python3 -m pip install -e . -vv -Ccmake.define.SymEngine_DIR=$our_install_dir -Cwheel.py-api="cp${SYMENGINE_PY_LIMITED_API/./}"
  python3 -m abi3audit --assume-minimum-abi3 ${SYMENGINE_PY_LIMITED_API} symengine/lib/symengine_wrapper.abi3.so -v
fi

# Test python wrappers
python3 -m pip install pytest
python3 -m pytest -s -v $PWD/symengine/tests/test_*.py
mkdir -p empty && cd empty
python3 $PYTHON_SOURCE_DIR/bin/test_python.py
