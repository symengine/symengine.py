#!/usr/bin/env bash

# symengine's bin/install_travis.sh will install miniconda

export conda_pkgs="python=${PYTHON_VERSION} pip pytest setuptools gmp mpfr"

if [[ "${WITH_NUMPY}" != "no" ]]; then
    export conda_pkgs="${conda_pkgs} numpy";
fi

if [[ "${WITH_SCIPY}" == "yes" ]]; then
    export conda_pkgs="${conda_pkgs} scipy";
fi

if [[ "${WITH_DOCS}" == "yes" ]]; then
    export conda_pkgs="${conda_pkgs} sphinx recommonmark";
fi

if [[ "${WITH_FLINT_PY}" == "yes" ]]; then
    export conda_pkgs="${conda_pkgs} python-flint";  # python-flint affects sympy, see e.g. sympy/sympy#26645
fi

if [[ "${SYMENGINE_PY_LIMITED_API}" != "" ]]; then
    export conda_pkgs="${conda_pkgs} abi3audit"
fi

if [[ "${WITH_SAGE}" == "yes" ]]; then
    # This is split to avoid the 10 minute limit
    conda install -q sagelib=8.1
    conda clean --all
    export conda_pkgs="${conda_pkgs} sage=8.1";
fi

conda install -q ${conda_pkgs} "cython>=0.29.24"

if [[ "${WITH_SYMPY}" != "no" ]]; then
    pip install sympy;
fi

conda clean --all
