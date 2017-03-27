#!/usr/bin/env bash

# symengine's bin/install_travis.sh will install miniconda
export conda_pkgs="python=${PYTHON_VERSION} pip cython sympy nose pytest"

if [[ "${WITH_NUMPY}" == "yes" ]]; then
    export conda_pkgs="${conda_pkgs} numpy";
fi

if [[ "${WITH_SAGE}" == "yes" ]]; then
    export conda_pkgs="${conda_pkgs} sage";
fi
