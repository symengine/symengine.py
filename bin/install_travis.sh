#!/usr/bin/env bash

# symengine's bin/install_travis.sh will install miniconda
export conda_pkgs="python=${PYTHON_VERSION} pip cython sympy nose pytest"

if [[ "${WITH_NUMPY}" != "no" ]]; then
    export conda_pkgs="${conda_pkgs} numpy";
fi

if [[ "${WITH_SAGE}" == "yes" ]]; then
    export conda_pkgs="${conda_pkgs} sage";
fi

conda update -q -n root conda
conda install ${conda_pkgs} libgap=4.8.3
source activate $our_install_dir;
