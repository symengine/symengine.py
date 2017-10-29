#!/usr/bin/env bash

# symengine's bin/install_travis.sh will install miniconda
conda update -q -n root conda

export conda_pkgs="python=${PYTHON_VERSION} pip cython nose pytest"

if [[ "${WITH_SYMPY}" != "no" ]]; then
    export conda_pkgs="${conda_pkgs} sympy";
fi

if [[ "${WITH_NUMPY}" != "no" ]]; then
    export conda_pkgs="${conda_pkgs} numpy";
fi

if [[ "${WITH_SCIPY}" == "yes" ]]; then
    export conda_pkgs="${conda_pkgs} scipy";
fi

if [[ "${WITH_SAGE}" == "yes" ]]; then
    # This is split to avoid the 10 minute limit
    conda install -q sagelib
    export conda_pkgs="${conda_pkgs} sage";
fi

conda install -q ${conda_pkgs}
source activate $our_install_dir;
