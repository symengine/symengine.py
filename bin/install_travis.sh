#!/usr/bin/env bash

if [[ "${WITH_SAGE}" != "yes" ]]; then
    # symengine's bin/install_travis.sh will install miniconda
    conda_pkgs="python=${PYTHON_VERSION} pip cython sympy nose pytest"
    if [[ "${WITH_NUMPY}" == "yes" ]]; then
        conda_pkgs="${conda_pkgs} numpy";
    fi
else
    wget -O- https://dl.dropboxusercontent.com/u/46807346/sage-6.9-x86_64-Linux-Ubuntu_12.04_64_bit.tar.gz | tar xz
    SAGE_ROOT=`pwd`/sage-6.9-x86_64-Linux
    export PATH="$SAGE_ROOT:$PATH"
    source $SAGE_ROOT/src/bin/sage-env
    export our_install_dir=$SAGE_LOCAL
fi
