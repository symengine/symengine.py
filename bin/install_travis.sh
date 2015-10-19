#!/usr/bin/env bash

if [[ "${WITH_SAGE}" != "yes" ]]; then
    if [[ "${TRAVIS_OS_NAME}" != "osx" ]]; then
        wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh;
    else
        wget https://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh -O miniconda.sh;
    fi
    bash miniconda.sh -b -p $our_install_dir/miniconda;
    export PATH="$our_install_dir/miniconda/bin:$PATH";
    hash -r;
    conda config --set always_yes yes --set changeps1 no;
    conda update -q conda;
    conda info -a;

    conda create -q -n test-environment python="${PYTHON_VERSION}" pip cython sympy nose pytest;
    source activate test-environment;
else
    wget -O- http://files.sagemath.org/linux/64bit/sage-6.9-x86_64-Linux-Ubuntu_12.04_64_bit.tar.lrz | lrzip -dq | tar x
    SAGE_ROOT=`pwd`/sage-6.9-x86_64-Linux
    export PATH="$SAGE_ROOT:$PATH"
    source $SAGE_ROOT/src/bin/sage-env
    export our_install_dir=$SAGE_LOCAL
fi