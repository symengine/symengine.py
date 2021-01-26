#!/usr/bin/env nix-shell
#!nix-shell ../shell.nix -i bash

python setup.py build_ext --inplace
sphinx-build docs/ genDocs/
