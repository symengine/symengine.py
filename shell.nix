let
  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs { };
  nsymengine = pkgs.callPackage ./nix/nsymengine.nix { };
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix/";
    ref = "refs/tags/3.1.1";
  }) {
    pkgs = pkgs;

    # optionally specify the python version
    # python = "python38";

    # optionally update pypi data revision from https://github.com/DavHau/pypi-deps-db
    # pypiDataRev = "some_revision";
    # pypiDataSha256 = "some_sha256";
  };
  customPython = mach-nix.mkPython rec {
    requirements = ''
      cython
      numpy
      sympy
      scipy
      sphinx
      sphinx-autobuild
      sphinx-book-theme
      sphinxcontrib-apidoc
      recommonmark
      m2r2
    '';
  };
in pkgs.mkShell {
  buildInputs = with pkgs; [
    customPython
    cmake
    nsymengine
    bashInteractive
    sage
    which
  ];
}
