{ stdenv
, fetchFromGitHub
, cmake
, gmp
, flint
, mpfr
, libmpc
}:

stdenv.mkDerivation rec {
  pname = "symengine";
  name = "symengine";
  version = "44eb47e3bbfa7e06718f2f65f3f41a0a9d133b70"; # From symengine-version.txt

  src = fetchFromGitHub {
    owner = "symengine";
    repo = "symengine";
    rev = "${version}";
    sha256 = "137cxk3x8vmr4p5x0knzjplir0slw0gmwhzi277si944i33781hd";
  };

  nativeBuildInputs = [ cmake ];

  buildInputs = [ gmp flint mpfr libmpc ];

  cmakeFlags = [
    "-DWITH_FLINT=ON"
    "-DINTEGER_CLASS=flint"
    "-DWITH_SYMENGINE_THREAD_SAFE=yes"
    "-DWITH_MPC=yes"
    "-DBUILD_TESTS=no"
    "-DBUILD_FOR_DISTRIBUTION=yes"
  ];

  doCheck = false;

}

# Derived from the upstream expression : https://github.com/r-ryantm/nixpkgs/blob/34730e0640710636b15338f20836165f29b3df86/pkgs/development/libraries/symengine/default.nix
