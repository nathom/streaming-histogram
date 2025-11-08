{
  description = "Streaming histogram PyO3 library with Python bindings";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;
        python = pkgs.python312;
        nativeLibPath = lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ];
      in {
        packages.default = pkgs.maturinBuild {
          pname = "streaming-histogram";
          version = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project.version;
          src = ./.;
          python = python;
          cargoExtraArgs = "--locked";
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python
            python312Packages.pytest
            nodejs_22
            uv
            maturin
            rustc
            cargo
            rustfmt
            clippy
            pkg-config
            openssl
            gcc
          ];

          LD_LIBRARY_PATH = nativeLibPath;

          shellHook = ''
            echo "Streaming Histogram devshell loaded"
            echo "Create a fresh environment with: uv venv --python 3.12 .venv"
            echo "Then install dev deps via: uv pip install --python .venv/bin/python maturin pytest"
          '';
        };

        formatter = pkgs.nixfmt-rfc-style;
      });
}
