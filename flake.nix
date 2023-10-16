{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.nixpkgs-staging.url = "github:NixOS/nixpkgs/staging";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, nixpkgs-staging, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        llvm = nixpkgs.legacyPackages.${system}.llvmPackages_latest; #nixpkgs-staging.legacyPackages.${system}.llvmPackages_15;
        clang-tools_16-libcxx = pkgs.clang-tools_16.override {
          llvmPackages = llvm;
          enableLibcxx = true;
        };
      in
      {
        formatter.x86_64-linux = pkgs.nixpkgs-fmt;

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            clang-tools_16-libcxx
            gcc13
            #llvm.lldb
            #llvm.libcxxClang
            #llvm.libcxx
            #llvm.libcxxabi
            nixpkgs-fmt
            cpplint
            gtest
            ninja
            cmake-format
          ];
        };
      }
    );
}
