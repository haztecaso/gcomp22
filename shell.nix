{ pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  name = "dev-env";
  nativeBuildInputs = with pkgs.python38Packages; [
    pkgs.python38Full
    numpy
    matplotlib
  ];
  # shellHook = ''
  # '';
}
