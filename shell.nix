{ pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  name = "dev-env";
  nativeBuildInputs = with pkgs.python38Packages; with pkgs; [
    python38Full
    numpy
    matplotlib
    ipython
    imagemagick
  ];
   shellHook = ''
     alias python=ipython
  '';
}
