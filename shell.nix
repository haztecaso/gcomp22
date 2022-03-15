{ pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  name = "dev-env";
  nativeBuildInputs = with pkgs; with pkgs.python38Packages; [
    python38Full
    numpy
    matplotlib
    ipython
    imagemagick
    graphviz
    scikit-learn
  ];
   shellHook = ''
     alias python=ipython
  '';
}
