{ pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  name = "dev-env";
  nativeBuildInputs = with pkgs; with pkgs.python38Packages; [
    graphviz
    imagemagick
    ipython
    matplotlib
    netcdf4
    nodejs
    numpy
    python38Full
    scikit-learn
    scikitimage
    yarn
  ];
   shellHook = ''
     alias python=ipython
  '';
}
