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
    netcdf4
    nodejs
    yarn
  ];
   shellHook = ''
     alias python=ipython
  '';
}
