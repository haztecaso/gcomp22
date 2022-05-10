#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# @author: Adrián Lattes Grassi

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy import pi, cos, sin

def sphere_polar(res):
    u = np.linspace(0, pi, res)
    v = np.linspace(0, 2*pi, 2*res)
    return u, v

def polar_to_cartesian(u, v, outer = True):
    prod = np.outer if outer else np.multiply
    x = prod(sin(u), sin(v))
    y = prod(sin(u), cos(v))
    z = prod(cos(u), np.ones_like(v))
    return x, y, z

def proj(x, z, z0 = 1, α = 1):
    z0 = z*0+z0
    ε = 1e-16
    x_trans = x/(abs(z0-z)**α+ε)
    return (x_trans)


def main():
    u, v = sphere_polar(30)
    X, Y, Z = polar_to_cartesian(u, v)

    gs  = gridspec.GridSpec(1, 2)
    fig = plt.figure()

    # Esfera
    ax = fig.add_subplot(gs[0,0], projection='3d')
    ax.set_title('2-esfera y curva.');
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    # Curva (parametrización en coordenadas polares)
    T = np.linspace(0, 2*pi, 1000)
    u = pi/2+pi/20*cos(T*20)
    v = T

    # Paso a coordenadas cartesianas 
    x, y, z = polar_to_cartesian(u, v, outer=False)
    ax.plot(x, y, z, '-b', zorder=3)

    ax = fig.add_subplot(gs[0,1], projection='3d')
    ax.set_title('Proyección estereográfica e imagen de la curva.');

    # Proyección estereográfica de la esfera
    z0, α = 1, 0.5

    Xp = proj(X, Z, z0, α)
    Yp = proj(Y, Z, z0, α)
    ax.plot_surface(Xp, Yp, Z*0+z0, rstride=1, cstride=1, cmap='viridis', alpha=0.5, edgecolor='purple')

    # Imagen de la curva dibujada anteriormente
    xp = proj(x, z, z0, α)
    yp = proj(y, z, z0, α)
    ax.plot(xp, yp, z0, '-b', zorder=3)

    plt.show()


if __name__ == "__main__":
    main()
