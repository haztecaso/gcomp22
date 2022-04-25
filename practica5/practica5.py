#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

"""
Autor: Adrián Lattes Grassi
"""
def g(x, y):
    return np.sqrt(1-x ** 2 - y ** 2)

def proj(x,z,z0=1,α=1):
    z0 = z*0+z0
    ε = 1e-16
    x_trans = x/(abs(z0-z)**α+ε)
    return (x_trans)

def sphere_polar(res):
    u = np.linspace(0, np.pi, res)
    v = np.linspace(0, 2*np.pi, 2*res)
    return u, v

def polar_to_cartesian(u,v):
    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))
    return x,y,z


def main():
    u, v = sphere_polar(30)
    X, Y, Z = polar_to_cartesian(u,v)

    # Esfera
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    print(X.shape,Y.shape,Z.shape)
    z0 = 1
    xp = proj(X, Z, z0, 0.5)
    yp = proj(Y, Z, z0, 0.5)

    # Proyección estereográfica
    # ax.plot_surface(xp, yp, Z*0+z0, rstride=1, cstride=1,
    #                 cmap='viridis', alpha=0.5, edgecolor='purple')
    # ax.set_xlim3d(-8,8)
    # ax.set_ylim3d(-8,8)
    # ax.set_zlim3d(-8,8)

    # Curva
    T = np.linspace(0.0001, 1, 200)
    u = T
    v = 2*T
    x, y, z = polar_to_cartesian(u,v)
    print(u.shape,v.shape)
    print(x.shape,y.shape,z.shape)
    return
    ax.plot(x, y, z, '-b',zorder=3)
    plt.show()


if __name__ == "__main__":
    main()
