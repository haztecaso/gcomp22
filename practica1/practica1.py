#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


from compartido import *

def orbita(r:float, x0:float, N:int):
    """
    :param float r: parámetro r de la función logística
    :param float x0: valor inicial de la órbita
    :param int N: número de iteraciones
    :return: Órbita de longitud N (array de numpy)
    """
    assert N < MAX_ITERS, "Demasiadas iteraciones"
    f = logistica(r)
    orb = np.empty((N,))
    orb[0] = x0
    for i in range(1, N):
        orb[i] = f(orb[i-1])
    return orb


def periodo(orbita:np.ndarray, epsilon:float = DEFAULT_EPSILON):
    """
    Calcula el periodo de una órbita dada una tolerancia.

    :param orbita: Array con la órbita (generado con la función orbita)
    :param float epsilon: Precisión 
    """
    assert N_ULT <= len(orbita), f"No se pueden seleccionar {N_ULT} valores de una órbita de longitud {len(orbita)}"
    suborbita = orbita[range(-N_ULT, 0, 1)]
    for i in range(2, N_ULT-1, 1):
        if abs(suborbita[N_ULT-1] - suborbita[N_ULT-i]) < epsilon:
            return i - 1

def atractor(orbita:np.ndarray, epsilon:float = DEFAULT_EPSILON):
    """
    Estima el conjunto atractor de una órbita concreta.

    :param orbita: Array con la órbita (generado con la función orbita)
    :param float epsilon: Precisión 
    """
    per = periodo(orbita, epsilon)
    return np.sort([orbita[-1-i] for i in range(per)]) if per else None


def orbita_atractor_plot(r:float, x0:float, N:int, epsilon:float = DEFAULT_EPSILON):
    """
    Gráfico de una órbita y el conjunto atractor correspondiente

    :param float r: parámetro r de la función logística
    :param float x0: valor inicial de la órbita
    :param int N: número de iteraciones
    :param float epsilon: Precisión 
    """
    orb = orbita(r, x0, N)
    atr = atractor(orb, epsilon)
    if atr is not None:
        plt.plot(orb)
        for valor in atr:
            plt.axhline(y = valor, color = 'r', linestyle = '-')
        plt.show()
    else:
        print("Error: ¡No se ha podido encontrar un periodo (y por tanto un conjunto atractor)!")
        print("Prueba a incrementar el número de iteraciones.")


def main():
    orbita_atractor_plot(3.5, .5, 100)

if __name__ == "__main__":
    main()
