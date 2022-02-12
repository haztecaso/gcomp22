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
    logging.info(f"Calculando órbita\t\t\t {(r, x0, N) = }")
    f = logistica(r)
    orb = np.empty((N,))
    orb[0] = x0
    logging.debug(f"{orb[0] = }")
    for n in range(1, N):
        orb[n] = f(orb[n-1])
        logging.debug(f"{n = };{orb[n] = }")
    return orb


def periodo(orbita:np.ndarray, ε:float = DEFAULT_ε):
    """
    Calcula el periodo de una órbita dada una tolerancia.

    :param orbita: Array con la órbita (generado con la función orbita)
    :param float ε: Precisión 
    """
    logging.info(f"Estimando el periodo de una órbita\t {ε = }")
    assert N_ULT <= len(orbita), f"No se pueden seleccionar {N_ULT} valores de una órbita de longitud {len(orbita)}"
    ultimos = orbita[range(-N_ULT, 0, 1)]
    for p in range(2, N_ULT-1, 1):
        logging.debug(f"{p = }; {abs(ultimos[-1] - ultimos[N_ULT-p-1]) = }")
        if abs(ultimos[N_ULT-1] - ultimos[N_ULT-p]) < ε:
            return p - 1

def atractor(orbita:np.ndarray, ε:float = DEFAULT_ε):
    """
    Estima el conjunto atractor de una órbita concreta.

    :param orbita: Array con la órbita (generado con la función orbita)
    :param float ε: Precisión 
    """
    logging.info(f"Estimando el conjunto atractor\t\t {ε = }")
    per = periodo(orbita, ε)
    logging.debug(f"{per = }")
    result = np.sort([orbita[-1-i] for i in range(per)]) if per else None
    logging.debug(f"{per = }")
    return result


def orbita_atractor_plot(r:float, x0:float, N:int, ε:float = DEFAULT_ε):
    """
    Gráfico de una órbita y el conjunto atractor correspondiente

    :param float r: parámetro r de la función logística
    :param float x0: valor inicial de la órbita
    :param int N: número de iteraciones
    :param float ε: Precisión 
    """
    orb = orbita(r, x0, N)
    atr = atractor(orb, ε)
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
