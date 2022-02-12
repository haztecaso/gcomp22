#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import logging

import matplotlib.pyplot as plt
import numpy as np

from compartido import *

def orbita(r:float, x0:float):
    """
    Versión perezosa de la función orbita, utilizando generadores de python

    :param float r: Parámetro r de la función logística
    :param float x0: Valor inicial de la órbita
    """
    logging.info(f"Calculando órbita\t\t\t {(r, x0)          = }")
    f = logistica(r)
    x = x0
    n = 0
    logging.debug(f"x_{n}: {x}")
    yield x
    while True:
        n += 1
        x = f(x)
        logging.debug(f"x_{n}: {x}")
        yield x

def periodo(r:float, x0:float, ε = DEFAULT_ε, valores:List[float]=None):
    """
    Calcula el periodo de una órbita dada una precisión.

    :param float r: Parámetro r de la función logística
    :param float x0: Valor inicial de la órbita
    :param float ε: Precisión 
    :param list valores: Lista donde guardar los valores de la órbita. Parámetro opcional.
    """
    logging.info(f"Estimando el periodo de una órbita\t {(r, x0, ε) = }")
    ultimos = np.empty((N_ULT,))
    orb = orbita(r, x0)
    iters = 0
    logging.debug(f"Llenado inicial del array ultimos ({N_ULT = })")
    for _ in range(N_ULT):
        ultimos = np.roll(ultimos, -1)
        valor = next(orb)
        if valores is not None: valores.append(valor)
        ultimos[-1] = valor
        iters +=1
    for valor in orb: # Búsqueda de periodos
        if iters > MAX_ITERS: raise DemasiadasIteraciones(MAX_ITERS)
        if valores is not None: valores.append(valor)
        ultimos = np.roll(ultimos, -1)
        ultimos[-1] = valor
        for p in range(1, N_ULT): # En cada iteración se comprueban todos los posibles periodos
            logging.debug(f"{p = }; {abs(ultimos[-1] - ultimos[N_ULT-p-1]) = }")
            if abs(ultimos[-1] - ultimos[N_ULT-p-1]) < ε:
                return p
        iters +=1

def atractor(r:float, x0:float, ε:float = DEFAULT_ε, valores:List[float] = None):
    """
    Estima el conjunto atractor de una órbita.

    :param float r: Parámetro r de la función logística
    :param float x0: Valor inicial de la órbita
    :param float ε: Precisión 
    :param list valores: Lista donde guardar los valores de la órbita. Parámetro opcional.
    """
    logging.info(f"Estimando el conjunto atractor\t\t {(r, x0, ε) = }")
    if valores is None:
        valores = []
    per = periodo(r, x0, ε, valores)
    result = np.sort([valores[-1-i] for i in range(per)]) if per else None
    logging.debug(f"Conjunto atractor estimado: {result}")
    return result

def orbita_atractor_plot(r:float, x0:float, ε:float = DEFAULT_ε):
    """
    Gráfico de una órbita y el conjunto atractor correspondiente

    :param float r: Parámetro r de la función logística
    :param float x0: Valor inicial de la órbita
    :param float ε: Precisión 
    """
    orb = []
    atr = atractor(r, x0, ε, orb)
    if atr is not None:
        plt.plot(orb)
        for valor in atr:
            plt.axhline(y = valor, color = 'r', linestyle = '-')
        # plt.show()
    else:
        print("Error: ¡No se ha podido encontrar un periodo (y por tanto un conjunto atractor)!")
        print("Prueba a incrementar el número de iteraciones.")

def main():
    orbita_atractor_plot(3.5, .5, 1e-10)

if __name__ == "__main__":
    main()
