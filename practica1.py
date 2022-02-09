#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from typing import Callable
import time
from math import inf

DEFAULT_EPSILON   = 1e-10
DEFAULT_N_ULTIMOS = 20
DEFAULT_MAX_ITERS  = inf

class DemasiadasIteraciones(Exception):
    def __init__(self, max_iters):
        self.max_iters = max_iters
        self.message = f"Se ha sobrepasado el máximo de iteraciones ({self.max_iters})"
        super().__init__(self.message)

def logistica(r:float) -> Callable[[float], float]:
    """
    Función logística parametrizada por el parámetro r.
    Dado un parámetro r de tipo float devuelve la función logística
    correspondiente de tipo float -> float.

    :param float r: Parámetro r de la función logística
    :return: Función logística de tipo float -> float
    """
    return lambda x: r*x*(1-x)

def orbita(r:float, x0:float, N:int):
    """
    :param float r: parámetro r de la función logística
    :param float x0: valor inicial de la órbita
    :param int N: número de iteraciones
    :return: Órbita de longitud N (array de numpy)
    """
    f = logistica(r)
    o = np.empty((N,))
    o[0] = x0
    for i in range(1, N):
        o[i] = f(o[i-1])
    return o

def orbita_gen(r:float, x0:float):
    """
    Versión perezosa de la función orbita, utilizando generadores de python

    :param float r: Parámetro r de la función logística
    :param float x0: Valor inicial de la órbita
    """
    f = logistica(r)
    x = x0
    yield x
    while True:
        x = f(x)
        yield x

def periodo(orbita, epsilon = DEFAULT_EPSILON, N = DEFAULT_N_ULTIMOS):
    """
    Calcula el periodo de una órbita, considerando solo los últimos valores.

    :param orbita: Array con la órbita (generado con la función orbita)
    :param int n: Número de valores que considerar para calcular el periodo
    :param float epsilon: Precisión 
    """
    assert N <= len(orbita), f"No se pueden seleccionar {n} valores de una órbita de longitud {len(orbita)}"
    suborbita = orbita[range(-N, 0, 1)]
    for i in range(2, N-1, 1):
        if abs(suborbita[N-1] - suborbita[N-i]) < epsilon:
            return i - 1


def periodo_gen(r:float, x0:float = 0.5, epsilon = DEFAULT_EPSILON, N =
        DEFAULT_N_ULTIMOS, max_iters = DEFAULT_MAX_ITERS):
    """
    Versión perezosa de la función periodo. En vez de calcular previamente la
    órbita, la va calculando poco a poco hasta que se alcance la precisión
    deseada.

    :param float r: Parámetro r de la función logística
    :param float x0: Valor inicial de la órbita
    :param float epsilon: Precisión 
    :param int N: Número de valores que considerar para calcular el periodo
    :param int max_iter: Número máximo de iteraciones
    """
    ultimos = np.empty((N,))
    orbita = orbita_gen(r, x0)
    iters = 0
    for _ in range(N): # Llenado inicial del array ultimos
        ultimos = np.roll(ultimos, -1)
        ultimos[-1] = next(orbita)
        iters +=1
    while True: # Búsqueda de periodos
        if iters > max_iters:
            raise DemasiadasIteraciones(max_iters)
        ultimos = np.roll(ultimos, -1)
        ultimos[-1] = next(orbita)
        for p in range(1, N): # En cada iteración se comprueban todos los posibles periodos
            if abs(ultimos[-1] - ultimos[N-p-1]) < epsilon:
                return p 
        iters +=1


def main():
    timeA = time.time()
    print(f"{periodo(orbita(3.45, .5, 20000)) = }")
    timeB = time.time()
    print(f"{periodo_gen(3.45)  = }")
    timeC = time.time()
    print(f"{timeB-timeA = }")
    print(f"{timeC-timeB = }")
    # plt.plot(o[range(N-100,N)])
    # plt.show()

if __name__ == "__main__":
    main()
