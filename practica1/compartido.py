#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable

# Valores por defecto de los parámetros
DEFAULT_EPSILON   = 1e-6

# Variables globales
MAX_ITERS  = int(1e7) # Máximo de iteraciones al calcular las órbitas
N_ULT      = 20 # Número de valores que considerar para calcular el periodo

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
