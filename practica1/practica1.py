#!/usr/bin/env python3
#which -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import logging
from typing import Callable

# Valores por defecto de los parámetros
DEFAULT_ε   = 1e-6

# Variables globales
MAX_ITERS  = int(1e4) # Máximo de iteraciones al calcular las órbitas
N_ULT      = 32 # Número de valores que considerar para calcular el periodo

class PeriodoNoEncontrado(Exception):
    def __init__(self):
        self.message = f"No se ha podido encontrar un periodo."
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

def periodo(orb:np.ndarray, ε:float = DEFAULT_ε):
    """
    Calcula el periodo de una órbita.

    :param orbita: Array con la órbita (generado con la función orbita)
    :param float ε: Precisión 
    """
    logging.info(f"Estimando el periodo de una órbita\t {ε = }")
    assert N_ULT <= len(orb), f"No se pueden seleccionar {N_ULT} valores de una órbita de longitud {len(orb)}"
    ultimos = orb[range(-N_ULT, 0, 1)]
    for p in range(2, N_ULT-1, 1):
        logging.debug(f"{p = }; {abs(ultimos[-1] - ultimos[N_ULT-p-1]) = }")
        if abs(ultimos[N_ULT-1] - ultimos[N_ULT-p]) < ε:
            return p - 1
    raise PeriodoNoEncontrado()

def atractor(orb:np.ndarray, ε:float = DEFAULT_ε, per:int=None):
    """
    Estima el conjunto atractor de una órbita concreta.

    :param np.ndarray orbita: Array con la órbita (generado con la función orbita)
    :param float ε: Precisión 
    """
    logging.info(f"Estimando el conjunto atractor\t\t {ε = }")
    if per is None: per = periodo(orb, ε)
    logging.debug(f"{per = }")
    result = np.sort([orb[-1-i] for i in range(per)])
    logging.info(f"Conjunto atractor estimado: {result}")
    return result

def estimar_errores_atractor(orb:np.ndarray, per:int, ε:float = DEFAULT_ε):
    """
    Dada una órbita y un periodo devuelve los errores estimados de los puntos
    del atractor correspondiente.

    :param np.ndarray orb: Array con la órbita (generado con la función orbita)
    :param int per: Periodo estimado de la órbita
    :param float ε: Precisión 
    """
    assert len(orb) >= 2*per, "Se necesitan al menos 2*{per}={2*per} valores para estimar el intervalo de error"
    errs = []
    for i in range(per):
        errs.append(orb[-1-i] - orb[-1-i-per])
    return errs

def estimar_error_atractor(orb:np.ndarray, per:int, ε:float = DEFAULT_ε):
    """
    Dada una órbita y un periodo devuelve el error estimado (el mayor de todos)
    de los puntos del atractor correspondiente.
    """
    errs = estimar_errores_atractor(orb, per, ε)
    return max(errs)

def orbita_atractor_plot(r:float, x0:float, N:int, ε:float = DEFAULT_ε, show:bool = True):
    """
    Gráfico de una órbita y el conjunto atractor correspondiente

    :param float r: parámetro r de la función logística
    :param float x0: valor inicial de la órbita
    :param int N: número de iteraciones
    :param float ε: Precisión 
    """
    orb = orbita(r, x0, N)
    per = periodo(orb, ε) 
    atr = atractor(orb, ε, per)
    plt.ylabel("x")
    plt.xlabel("n")
    plt.plot(orb)
    for valor in atr:
        plt.axhline(y = valor, color = 'r', linestyle = '-')
    plt.title(f"Órbita y conjunto atractor\n{r, x0, N, ε = } ")
    if show: plt.show()
    return (orb, per, atr)

def conjunto_atractor_plot(rs:np.ndarray, x0:float, N:int, ε:float =DEFAULT_ε, show:bool = True):
    """
    Dibujo de un conjunto atractor para múltiples r's

    :param np.ndarray rs: Valores de r 
    :param float x0: valor inicial de las órbita
    :param int N: número de iteraciones
    :param float ε: Precisión 
    :param bool show: Pintar la gráfica
    """
    for r in rs:
        try:
            orb = orbita(r, x0, N)
            atr = atractor(orb, ε)
            for v in atr:
                plt.plot(r, v, 'ro', markersize = 1)
        except PeriodoNoEncontrado:
            print(f"Periodo no encontrado para {r, N, ε = }")
    plt.title(f"Conjunto atractor para r en ({rs[0]},{rs[-1]}), {x0 = }, {N = }, {ε = }")
    plt.ylabel("x")
    plt.xlabel("n")
    if show: plt.show()

def atractores_con_periodo(p:int, rs:np.ndarray, x0:float, N:int, ε:float = DEFAULT_ε, plot:bool = False, show:bool = True):
    """
    Dado un periodo fijo encuentra los valores de r, con sus atractores
    correspondientes, cuyas órbitas tienen ese periodo.
    También incluye la opción plot para dibujar los atractores obtenidos.

    :param int p: El periodo que se busca
    :param np.ndarray rs: Valores de r que testear
    :param float x0: valor inicial de las órbita
    :param int N: número de iteraciones
    :param float ε: Precisión 
    :param bool plot: Plotear la gráfica
    :param bool show: Pintar la gráfica
    """
    logging.info(f"Buscando atractores con periodo {p} en el intervalo {rs[0],rs[-1]} ({N = })")
    result_rs = []
    result_atrs = []
    for r in rs:
        logging.debug(f"{r = }")
        try:
            orb = orbita(r, x0, N)
            per = periodo(orb, ε)
            logging.debug(f"{per = }")
            if per == p:
                atr = atractor(orb, ε, per)
                result_rs.append(r)
                result_atrs.append(atr)
                for i in range(per):
                    if plot: plt.plot(r, atr[i], 'ro' if per % 2 == 0 else 'bo', markersize=1)
        except PeriodoNoEncontrado:
            pass
    if plot and show: plt.show()
    return result_rs, result_atrs


def apartado1(r1:float, r2:float):
    """
    Ejemplo de conjuntos atractores con sus correspondientes intervalos de error.
    """
    x0, N, ε =  0.1, 300, 1e-5

    plt.subplot(2, 2, 3)
    orb1, per1, atr1 = orbita_atractor_plot(r1, x0, N, ε, show = False)
    err1 = estimar_error_atractor(orb1, per1, ε)
    print(f"{r1, per1, atr1, err1 = }")

    plt.subplot(2, 2, 4)
    orb2, per2, atr2 = orbita_atractor_plot(r2, x0, N, ε, show = False)
    err2 = estimar_error_atractor(orb2, per2, ε)
    print(f"{r2, per2, atr2, err2 = }")
    plt.subplot(2, 1, 1)
    conjunto_atractor_plot(np.linspace(3,3.544,300), x0, 1000, ε, show = False)

def main():
    apartado1(3.01, 3.54)
    return
    p, x0, N = 8, 0.5, 100
    rs = np.linspace(3.544,4, 500)
    atractores_con_periodo(p, rs, x0, N, plot = True)

if __name__ == "__main__":
    # Configuración del logger. Se puede cambiar el nivel del logger para debuguear o no imprimir ningún mensaje.
    logging.basicConfig(level=logging.WARN, format='%(message)s')
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    main()
