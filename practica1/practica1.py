#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import logging
from typing import Callable

# Valores por defecto de los parámetros
DEFAULT_ε   = 1e-4

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
        result = True
        for j in range(0,p):
            result = result and abs(ultimos[N_ULT-1-j] - ultimos[N_ULT-p-j]) < ε
        if result:
            return p-1
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

def estimar_errores_atractor(orb:np.ndarray, per:int):
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
        errs.append(abs(orb[-1-i] - orb[-1-i-per]))
    return errs

def estimar_error_atractor(orb:np.ndarray, per:int):
    """
    Dada una órbita y un periodo devuelve el error estimado (el mayor de todos)
    de los puntos del atractor correspondiente.
    """
    errs = estimar_errores_atractor(orb, per)
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
    plt.xlabel("r")
    if show: plt.show()

def atractores_con_periodo(p:int, rs:np.ndarray, x0:float, N:int, ε:float = DEFAULT_ε, **kwargs):
    """
    Dado un periodo fijo encuentra los valores de r, con sus atractores
    correspondientes, cuyas órbitas tienen ese periodo.
    También incluye la opción plot para dibujar los atractores obtenidos.

    :param int p: El periodo que se busca
    :param np.ndarray rs: Valores de r que testear
    :param float x0: valor inicial de las órbita
    :param int N: número de iteraciones
    :param float ε: Precisión 
    :param bool plot: Plotear la gráfica. Valor por defecto True.
    :param str fmt: Formato de la gráfica. Valor por defecto 'ro'.
    :param bool show: Pintar la gráfica. Valor por defecto True.
    """
    logging.info(f"Buscando atractores con periodo {p} en el intervalo {rs[0],rs[-1]} ({N = })")
    result_rs = []
    result_atrs = []
    plot = kwargs.get('plot', True)
    show = kwargs.get('show', True)
    fmt  = kwargs.get('fmt', 'ro')
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
                    if plot: plt.plot(r, atr[i], fmt, markersize=1)
        except PeriodoNoEncontrado:
            pass
    if plot and show: plt.show()
    return result_rs, result_atrs

def apartado1():
    """
    Ejemplo de conjuntos atractores con sus correspondientes intervalos de error.
    """
    x0, N, ε =  0.1, 100, 1e-4

    r1 = 3.0241
    plt.subplot(2, 2, 1)
    orb1, per1, atr1 = orbita_atractor_plot(r1, x0, N, ε, show = False)
    errs1 = estimar_errores_atractor(orb1, per1)
    print(f"Estimación del atractor para r = {r1} con N = {N} y ε = {ε}:")
    print(f"- Periodo estimado: {per1}")
    print(f"- Puntos del atractor (no están escritos en notación estándar, ver memoria):")
    for i in range(len(atr1)):
        print(f"  - x_{N-1-i} = {atr1[i]} ±{errs1[i]}")
    print("")

    r2 = 3.4657
    plt.subplot(2, 2, 2)
    orb2, per2, atr2 = orbita_atractor_plot(r2, x0, N, ε, show = False)
    errs2 = estimar_errores_atractor(orb2, per2)
    print(f"Estimación del atractor para r = {r2} con N = {N} y ε = {ε}:")
    print(f"- Periodo estimado: {per2}")
    print(f"- Puntos del atractor (no están escritos en notación estándar, ver memoria):")
    for i in range(len(atr2)):
        print(f"  - x_{N-i} = {atr2[i]} ±{errs2[i]}")
    plt.subplot(2, 1, 2)
    plt.axvline(x = r1, color = 'g', linestyle = '-')
    plt.axvline(x = r2, color = 'g', linestyle = '-')

    # Gráfico de estimaciones del atractor en todo un intervalo
    conjunto_atractor_plot(np.linspace(2.95,3.544,1000), x0, 400, 1e-4)

def apartado2():
    """
    Estimación de valores de r en un intervalo para los que la órbita tiene periodo 8.
    """
    p, x0, N, ε = 8, 0.5, 1000, 1e-5
    M = 1000        # Número de r's que considerar
    a = 3.544       # Extremo inferior del intervalo de las r's
    b = 4.0         # Extremo superior del intervalo de las r's
    delta = (b-a)/(M-1) # Tamaño de los subintervalos en los que hemos dividido
    rs = np.linspace(a, b, M)
    rsp, _ = atractores_con_periodo(p, rs, x0, N, ε, plot = True, show = False)
    print(f"Se han obtenido {len(rsp)} valores de r con periodo {p}:")
    for r in rsp:
        print(f"- {r}±{delta}")
    plt.show()

def main():
    apartado1()
    apartado2()

if __name__ == "__main__":
    # Configuración del logger. Se puede cambiar el nivel del logger para debuguear o no imprimir ningún mensaje.
    logging.basicConfig(level=logging.WARN, format='%(message)s')
    logging.getLogger('matplotlib').setLevel(logging.WARN)
    main()
