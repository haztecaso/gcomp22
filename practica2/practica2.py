#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Autor: Adrián Lattes Grassi
"""

from typing import Dict
from decimal import Decimal
from heapq import heappop, heappush

FICHERO_MUESTRA_ES = "./GCOM2022_pract2_auxiliar_esp.txt"
FICHERO_MUESTRA_EN = "./GCOM2022_pract2_auxiliar_eng.txt"

def frecuencias(texto:str)-> Dict[str, Decimal]:
    letras = dict()
    for char in texto:
        if char not in letras:
            letras[char] = Decimal(1)
        else:
            letras[char] += 1
    n = len(texto)
    return {char:frec/n for (char, frec) in letras.items()}

class Arbol():
    def __init__(self, **kwargs):
        assert 'clave' in kwargs or ('iz' in kwargs and 'dr' in kwargs)
        if 'clave' in kwargs:
            assert 'peso' in kwargs
            self.peso = kwargs['peso']
            self.hoja = True
            self.clave = kwargs['clave']
        else:
            self.hoja = False
            self.iz = kwargs['iz']
            self.dr = kwargs['dr']
            self.peso = self.iz.peso + self.dr.peso
            self.clave = self.iz.clave + self.dr.clave

    def __lt__(self, other):
        return self.peso < other.peso

    def __eq__(self, other):
        return self.peso == other.peso

    def __repr__(self):
        return f"['{self.clave}', {self.peso:.6f}]"

def huffman(frecs:Dict[str, Decimal]):
    h = []
    for clave, peso in frecs.items():
        nodo = Arbol(clave = clave, peso = peso)
        heappush(h, nodo)
    while len(h) > 1:
        iz = heappop(h)
        dr = heappop(h)
        a = Arbol(iz = iz, dr = dr)
        heappush(h, a)
    return h[0]


def main():
    with open(FICHERO_MUESTRA_ES) as f:
        texto = '\n'.join(f.readlines())
        frec_es = frecuencias(texto)
    with open(FICHERO_MUESTRA_EN) as f:
        texto = '\n'.join(f.readlines())
        frec_en = frecuencias(texto)
    h = huffman(frec_es)
    print(h)


    
    

if __name__ == "__main__":
    main()
