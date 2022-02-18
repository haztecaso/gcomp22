#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Autor: Adrián Lattes Grassi
"""

from typing import Dict, List
from decimal import Decimal
from heapq import heappop, heappush
from graphviz import Digraph

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
        self._tabla_codigos = None

    def __lt__(self, other):
        return self.peso < other.peso

    def __eq__(self, other):
        return self.peso == other.peso

    def __repr__(self):
        return f"['{self.clave}', {self.peso:.6f}]"

    @property
    def tabla_codigos(self):
        if self._tabla_codigos is None:
            if self.hoja:
                self._tabla_codigos = {self.clave: []}
            else:
                codigos_iz = { clave: [0] + codigo for (clave, codigo) in self.iz.tabla_codigos.items()}
                codigos_dr = { clave: [1] + codigo for (clave, codigo) in self.dr.tabla_codigos.items()}
                self._tabla_codigos = {**codigos_iz, **codigos_dr}
        return self._tabla_codigos

    def graph(self, dot:Digraph = Digraph(comment='Árbol de Huffman'), render:bool = True, title:str='arbol'):
        if self.hoja:
            dot.node(self.clave, repr(self))
        else:
            dot.node(self.clave, repr(self))
            self.iz.graph(dot = dot, render = False)
            self.dr.graph(dot = dot, render = False)
            dot.edge(self.clave, self.iz.clave, label="0")
            dot.edge(self.clave, self.dr.clave, label="1")
        if render:
            print(f"Guardando árbol en {title}.pdf")
            dot.render(title)

def huffman(frecs:Dict[str, Decimal]):
    heap = []
    for clave, peso in frecs.items():
        nodo = Arbol(clave = clave, peso = peso)
        heappush(heap, nodo)
    while len(heap) > 1:
        iz = heappop(heap)
        dr = heappop(heap)
        a = Arbol(iz = iz, dr = dr)
        heappush(heap, a)
    return heap[0]

def codigo_str(codigo:List[int]):
    return ''.join(map(str, codigo))

def main():
    with open(FICHERO_MUESTRA_ES) as f:
        texto = '\n'.join(f.readlines())
        frec_es = frecuencias(texto)
    with open(FICHERO_MUESTRA_EN) as f:
        texto = '\n'.join(f.readlines())
        frec_en = frecuencias(texto)
    arbol_es = huffman(frec_es)
    arbol_en = huffman(frec_en)
    arbol_es.graph(title = 'arbol_es')
    arbol_en.graph(title = 'arbol_en')

if __name__ == "__main__":
    main()
