#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Autor: Adrián Lattes Grassi
"""

from decimal import Decimal
from functools import reduce
from heapq import heappop, heappush
from os import remove as remove_file
from typing import Dict, List, Optional


def frecuencias(texto:str)-> Dict[str, Decimal]:
    """
    Dado un texto calcula las frecuencias de cada caracter del texto.
    """
    letras = dict()
    for char in texto:
        if char not in letras:
            letras[char] = Decimal(1)
        else:
            letras[char] += 1
    n = len(texto)
    return {char:frec/n for (char, frec) in letras.items()}


class Codigo():
    def __init__(self, codigo:List[bool] = None):
        self._codigo = codigo if codigo is not None else []

    def pre(self, valor:bool):
        return Codigo([valor]+self._codigo)

    @property
    def vacio(self):
        return len(self._codigo) == 0

    def __add__(self, other):
        return Codigo(self._codigo + other._codigo)

    def __len__(self):
        return len(self._codigo)

    def __iter__(self):
        yield from self._codigo

    def __repr__(self):
        return f"[{''.join(map(lambda b: '1' if b else '0', self._codigo))}]"


class ArbolHuffman():
    """
    Clase para árboles de Huffman.
    """
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
        return f"['{self.clave}', {self.peso:.10f}]"

    @property
    def tabla_codigos(self):
        if self._tabla_codigos is None:
            if self.hoja:
                self._tabla_codigos = {self.clave: Codigo()}
            else:
                codigos_iz = { clave: codigo.pre(False) for (clave, codigo) in self.iz.tabla_codigos.items()}
                codigos_dr = { clave: codigo.pre(True) for (clave, codigo) in self.dr.tabla_codigos.items()}
                self._tabla_codigos:Optional[Dict[str,Codigo]] = {**codigos_iz, **codigos_dr}
        return self._tabla_codigos

    def codificar(self, data:str):
        return reduce(lambda x,y: x+y,
                map(lambda e: self.tabla_codigos[e], data),
                Codigo())

    def decodificar(self, codigo:Codigo):
        actual = self
        codigo_actual = Codigo()
        result = ""
        for b in codigo:
            actual = actual.dr if b else actual.iz
            codigo_actual += Codigo([b])
            if actual.hoja:
                result += actual.clave
                actual = self
                codigo_actual = Codigo()
        assert codigo_actual.vacio, f"No se ha terminado de decodificar el código. Código restante: {codigo_actual}"
        return result

    def graph(self, dot = None, render:bool = True, title:str='arbol'):
        try:
            from graphviz import Digraph
        except ModuleNotFoundError as e:
            print(f"ERROR: Para generar la gráfica de un ArbolHuffman es necesario instalar el paquete de python {e.name}.")
        else:
            if dot is None:
                dot = Digraph(comment = title)
            if self.hoja:
                dot.node(self.clave, repr(self))
            else:
                dot.node(self.clave, repr(self))
                self.iz.graph(dot, False)
                self.dr.graph(dot, False)
                dot.edge(self.clave, self.iz.clave, label="0")
                dot.edge(self.clave, self.dr.clave, label="1")
            if render:
                print(f"Guardando árbol en {title}.pdf")
                dot.render(title)
                remove_file(title) # Borrando archivo extra que crea graphviz
            return dot


def huffman(frecs:Dict[str, Decimal]) -> ArbolHuffman:
    """
    Implementación del algoritmo de Huffman.
    """
    heap = []
    for clave, peso in frecs.items():
        heappush(heap, ArbolHuffman(clave = clave, peso = peso))
    while len(heap) > 1:
        iz = heappop(heap)
        dr = heappop(heap)
        heappush(heap, ArbolHuffman(iz = iz, dr = dr))
    return heap[0]


def longitud_media(frecuencias:Dict[str, Decimal], tabla_codigos:Dict[str,Codigo]) -> Decimal:
    result = Decimal(0)
    for clave, peso in frecuencias.items():
        result += peso*len(tabla_codigos[clave])
    return result

def main():
    with open("./GCOM2022_pract2_auxiliar_esp.txt") as f:
        texto = '\n'.join(f.readlines())
        frec_es = frecuencias(texto)
    with open("./GCOM2022_pract2_auxiliar_eng.txt") as f:
        texto = '\n'.join(f.readlines())
        frec_en = frecuencias(texto)

    arbol_es = huffman(frec_es)
    arbol_en = huffman(frec_en)

    print(f"{longitud_media(frec_es, arbol_es.tabla_codigos) = }")
    print(f"{longitud_media(frec_en, arbol_en.tabla_codigos) = }")

    arbol_es.graph(title = 'arbol_es')
    arbol_en.graph(title = 'arbol_en')

    palabra = "medieval"

    codigo_palabra_es = arbol_es.codificar(palabra)
    codigo_palabra_en = arbol_en.codificar(palabra)

    print(f"{arbol_es.decodificar(codigo_palabra_es), codigo_palabra_es = }")
    print(f"{arbol_en.decodificar(codigo_palabra_en), codigo_palabra_en = }")

    codigo = Codigo(list(map(lambda d:True if d == '1' else False, "10111101101110110111011111")))
    print(f"{codigo, arbol_en.decodificar(codigo)                       = }")


if __name__ == "__main__":
    main()
