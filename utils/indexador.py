from __future__ import annotations
from collections import defaultdict
from typing import Iterable

Indice = dict[int, dict[str, list[str]]]

def construir_indice(palabras: Iterable[str]) -> Indice:
    indice: Indice = defaultdict(lambda: defaultdict(list))
    for palabra in palabras:
        for pos, letra in enumerate(palabra):
            indice[pos][letra].append(palabra)
    return indice

def buscar(indice: Indice, patron: list[str]) -> list[str]:
    subconjuntos = [
        set(indice.get(i, {}).get(ch, []))
        for i, ch in enumerate(patron) if ch
    ]
    return list(set.intersection(*subconjuntos)) if subconjuntos else []
