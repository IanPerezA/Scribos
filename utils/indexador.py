from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Iterable

Indice = dict[int, dict[str, list[str]]]

def construir_indice(palabras: Iterable[str]) -> Indice:
    indice: Indice = defaultdict(lambda: defaultdict(list))
    for palabra in palabras:
        for pos, letra in enumerate(palabra):
            indice[pos][letra].append(palabra)
    return indice

def cargar_index(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

"""
FUNCION PRINCIPAL
"""
def buscar(indice: Indice, patron: list[str]) -> list[str]:
    longitud_objetivo = len(patron)

    subconjuntos = [
        set(indice.get(str(i), {}).get(ch, []))  # usar str(i) si las claves son strings
        for i, ch in enumerate(patron) if ch
    ]

    if not subconjuntos:
        return []

    coincidencias = set.intersection(*subconjuntos)
    return [palabra for palabra in coincidencias if len(palabra) == longitud_objetivo]
