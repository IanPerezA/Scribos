from pathlib import Path
from utils.indexador import construir_indice, buscar
from config.config import settings
class SuggestionService:
    def __init__(self):
        palabras = Path(settings.WORDS_FILE).read_text(encoding="utf-8").splitlines()
        self.indice = construir_indice(palabras)

    def sugerir(self, patron: list[str]) -> list[str]:
        return buscar(self.indice, patron)
