from pathlib import Path
from utils.indexador import cargar_index, buscar
from config.config import settings
class SuggestionService:
    def __init__(self):
        self.indice = cargar_index(settings.INDEX_FILE)

    def sugerir(self, patron: list[str]) -> list[str]:
        return buscar(self.indice, patron)
