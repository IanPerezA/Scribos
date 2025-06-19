from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from config.config import settings
import os
import requests

_MODELO_CACHE: dict[str, tuple] = {}

# Mapeo de archivos del modelo y sus IDs de Drive
MODEL_FILES = {
    "config.json": "1fxFzd6Sq4k_IkrgBANGUkhjIyYAW3XQs",
    "tokenizer_config.json": "182dKzFrps8j9Km9LnVQSDFvVz1ajryxV",
    "tokenizer.json": "1f40a9O13ZtmiinPU2kbfzPAnmYFdpQLC",
    "vocab.txt": "1bIcjcPPa9vut_DsTZuBudnv92Zvs4-H0",
    "special_tokens_map.json": "10IpJEw5WzwC_EYWuPGS1Qw2reNbQ8FSp",
    "model.safetensors": "1SzSStMqjEAgnh3sEzxieu0kzs1Xkp9Jq"
}


# Ruta donde se guardarán
MODEL_DIR = Path(settings.LOCAL_MODEL_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def descargar_desde_drive(file_id: str, destino_local: str):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(destino_local, "wb") as f:
            f.write(response.content)
        print(f"✔ Descarga exitosa: {destino_local}")
    except Exception as e:
        print(f"❌ Error al descargar desde Drive: {e}")
        raise

def asegurar_modelo_descargado():
    for nombre_archivo, file_id in MODEL_FILES.items():
        destino = MODEL_DIR / nombre_archivo
        if not destino.exists():
            print(f"⬇️ Descargando {nombre_archivo}...")
            descargar_desde_drive(file_id, str(destino))

def get_model():
    name = settings.MODEL_NAME
    if name in _MODELO_CACHE:
        return _MODELO_CACHE[name]

    asegurar_modelo_descargado()

    print(f"📂 Cargando modelo desde: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR)

    _MODELO_CACHE[name] = (tokenizer, model)
    return tokenizer, model

# --- Archivos auxiliares: palabras e índice ---

EXTRAS = {
    "data/words.json": "17ytyBG1Vsk1qDF8JnY_LbQij19vNYHDV",
    "data/index.json": "1aRd4Pvd44XUPz7IziZJvrt_c_FGLljy5"
}

Path("data").mkdir(exist_ok=True)

for ruta, file_id in EXTRAS.items():
    if not Path(ruta).exists():
        print(f"⬇️ Descargando archivo auxiliar: {ruta}")
        descargar_desde_drive(file_id, ruta)
