from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from config.config import settings
import os
import requests

_MODELO_CACHE: dict[str, tuple] = {}

# IDs de Drive para cada archivo
MODEL_FILES = {
    "config.json":               "1RgRZJCEw78f5DrkautcCI3URS7ybgsmF",  # tu nuevo config.json
    "pytorch_model.bin":         "1X1tBl7MwIDULfjQ-LRvheII-akYNFfVj",  # binario PyTorch
    "tokenizer_config.json":     "182dKzFrps8j9Km9LnVQSDFvVz1ajryxV",
    "tokenizer.json":            "1f40a9O13ZtmiinPU2kbfzPAnmYFdpQLC",
    "vocab.txt":                 "1bIcjcPPa9vut_DsTZuBudnv92Zvs4-H0",
    "special_tokens_map.json":   "10IpJEw5WzwC_EYWuPGS1Qw2reNbQ8FSp",
}

# Carpeta donde caerán los archivos
MODEL_DIR = Path(settings.LOCAL_MODEL_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def descargar_desde_drive(file_id: str, destino: Path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(destino, "wb") as f:
        for chunk in resp.iter_content(1024 * 1024):
            f.write(chunk)
    print(f"✔ {destino.name} descargado.")

def asegurar_modelo_descargado():
    for fname, fid in MODEL_FILES.items():
        dst = MODEL_DIR / fname
        if not dst.exists():
            print(f"⬇️ Bajando {fname} …")
            descargar_desde_drive(fid, dst)

def get_model():
    key = settings.MODEL_NAME
    if key in _MODELO_CACHE:
        return _MODELO_CACHE[key]

    # baja los pesos si no están
    asegurar_modelo_descargado()

    print(f"📂 Cargando modelo desde {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model     = AutoModelForMaskedLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    _MODELO_CACHE[key] = (tokenizer, model)
    return tokenizer, model

# — archivos auxiliares (índice y vocabulario) —

EXTRAS = {
    settings.WORDS_FILE.as_posix(): settings.WORDS_FILE,  # ej: "data/words.txt"
    settings.INDEX_FILE.as_posix(): settings.INDEX_FILE,  # ej: "data/index.json"
}
# Mapea cada path local a su file_id de Drive
EXTRAS_IDS = {
    settings.WORDS_FILE.as_posix():   "17ytyBG1Vsk1qDF8JnY_LbQij19vNYHDV",
    settings.INDEX_FILE.as_posix():   "1aRd4Pvd44XUPz7IziZJvrt_c_FGLljy5",
}

# Crear carpeta data y bajar
Path("data").mkdir(exist_ok=True)
for ruta, file_id in EXTRAS_IDS.items():
    if not Path(ruta).exists():
        print(f"⬇️ Bajando recurso auxiliar {ruta} …")
        descargar_desde_drive(file_id, Path(ruta))
