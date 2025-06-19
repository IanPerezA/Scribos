# utils/cargador_modelo.py

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from config.config import settings
import requests
import os

# Caché para no recargar mil veces
_MODELO_CACHE: dict[str, tuple] = {}

# IDs de Drive para cada archivo
MODEL_FILES = {
    "config.json":             "1RgRZJCEw78f5DrkautcCI3URS7ybgsmF",  # tu config.json
    "pytorch_model.bin":       "1X1tBl7MwIDULfjQ-LRvheII-akYNFfVj",  # tu bin de PyTorch
    "tokenizer_config.json":   "182dKzFrps8j9Km9LnVQSDFvVz1ajryxV",
    "tokenizer.json":          "1f40a9O13ZtmiinPU2kbfzPAnmYFdpQLC",
    "vocab.txt":               "1bIcjcPPa9vut_DsTZuBudnv92Zvs4-H0",
    "special_tokens_map.json": "10IpJEw5WzwC_EYWuPGS1Qw2reNbQ8FSp",
}

# Carpeta donde se desplegará todo
MODEL_DIR = Path(settings.LOCAL_MODEL_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def _download_from_drive(file_id: str, dest: Path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(1024 * 1024):
            f.write(chunk)
    print(f"✔ {dest.name} descargado.")

def _ensure_model_files():
    for fname, fid in MODEL_FILES.items():
        dst = MODEL_DIR / fname
        if not dst.exists():
            print(f"⬇️ Bajando {fname} …")
            _download_from_drive(fid, dst)

def get_model():
    key = settings.MODEL_NAME
    if key in _MODELO_CACHE:
        return _MODELO_CACHE[key]

    # Descarga todo si hace falta
    _ensure_model_files()

    print(f"📂 Cargando modelo desde {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model     = AutoModelForMaskedLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    _MODELO_CACHE[key] = (tokenizer, model)
    return tokenizer, model

# — Ahora los auxiliares de datos —
EXTRAS = {
    settings.WORDS_FILE.as_posix():   "17ytyBG1Vsk1qDF8JnY_LbQij19vNYHDV",
    settings.INDEX_FILE.as_posix():   "1aRd4Pvd44XUPz7IziZJvrt_c_FGLljy5",
}

Path("data").mkdir(exist_ok=True)
for ruta, fid in EXTRAS.items():
    p = Path(ruta)
    if not p.exists():
        print(f"⬇️ Bajando recurso {ruta} …")
        _download_from_drive(fid, p)
