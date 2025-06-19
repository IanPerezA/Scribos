# utils/cargador_modelo.py

from pathlib import Path
import os
import requests
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Import correcto de tus settings
from config.config import settings

_MODELO_CACHE: dict[str, tuple] = {}

# IDs de Google Drive para cada archivo necesario
MODEL_FILES = {
    "config.json":           "1RgRZJCEw78f5DrkautcCI3URS7ybgsmF",  # config
    "pytorch_model.bin":     "1X1tBl7MwIDULfjQ-LRvheII-akYNFfVj",  # checkpoint PyTorch
    "tokenizer_config.json": "182dKzFrps8j9Km9LnVQSDFvVz1ajryxV",
    "tokenizer.json":        "1f40a9O13ZtmiinPU2kbfzPAnmYFdpQLC",
    "vocab.txt":             "1bIcjcPPa9vut_DsTZuBudnv92Zvs4-H0",
    "special_tokens_map.json":"10IpJEw5WzwC_EYWuPGS1Qw2reNbQ8FSp",
}

# Directorio local donde caerá el modelo
MODEL_DIR = Path(settings.LOCAL_MODEL_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def _descargar_drive(file_id: str, destino: Path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(destino, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    print(f"✔ Descargado {destino.name}")

def asegurar_modelo_descargado():
    for fname, file_id in MODEL_FILES.items():
        dst = MODEL_DIR / fname
        if not dst.exists():
            print(f"⬇️  Bajando {fname} …")
            _descargar_drive(file_id, dst)

def get_model():
    """
    Descarga los pesos y archivos si falta alguno, y luego
    construye/tokeniza el modelo localmente.
    """
    key = settings.MODEL_NAME
    if key in _MODELO_CACHE:
        return _MODELO_CACHE[key]

    # Asegura que estén todos los archivos
    asegurar_modelo_descargado()

    print(f"📂 Cargando modelo desde: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        from_tf=False,
        trust_remote_code=False
    )

    _MODELO_CACHE[key] = (tokenizer, model)
    return tokenizer, model
