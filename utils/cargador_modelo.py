import os
import requests
from transformers import AutoModelForMaskedLM, AutoTokenizer
from config.config import settings

MODEL_FILES = {
    "config.json": "1RgRZJCEw78f5DrkautcCI3URS7ybgsmF",  # config.json
    "pytorch_model.bin": "1X1tBl7MwIDULfjQ-LRvheII-akYNFfVj"  # pytorch weights
}

def download_from_drive(file_id: str, dest_path: str):
    """Descarga un archivo público de Google Drive por ID."""
    print(f"⬇️ Descargando {dest_path}...")
    URL = "https://drive.google.com/uc?export=download"
    response = requests.get(URL, params={"id": file_id}, stream=True)
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✔ Descarga exitosa: {dest_path}")
    else:
        raise Exception(f"❌ Error al descargar {dest_path}: {response.status_code}")

def ensure_model_files():
    """Verifica y descarga los archivos necesarios si no existen."""
    os.makedirs(settings.LOCAL_MODEL_DIR, exist_ok=True)
    for filename, file_id in MODEL_FILES.items():
        local_path = os.path.join(settings.LOCAL_MODEL_DIR, filename)
        if not os.path.exists(local_path):
            download_from_drive(file_id, local_path)

def get_model():
    """Asegura y carga el modelo desde la carpeta local."""
    ensure_model_files()
    print(f"📂 Cargando modelo desde: {settings.LOCAL_MODEL_DIR}")
    model = AutoModelForMaskedLM.from_pretrained(settings.LOCAL_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-uncased")
    return tokenizer, model
