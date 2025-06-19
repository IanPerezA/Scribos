from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from config.config import settings
import gdown
import zipfile
import os
import requests

_MODELO_CACHE: dict[str, tuple] = {}

def descargar_modelo_drive():
    url = settings.DRIVE_MODEL_URL
    output_zip = "modelo_beto.zip"

    if Path(output_zip).exists():
        os.remove(output_zip)

    print("Descargando modelo desde Drive...")
    gdown.download(url, output_zip, quiet=False)

    try:
        with zipfile.ZipFile(output_zip, "r") as zip_ref:
            zip_ref.extractall(settings.LOCAL_MODEL_DIR)
        print("Modelo extraído correctamente.")
    except zipfile.BadZipFile:
        print("❌ Archivo ZIP inválido. Eliminando y abortando carga.")
        os.remove(output_zip)
        raise

def get_model():
    name = settings.MODEL_NAME
    local_dir = Path(settings.LOCAL_MODEL_DIR)

    if name in _MODELO_CACHE:
        return _MODELO_CACHE[name]

    if not local_dir.exists() or not any(local_dir.iterdir()):
        descargar_modelo_drive()

    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForMaskedLM.from_pretrained(local_dir)

    _MODELO_CACHE[name] = (tokenizer, model)
    return tokenizer, model

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

# Descarga los recursos si no existen
URL_WORDS_ID = "17ytyBG1Vsk1qDF8JnY_LbQij19vNYHDV"
URL_INDEX_ID = "1aRd4Pvd44XUPz7IziZJvrt_c_FGLljy5"

LOCAL_PATH1 = "data/words.json"
LOCAL_PATH2 = "data/index.json"

Path("data").mkdir(exist_ok=True)

if not Path(LOCAL_PATH1).exists():
    descargar_desde_drive(URL_WORDS_ID, LOCAL_PATH1)

if not Path(LOCAL_PATH2).exists():
    descargar_desde_drive(URL_INDEX_ID, LOCAL_PATH2)
