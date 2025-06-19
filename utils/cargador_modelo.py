from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from config.config import settings
import gdown
import zipfile
import os
import requests

_MODELO_CACHE: dict[str, tuple] = {}

def extraer_file_id(url: str) -> str:
    """Extrae el ID de archivo desde una URL estándar de Google Drive."""
    if "id=" in url:
        return url.split("id=")[1].split("&")[0]
    elif "/d/" in url:
        return url.split("/d/")[1].split("/")[0]
    raise ValueError("URL de Drive no válida.")

def descargar_modelo_drive():
    url = settings.DRIVE_MODEL_URL
    output_zip = "modelo_beto.zip"
    file_id = extraer_file_id(url)

    if Path(output_zip).exists():
        os.remove(output_zip)

    print("⬇️ Descargando archivo ZIP desde Google Drive...")
    gdown.download(id=file_id, output=output_zip, quiet=False)

    try:
        print("📦 Intentando descomprimir...")
        with zipfile.ZipFile(output_zip, "r") as zip_ref:
            zip_ref.extractall(settings.LOCAL_MODEL_DIR)
        print("✅ Modelo extraído correctamente.")
    except zipfile.BadZipFile:
        print("❌ El archivo descargado no es un archivo ZIP válido.")
        os.remove(output_zip)
        raise
def get_model():
    name = settings.MODEL_NAME
    base_dir = Path(settings.LOCAL_MODEL_DIR)

    if name in _MODELO_CACHE:
        return _MODELO_CACHE[name]

    # Descargar si no existe o está vacío
    if not base_dir.exists() or not any(base_dir.iterdir()):
        descargar_modelo_drive()

    # Buscar recursivamente la carpeta que contenga los archivos del modelo
    posibles_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if {"config.json", "pytorch_model.bin", "tokenizer_config.json"}.intersection(files):
            posibles_dirs.append(Path(root))

    if not posibles_dirs:
        raise RuntimeError(" No se encontró una carpeta válida con archivos del modelo.")

    model_dir = posibles_dirs[0]  # Usa la primera carpeta válida encontrada

    print(f"📂 Cargando modelo desde: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir)

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
