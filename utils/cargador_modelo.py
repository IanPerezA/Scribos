from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from config import config
from config.config import settings
import gdown
import zipfile

_MODELO_CACHE: dict[str, tuple] = {}

def descargar_modelo_drive():
    url = settings.DRIVE_MODEL_URL  # Debes definir esta variable en config
    output_zip = "modelo_beto.zip"
    gdown.download(url, output_zip, quiet=False)
    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_ref.extractall(settings.LOCAL_MODEL_DIR)

def get_model():
    name = settings.MODEL_NAME
    local_dir = Path(settings.LOCAL_MODEL_DIR)

    if name in _MODELO_CACHE:
        return _MODELO_CACHE[name]

    if not local_dir.exists():
        descargar_modelo_drive()

    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForMaskedLM.from_pretrained(local_dir)

    _MODELO_CACHE[name] = (tokenizer, model)
    return tokenizer, model
