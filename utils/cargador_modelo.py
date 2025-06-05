from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from config import config
from config.config import settings

_MODELO_CACHE: dict[str, tuple] = {}   

def get_model():
    """Devuelve (tokenizer, model) y descarga si hace falta."""
    name = settings.MODEL_NAME
    local_dir = Path(settings.LOCAL_MODEL_DIR)

    if name in _MODELO_CACHE:
        return _MODELO_CACHE[name]

    if local_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForMaskedLM.from_pretrained(local_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForMaskedLM.from_pretrained(name)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)

    _MODELO_CACHE[name] = (tokenizer, model)
    return tokenizer, model
