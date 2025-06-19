from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from config.config import settings
import os

_MODELO_CACHE = {}

def get_model():
    key = settings.MODEL_NAME
    if key in _MODELO_CACHE:
        return _MODELO_CACHE[key]

    model_path = Path(settings.LOCAL_MODEL_DIR)
    model_path.mkdir(parents=True, exist_ok=True)

    # Intenta cargar desde caché local primero
    try:
        if (model_path / "pytorch_model.bin").exists():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            _MODELO_CACHE[key] = (tokenizer, model)
            return tokenizer, model
    except Exception as e:
        print(f"⚠ Error al cargar modelo local: {e}")

    # Si no existe, descarga directamente de HuggingFace
    print("⬇️ Descargando modelo desde HuggingFace...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(settings.MODEL_NAME)
        
        # Guarda localmente para futuras ejecuciones
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        _MODELO_CACHE[key] = (tokenizer, model)
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error crítico al descargar el modelo: {e}")
        raise