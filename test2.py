from transformers import AutoModelForMaskedLM

MODEL_DIR = "modelo_beto"
OUTPUT_DIR = "modelo_beto_bin"

print("🔍 Cargando modelo...")
model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR)

print("💾 Guardando modelo en formato bin...")
model.save_pretrained(OUTPUT_DIR, safe_serialization=False)

print(f"✅ Modelo guardado en: {OUTPUT_DIR}")
