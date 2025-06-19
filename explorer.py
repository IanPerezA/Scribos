from transformers import AutoTokenizer, AutoModelForMaskedLM
import os

model_dir = "D:/scribos/repo-scribos/Scribos/modelo_beto_test/modelo_beto"

# Ver qué archivos hay realmente
print("🗂 Contenido del modelo:")
for f in os.listdir(model_dir):
    print(f)

# Forzar uso local solamente
print("\n🔍 Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

print("🔍 Cargando modelo...")
model = AutoModelForMaskedLM.from_pretrained(model_dir, local_files_only=True)

print("✅ Todo cargado correctamente")
