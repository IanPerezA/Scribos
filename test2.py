import os
import zipfile
import gdown
import shutil

# Simula tu variable de entorno
DRIVE_MODEL_URL = "https://drive.google.com/file/d/14kk86vRe98SzPP2WqZE7wuIN2eT9whVJ/view?usp=sharing"
LOCAL_MODEL_DIR = "modelo_beto_test"
ZIP_DEST_PATH = "modelo_beto_test.zip"

def extraer_file_id(url: str) -> str | None:
    try:
        if "file/d/" in url:
            return url.split("file/d/")[1].split("/")[0]
    except Exception:
        return None
    return None

def descargar_y_extraer_zip_drive():
    file_id = extraer_file_id(DRIVE_MODEL_URL)
    if not file_id:
        print("❌ No se pudo extraer el ID del archivo de Drive.")
        return

    print("⬇️ Descargando archivo ZIP desde Google Drive...")
    try:
        gdown.download(id=file_id, output=ZIP_DEST_PATH, quiet=False)
    except Exception as e:
        print(f"❌ Error durante la descarga: {e}")
        return

    print("📦 Intentando descomprimir...")

    if not zipfile.is_zipfile(ZIP_DEST_PATH):
        print("❌ El archivo descargado no es un archivo ZIP válido.")
        return

    # Crear directorio destino si no existe
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    try:
        with zipfile.ZipFile(ZIP_DEST_PATH, "r") as zip_ref:
            zip_ref.extractall(LOCAL_MODEL_DIR)
        print("✅ Descompresión completada correctamente.")
    except Exception as e:
        print(f"❌ Error al descomprimir: {e}")
    finally:
        # Limpieza opcional del zip
        os.remove(ZIP_DEST_PATH)

if __name__ == "__main__":
    # Elimina carpeta previa si existe para pruebas limpias
    if os.path.exists(LOCAL_MODEL_DIR):
        shutil.rmtree(LOCAL_MODEL_DIR)

    descargar_y_extraer_zip_drive()
