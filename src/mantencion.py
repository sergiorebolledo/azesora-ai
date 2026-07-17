import os
import sys
import json
import hashlib

# La consola de Windows no usa UTF-8 por defecto y rompe al hacer print() de emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

try:
    # Cuando se importa como parte del paquete src (ej. desde app.py)
    from .indexacion import inicializar_base_vectorial
except ImportError:
    # Cuando se ejecuta directamente: python src/mantencion.py
    from indexacion import inicializar_base_vectorial

RAIZ_DOCUMENTOS = "data_source"
RUTA_MANIFEST = os.path.join("vector_db", ".manifest_documentos.json")


def _calcular_huella_documentos():
    """Devuelve {ruta_relativa: hash_sha256} de todos los archivos actuales en data_source/."""
    huella = {}
    if not os.path.isdir(RAIZ_DOCUMENTOS):
        return huella
    for carpeta_actual, _, archivos in os.walk(RAIZ_DOCUMENTOS):
        for nombre_archivo in archivos:
            ruta = os.path.join(carpeta_actual, nombre_archivo)
            ruta_relativa = os.path.relpath(ruta, RAIZ_DOCUMENTOS).replace("\\", "/")
            with open(ruta, "rb") as f:
                huella[ruta_relativa] = hashlib.sha256(f.read()).hexdigest()
    return huella


def _cargar_manifest_anterior():
    if os.path.exists(RUTA_MANIFEST):
        with open(RUTA_MANIFEST, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _guardar_manifest(huella):
    os.makedirs(os.path.dirname(RUTA_MANIFEST), exist_ok=True)
    with open(RUTA_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(huella, f, ensure_ascii=False, indent=2)


def sincronizar_documentos(forzar=False):
    """Detecta archivos nuevos, modificados o eliminados en data_source/ desde la última sincronización
    (comparando hashes SHA-256) y reindexa automáticamente solo si hubo cambios, o si forzar=True.

    Pensado para ejecutarse al iniciar la app (detección automática) o como rutina periódica vía
    cron/Task Scheduler (`python src/mantencion.py`).
    """
    huella_anterior = _cargar_manifest_anterior()
    huella_actual = _calcular_huella_documentos()

    nuevos = [r for r in huella_actual if r not in huella_anterior]
    modificados = [r for r in huella_actual if r in huella_anterior and huella_actual[r] != huella_anterior[r]]
    eliminados = [r for r in huella_anterior if r not in huella_actual]

    hay_cambios = bool(nuevos or modificados or eliminados)

    if not hay_cambios and not forzar:
        print("✅ No hay cambios en data_source/. El índice vectorial ya está al día.")
        return {"cambios": False, "nuevos": [], "modificados": [], "eliminados": []}

    print(f"🔄 Cambios detectados -> nuevos: {len(nuevos)}, modificados: {len(modificados)}, eliminados: {len(eliminados)}")

    # Limpiar del índice los archivos que ya no existen en data_source/: el pipeline normal no puede
    # tocarlos porque solo procesa lo que encuentra en disco, así que sus chunks quedarían huérfanos.
    if eliminados:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory="vector_db", embedding_function=embeddings)
        for ruta_relativa in eliminados:
            nombre_archivo = os.path.basename(ruta_relativa)
            vector_store.delete(where={"fuente": nombre_archivo})
            print(f"🗑️ Eliminado del índice: {nombre_archivo}")

    # Reprocesar y reindexar el corpus vigente (el pipeline ya reemplaza por fuente, sin duplicar)
    inicializar_base_vectorial()

    _guardar_manifest(huella_actual)
    return {"cambios": True, "nuevos": nuevos, "modificados": modificados, "eliminados": eliminados}


if __name__ == "__main__":
    sincronizar_documentos()
