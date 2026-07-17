import os
import sys

# La consola de Windows no usa UTF-8 por defecto y rompe al hacer print() de emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_huggingface import HuggingFaceEmbeddings
# Cambiamos langchain_community por langchain_chroma:
from langchain_chroma import Chroma
from ingesta import procesar_archivos_corporativos

def inicializar_base_vectorial():
    print("🧠 Inicializando el modelo de Embeddings (All-MiniLM-L6-v2)...")
    # Usamos un modelo open-source muy eficiente para ejecutar en local
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 1. Obtener los fragmentos de texto del pipeline anterior
    documentos = procesar_archivos_corporativos()

    # 2. Conectar (o crear) el índice vectorial local en la carpeta 'vector_db'
    ruta_db = "vector_db"
    vector_store = Chroma(persist_directory=ruta_db, embedding_function=embeddings)

    # 3. Reemplazar solo las fuentes que vamos a re-procesar (evita duplicar chunks viejos sin
    # número de página, sin borrar documentos subidos manualmente vía la app que no pasan por este pipeline)
    fuentes_a_reemplazar = list({doc.metadata.get("fuente") for doc in documentos if doc.metadata.get("fuente")})
    if fuentes_a_reemplazar:
        print(f"🗑️ Eliminando versiones previas de: {', '.join(fuentes_a_reemplazar)}...")
        vector_store.delete(where={"fuente": {"$in": fuentes_a_reemplazar}})

    print(f"\n📦 Guardando e indexando {len(documentos)} fragmentos en ChromaDB...")
    vector_store.add_documents(documentos)

    print(f"✅ Base de datos vectorial persistida con éxito en la carpeta '{ruta_db}/'!")
    return vector_store

if __name__ == "__main__":
    inicializar_base_vectorial()