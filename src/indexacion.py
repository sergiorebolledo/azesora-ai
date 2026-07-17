import os
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
    
    print(f"\n📦 Guardando e indexando {len(documentos)} fragmentos en ChromaDB...")
    
    # 2. Crear el índice vectorial local en la carpeta 'vector_db'
    ruta_db = "vector_db"
    vector_store = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        persist_directory=ruta_db
    )
    
    print(f"✅ Base de datos vectorial persistida con éxito en la carpeta '{ruta_db}/'!")
    return vector_store

if __name__ == "__main__":
    inicializar_base_vectorial()