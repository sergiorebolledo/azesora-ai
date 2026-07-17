import os
import time
from dotenv import load_dotenv

# Forzamos a Python a buscar el archivo .env un nivel arriba de la carpeta 'src'
ruta_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=ruta_env)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def consultar_azesora(pregunta, historial_mensajes=None, filtro_pais=None, modo_experto=False):
    """
    Agente RAG con memoria conversacional.
    historial_mensajes debe ser una lista de dicts: [{"role": "user"/"agent", "content": "..."}]
    """
    inicio = time.time()
    
    # 1. Cargar el mismo modelo de embeddings local
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Conectar con la base de datos vectorial existente
    vector_store = Chroma(persist_directory="vector_db", embedding_function=embeddings)
    
    # 3. Configurar la búsqueda semántica básica o con filtro de metadatos (País)
    argumentos_busqueda = {"k": 4}  # Subimos a 4 para capturar más contexto tabular si existe
    if filtro_pais:
        argumentos_busqueda["filter"] = {"pais": filtro_pais.lower()}
        
    retriever = vector_store.as_retriever(search_kwargs=argumentos_busqueda)
    
    # 4. Recuperar los fragmentos relevantes para la pregunta
    fragmentos_recuperados = retriever.invoke(pregunta)
    
    # Ensamblar el contexto y rastrear las fuentes
    contexto = ""
    fuentes = set()
    for doc in fragmentos_recuperados:
        contexto += doc.page_content + "\n\n"
        fuentes.add(doc.metadata.get("fuente", "Desconocida"))
        
    # 5. Configurar el LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2 # Subimos un toque a 0.2 para dar fluidez conversacional sin alucinar
    )
    
    # 6. Crear el System Prompt Evolucionado (Instrucciones para Tablas y Memoria)
    if modo_experto:
        instrucciones_modo = (
            "Estás en MODO EXPERTO / TÉCNICO. Responde con rigor legal y técnico, dirigido a un profesional del sector. "
            "Cita explícitamente los artículos de ley, formularios, tasas y normativas mencionadas en el contexto (ej. 'Artículo 107 de la LIR', "
            "'Formulario W-8BEN'). Usa terminología precisa sin simplificar conceptos. Estructura la respuesta priorizando exactitud sobre fluidez."
        )
    else:
        instrucciones_modo = (
            "Estás en MODO SIMPLE / AMIGABLE. Responde de forma cercana y didáctica, como si le explicaras a alguien sin conocimientos "
            "técnicos previos. Prioriza la claridad y evita jerga legal innecesaria, explicando cualquier término técnico que uses."
        )

    system_prompt = (
        "Eres Azesora AI, un asesor financiero experto especializado en inversiones internacionales y regulaciones locales.\n"
        "Responde de forma clara utilizando ÚNICAMENTE el contexto provisto abajo. Si la respuesta incluye datos numéricos, "
        "estructuras comparativas o rentabilidades, preséntalas en tablas limpias de Markdown para facilitar su lectura.\n"
        "Si el contexto no contiene la información, di estrictamente que no posees los registros institucionales. No inventes nada.\n\n"
        f"{instrucciones_modo}\n\n"
        "CONTEXTO DE CONOCIMIENTO:\n"
        f"{contexto}"
    )
    
    # 7. Formatear el historial acumulado al formato nativo de mensajes de LangChain
    mensajes_langchain = [("system", system_prompt)]
    
    if historial_mensajes:
        for msg in historial_mensajes:
            if msg["role"] == "user":
                mensajes_langchain.append(("human", msg["content"]))
            elif msg["role"] == "agent":
                mensajes_langchain.append(("ai", msg["content"]))
                
    # Añadimos la pregunta actual al cierre del hilo conversacional
    mensajes_langchain.append(("human", "{pregunta}"))
    
    prompt_template = ChatPromptTemplate.from_messages(mensajes_langchain)
    
    # 8. Ejecutar la cadena (Prompt + LLM), con Fallback automático a Groq si Gemini falla por cuota o autenticación
    try:
        cadena_rag = prompt_template | llm
        respuesta_ia = cadena_rag.invoke({"pregunta": pregunta})
        print("🧠 Respuesta generada con Gemini (gemini-2.5-flash).")
    except Exception as e:
        error_texto = str(e)
        es_error_cuota = "RESOURCE_EXHAUSTED" in error_texto or "429" in error_texto
        es_error_auth = "UNAUTHENTICATED" in error_texto or "401" in error_texto

        if es_error_cuota or es_error_auth:
            motivo = "cuota agotada" if es_error_cuota else "fallo de autenticación"
            print(f"⚠️ Gemini no disponible ({motivo}). Saltando al respaldo Groq (Llama 3.1)...")
            llm_respaldo = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.2,
                api_key=os.getenv("GROQ_API_KEY")
            )
            cadena_rag = prompt_template | llm_respaldo
            respuesta_ia = cadena_rag.invoke({"pregunta": pregunta})
            print("🧠 Respuesta generada con Groq (llama-3.1-8b-instant) como respaldo.")
        else:
            raise
    
    # Calcular tiempo transcurrido
    fin = time.time()
    tiempo_ejecucion = round(fin - inicio, 2)
    
    return {
        "respuesta": respuesta_ia.content,
        "fuentes": list(fuentes),
        "tiempo": tiempo_ejecucion
    }

if __name__ == "__main__":
    print("🤖 Módulo de Azesora AI cargado listo para consultas conversacionales.")