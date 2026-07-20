import os
import sys
import time
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

# La consola de Windows no usa UTF-8 por defecto y rompe al hacer print() de emojis;
# forzamos la codificación de salida para que los logs con emoji no crasheen el proceso.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Forzamos a Python a buscar el archivo .env un nivel arriba de la carpeta 'src'
ruta_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=ruta_env)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

RUTA_LOGS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "consultas.jsonl")

# Candidatos iniciales que trae la búsqueda vectorial bruta, antes de reordenarlos por relevancia real.
CANDIDATOS_RERANKING = 12
# Fragmentos finales que se quedan en el contexto tras el reranking (3-5 según sugiere el pipeline).
MAX_FRAGMENTOS_CONTEXTO = 4
# Umbral mínimo de relevancia (score del cross-encoder, no acotado) para considerar un fragmento útil.
# Calibrado empíricamente: preguntas cubiertas por los documentos puntúan entre +4 y +8;
# preguntas fuera de dominio, entre -4 y -10. El umbral en 0 separa ambos grupos con margen amplio.
UMBRAL_RERANK_MINIMO = 0.0

FRASE_SIN_REGISTROS = "no poseo los registros institucionales"


def _respuesta_tiene_respaldo(respuesta_texto, fragmentos_recuperados):
    """Verificación de consistencia: la respuesta debe citar al menos una de las fuentes realmente
    recuperadas, o declarar explícitamente que no tiene información. Si no cumple ninguna de las dos,
    no hay forma de verificar que la respuesta esté respaldada por el contexto."""
    texto_normalizado = respuesta_texto.lower()
    if FRASE_SIN_REGISTROS in texto_normalizado:
        return True
    nombres_fuente = {doc.metadata.get("fuente", "").lower() for doc in fragmentos_recuperados}
    return any(nombre and nombre in texto_normalizado for nombre in nombres_fuente)


_reranker = None


def _obtener_reranker():
    """Carga el cross-encoder de reranking una sola vez (perezoso, se reutiliza entre llamadas)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


_vector_store = None


def obtener_vector_store():
    """Carga el modelo de embeddings y conecta a ChromaDB una sola vez por proceso (perezoso).
    Antes esto se repetía en cada consulta, recargando torch/el modelo cada vez -- muy costoso
    en máquinas con poca RAM/CPU."""
    global _vector_store
    if _vector_store is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vector_store = Chroma(persist_directory="vector_db", embedding_function=embeddings)
    return _vector_store


def _registrar_ejecucion(pregunta, filtro_pais, modo_experto, fragmentos_recuperados, respuesta, motor_usado, tiempo_ejecucion):
    """Deja constancia en logs/consultas.jsonl de cada ejecución (pregunta, contexto usado, respuesta,
    motor y tiempo). Nunca debe interrumpir la respuesta al colaborador si la escritura falla."""
    try:
        entrada = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pregunta": pregunta,
            "filtro_pais": filtro_pais,
            "modo_experto": modo_experto,
            "fragmentos_contexto": [
                {"fuente": doc.metadata.get("fuente", "Desconocida"), "pagina": doc.metadata.get("pagina")}
                for doc in fragmentos_recuperados
            ],
            "respuesta": respuesta,
            "modelo": motor_usado,
            "tiempo_segundos": tiempo_ejecucion
        }
        os.makedirs(os.path.dirname(RUTA_LOGS), exist_ok=True)
        with open(RUTA_LOGS, "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ No se pudo escribir el log de la consulta: {e}")


RUTA_LOGS_FEEDBACK = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "feedback.jsonl")


def registrar_feedback(pregunta, respuesta, valor):
    """Guarda la retroalimentación (positiva/negativa) del colaborador sobre una respuesta,
    para monitoreo de calidad. 'valor' debe ser 'positivo' o 'negativo'."""
    try:
        entrada = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pregunta": pregunta,
            "respuesta": respuesta,
            "valor": valor
        }
        os.makedirs(os.path.dirname(RUTA_LOGS_FEEDBACK), exist_ok=True)
        with open(RUTA_LOGS_FEEDBACK, "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ No se pudo escribir el log de feedback: {e}")


def consultar_azesora(pregunta, historial_mensajes=None, filtro_pais=None, modo_experto=False):
    """
    Agente RAG con memoria conversacional.
    historial_mensajes debe ser una lista de dicts: [{"role": "user"/"agent", "content": "..."}]
    """
    inicio = time.time()

    # 1-2. Modelo de embeddings + conexión a ChromaDB (cacheados, ver obtener_vector_store)
    vector_store = obtener_vector_store()

    # 3. Configurar la búsqueda semántica básica o con filtro de metadatos (País)
    argumentos_busqueda = {"k": CANDIDATOS_RERANKING}
    if filtro_pais:
        argumentos_busqueda["filter"] = {"pais": filtro_pais.lower()}

    # 4. Búsqueda vectorial amplia: traemos más candidatos de los que necesitamos para dejarle
    # trabajo de precisión al reranker (la similitud vectorial pura es rápida pero aproximada).
    candidatos = vector_store.similarity_search_with_score(pregunta, **argumentos_busqueda)

    # 5. Reclasificación (reranking): un cross-encoder evalúa la relación real entre la pregunta
    # completa y cada fragmento candidato, mucho más preciso que la distancia vectorial bruta.
    fragmentos_recuperados = []
    if candidatos:
        reranker = _obtener_reranker()
        pares_pregunta_fragmento = [(pregunta, doc.page_content) for doc, _distancia in candidatos]
        puntajes_rerank = reranker.predict(pares_pregunta_fragmento)
        candidatos_reordenados = sorted(zip(candidatos, puntajes_rerank), key=lambda item: item[1], reverse=True)
        fragmentos_recuperados = [
            doc for (doc, _distancia), puntaje in candidatos_reordenados
            if puntaje >= UMBRAL_RERANK_MINIMO
        ][:MAX_FRAGMENTOS_CONTEXTO]

    if not fragmentos_recuperados:
        fin = time.time()
        tiempo_ejecucion = round(fin - inicio, 2)
        respuesta_sin_contexto = (
            "No encontré información suficientemente relacionada en los documentos disponibles para responder esta "
            "pregunta con confianza. Te sugiero contactar directamente al área responsable (por ejemplo, soporte de tu "
            "broker o el equipo de Finanzas/Legal correspondiente) para obtener una respuesta oficial."
        )
        _registrar_ejecucion(
            pregunta=pregunta, filtro_pais=filtro_pais, modo_experto=modo_experto,
            fragmentos_recuperados=[], respuesta=respuesta_sin_contexto,
            motor_usado="Sin consulta al LLM (umbral de confianza)", tiempo_ejecucion=tiempo_ejecucion
        )
        return {
            "respuesta": respuesta_sin_contexto,
            "fuentes": [],
            "tiempo": tiempo_ejecucion,
            "modelo": "Sin consulta al LLM (umbral de confianza)"
        }

    # Ensamblar el contexto y rastrear las fuentes (cada fragmento va etiquetado con su origen exacto)
    contexto = ""
    fuentes = set()
    for doc in fragmentos_recuperados:
        nombre_fuente = doc.metadata.get("fuente", "Desconocida")
        pagina = doc.metadata.get("pagina")
        etiqueta_origen = f"[Fuente: {nombre_fuente}, página {pagina}]" if pagina else f"[Fuente: {nombre_fuente}]"
        contexto += f"{etiqueta_origen}\n{doc.page_content}\n\n"
        fuentes.add(nombre_fuente)
        
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
        f"Si el contexto no contiene la información que responde la pregunta, di estrictamente que '{FRASE_SIN_REGISTROS}' "
        "para ese tema y detente ahí. NUNCA completes la respuesta con conocimiento general propio, "
        "estimaciones, cifras aproximadas o ejemplos inventados como si fueran datos reales, incluso si el contexto habla de "
        "un tema relacionado pero distinto. No inventes nada.\n\n"
        "Cada fragmento del contexto viene precedido por una etiqueta entre corchetes con su origen exacto, por ejemplo "
        "'[Fuente: regulacion_impuestos_chile.pdf, página 3]'. Cuando tu respuesta se apoye en un fragmento, es OBLIGATORIO "
        "citar esa referencia al final de la afirmación correspondiente con el formato exacto: "
        "'Fuente: [nombre_archivo], página [X]'. Si la etiqueta del fragmento no incluye página (documentos Markdown o CSV), "
        "cita solo 'Fuente: [nombre_archivo]' sin inventar un número de página.\n\n"
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
    
    # 8. Ejecutar la cadena (Prompt + LLM), con Fallback automático a Groq si Gemini falla por cuota, autenticación o servicio
    try:
        cadena_rag = prompt_template | llm
        respuesta_ia = cadena_rag.invoke({"pregunta": pregunta})
        motor_usado = "Gemini 2.5 Flash"
        llm_usado = llm
        print("🧠 Respuesta generada con Gemini (gemini-2.5-flash).")
    except Exception as e:
        error_texto = str(e)
        es_error_cuota = "RESOURCE_EXHAUSTED" in error_texto or "429" in error_texto
        es_error_auth = "UNAUTHENTICATED" in error_texto or "401" in error_texto
        es_error_servicio = "UNAVAILABLE" in error_texto or "503" in error_texto or "INTERNAL" in error_texto or "500" in error_texto

        if es_error_cuota or es_error_auth or es_error_servicio:
            if es_error_cuota:
                motivo = "cuota agotada"
            elif es_error_auth:
                motivo = "fallo de autenticación"
            else:
                motivo = "servicio no disponible"
            print(f"⚠️ Gemini no disponible ({motivo}). Saltando al respaldo Groq (Llama 3.1)...")
            llm_respaldo = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.2,
                api_key=os.getenv("GROQ_API_KEY")
            )
            cadena_rag = prompt_template | llm_respaldo
            respuesta_ia = cadena_rag.invoke({"pregunta": pregunta})
            motor_usado = "Groq Llama 3.1 8B (respaldo)"
            llm_usado = llm_respaldo
            print("🧠 Respuesta generada con Groq (llama-3.1-8b-instant) como respaldo.")
        else:
            raise

    # 9. Verificación de consistencia: la respuesta debe citar una fuente real o declarar que no tiene
    # información. Si no cumple ninguna, se pide una única corrección explícita antes de aceptarla.
    if not _respuesta_tiene_respaldo(respuesta_ia.content, fragmentos_recuperados):
        print("⚠️ La respuesta no citó ninguna fuente reconocible. Solicitando una corrección...")
        try:
            mensaje_correctivo = (
                "Tu respuesta anterior no incluyó ninguna cita de fuente reconocible ni indicó que no tienes la "
                "información. Corrígela: si tu afirmación se basa en el contexto, agrega la cita obligatoria "
                "'Fuente: [nombre_archivo], página [X]'; si no encuentras la información en el contexto, dilo "
                f"explícitamente ('{FRASE_SIN_REGISTROS}') sin agregar nada más."
            )
            mensajes_con_correccion = list(prompt_template.format_messages(pregunta=pregunta)) + [
                AIMessage(content=respuesta_ia.content),
                HumanMessage(content=mensaje_correctivo)
            ]
            respuesta_ia = llm_usado.invoke(mensajes_con_correccion)
        except Exception as e:
            print(f"⚠️ No se pudo obtener la corrección de consistencia, se mantiene la respuesta original: {e}")

    # Calcular tiempo transcurrido
    fin = time.time()
    tiempo_ejecucion = round(fin - inicio, 2)

    # 10. Registrar la ejecución en logs/consultas.jsonl (trazabilidad para auditoría)
    _registrar_ejecucion(
        pregunta=pregunta,
        filtro_pais=filtro_pais,
        modo_experto=modo_experto,
        fragmentos_recuperados=fragmentos_recuperados,
        respuesta=respuesta_ia.content,
        motor_usado=motor_usado,
        tiempo_ejecucion=tiempo_ejecucion
    )

    return {
        "respuesta": respuesta_ia.content,
        "fuentes": list(fuentes),
        "tiempo": tiempo_ejecucion,
        "modelo": motor_usado
    }

if __name__ == "__main__":
    print("🤖 Módulo de Azesora AI cargado listo para consultas conversacionales.")