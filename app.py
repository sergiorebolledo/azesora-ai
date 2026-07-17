import os
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.agente import consultar_azesora, registrar_feedback
from src.ingesta import extraer_texto_docx, extraer_texto_pptx, extraer_texto_html, extraer_registros_json
from src.mantencion import sincronizar_documentos

# Configuración premium de la página
st.set_page_config(
    page_title="Azesora AI - Centro de Inversiones",
    page_icon="💼",
    layout="wide"
)

# --- CACHÉ AVANZADO PARA OPTIMIZAR RENDIMIENTO (Evita demoras de recarga) ---
@st.cache_resource
def sincronizar_documentos_al_iniciar():
    # Se ejecuta una sola vez por proceso del servidor (no en cada clic): detecta si data_source/
    # cambió desde la última vez y reindexa automáticamente antes de servir la primera consulta.
    return sincronizar_documentos()

sincronizar_documentos_al_iniciar()

@st.cache_resource
def obtener_modelo_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def conectar_base_vectorial():
    # Cacheamos la conexión a la base de datos para no leer el disco duro en cada clic
    embeddings = obtener_modelo_embeddings()
    return Chroma(persist_directory="vector_db", embedding_function=embeddings)

# --- SISTEMA DE ESTILOS PREMIUM ---
st.markdown("""
    <style>
    .block-container { max-width: 850px; padding-top: 2rem; padding-bottom: 2rem; }
    .brand-title { font-size: 36px !important; font-weight: 800; color: #0F172A; margin-bottom: 0px; display: flex; align-items: center; gap: 12px; }
    .brand-subtitle { font-size: 15px !important; color: #64748B; margin-bottom: 30px; }
    .chat-bubble-user { background-color: #F8FAFC; padding: 16px 20px; border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 15px; font-size: 15px; color: #1E293B; }
    
    .agent-card-container { 
        background-color: #EFF6FF; 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 4px solid #3B82F6; 
        margin-top: 10px;
        margin-bottom: 15px;
        word-wrap: break-word;
    }
    
    .source-badge { background-color: #F1F5F9; color: #475569; padding: 4px 10px; border-radius: 6px; font-size: 13px; font-weight: 500; border: 1px solid #E2E8F0; display: inline-block; margin-right: 8px; margin-top: 5px; }
    .sidebar-metric { background-color: #F8FAFC; padding: 10px; border-radius: 6px; border: 1px solid #E2E8F0; margin-bottom: 8px; font-size: 12px; }
    .history-item { padding: 8px 12px; border-radius: 6px; margin-bottom: 5px; font-size: 13px; color: #334155; }
    .metadata-caption { font-size: 12px; color: #64748B; margin-bottom: 15px; padding-left: 5px; }
    </style>
""", unsafe_allow_html=True)

# Inicializar estados de memoria
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
if "historial_titulos" not in st.session_state:
    st.session_state.historial_titulos = []

# --- BARRA LATERAL: FILTROS + CARGADOR DE DOCUMENTOS ---
with st.sidebar:
    st.markdown("### 🛠️ Configuración")
    filtro_pais = st.selectbox("📍 Jurisdicción Base:", ["Todos los países", "Chile", "Global"])

    modo_respuesta = st.radio("🎓 Modo de Respuesta:", ["Respuesta Simple", "Respuesta Técnica / Experto"])
    modo_experto = modo_respuesta == "Respuesta Técnica / Experto"

    st.divider()
    
    st.markdown("### 📥 Indexar Nueva Documentación")
    archivo_subido = st.file_uploader(
        "Sube un documento corporativo (PDF, Word, Excel, PowerPoint, Markdown, CSV, JSON o HTML):",
        type=["pdf", "md", "csv", "docx", "xlsx", "pptx", "json", "html", "htm"]
    )
    
    if archivo_subido is not None:
        if st.button("🚀 Indexar en Base de Datos"):
            with st.spinner("Procesando y vectorizando archivo..."):
                try:
                    documentos_nuevos = []
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
                    nombre_archivo = archivo_subido.name
                    
                    if nombre_archivo.endswith(".pdf"):
                        reader = PdfReader(archivo_subido)
                        texto = "".join([pagina.extract_text() + "\n" for pagina in reader.pages])
                        chunks = text_splitter.split_text(texto)
                        for c in chunks:
                            documentos_nuevos.append(Document(page_content=c, metadata={"pais": "global", "fuente": nombre_archivo}))
                            
                    elif nombre_archivo.endswith(".md"):
                        texto = archivo_subido.read().decode("utf-8")
                        chunks = text_splitter.split_text(texto)
                        for c in chunks:
                            documentos_nuevos.append(Document(page_content=c, metadata={"pais": "global", "fuente": nombre_archivo}))
                            
                    elif nombre_archivo.endswith(".csv"):
                        df = pd.read_csv(archivo_subido)
                        for _, fila in df.iterrows():
                            texto_fila = " ".join([f"{col}: {val}." for col, val in fila.items()])
                            documentos_nuevos.append(Document(page_content=texto_fila, metadata={"pais": "global", "fuente": nombre_archivo}))

                    elif nombre_archivo.endswith(".docx"):
                        texto = extraer_texto_docx(archivo_subido)
                        chunks = text_splitter.split_text(texto)
                        for c in chunks:
                            documentos_nuevos.append(Document(page_content=c, metadata={"pais": "global", "fuente": nombre_archivo}))

                    elif nombre_archivo.endswith(".xlsx"):
                        hojas = pd.read_excel(archivo_subido, sheet_name=None)
                        for nombre_hoja, df_hoja in hojas.items():
                            for _, fila in df_hoja.iterrows():
                                texto_fila = ". ".join(f"{col}: {val}" for col, val in fila.items())
                                documentos_nuevos.append(Document(page_content=texto_fila, metadata={"pais": "global", "fuente": nombre_archivo, "hoja": nombre_hoja}))

                    elif nombre_archivo.endswith(".pptx"):
                        for num_slide, texto_slide in extraer_texto_pptx(archivo_subido):
                            if not texto_slide.strip():
                                continue
                            for c in text_splitter.split_text(texto_slide):
                                documentos_nuevos.append(Document(page_content=c, metadata={"pais": "global", "fuente": nombre_archivo, "diapositiva": num_slide}))

                    elif nombre_archivo.endswith(".json"):
                        contenido = extraer_registros_json(archivo_subido.read().decode("utf-8"))
                        if isinstance(contenido, list):
                            for texto_registro in contenido:
                                documentos_nuevos.append(Document(page_content=texto_registro, metadata={"pais": "global", "fuente": nombre_archivo}))
                        else:
                            for c in text_splitter.split_text(contenido):
                                documentos_nuevos.append(Document(page_content=c, metadata={"pais": "global", "fuente": nombre_archivo}))

                    elif nombre_archivo.endswith((".html", ".htm")):
                        texto = extraer_texto_html(archivo_subido.read().decode("utf-8"))
                        chunks = text_splitter.split_text(texto)
                        for c in chunks:
                            documentos_nuevos.append(Document(page_content=c, metadata={"pais": "global", "fuente": nombre_archivo}))

                    if documentos_nuevos:
                        vector_store = conectar_base_vectorial()
                        vector_store.add_documents(documentos_nuevos)
                        st.success(f"¡Éxito! {len(documentos_nuevos)} fragmentos indexados desde {nombre_archivo}")
                        st.cache_resource.clear() # Limpiamos el caché de la BD para que reconozca los nuevos datos de inmediato
                    else:
                        st.error("No se pudo extraer texto del archivo.")
                except Exception as e:
                    st.error(f"Error procesando archivo: {str(e)}")
                    
    st.divider()
    
    st.markdown("### 📜 Historial de Consultas")
    if st.session_state.historial_titulos:
        for titulo in st.session_state.historial_titulos:
            st.markdown(f'<div class="history-item">💬 {titulo}</div>', unsafe_allow_html=True)
    else:
        st.caption("No hay consultas en esta sesión.")
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### ⚙️ Panel Técnico (Dev)")
    st.markdown('<div class="sidebar-metric">🟢 <b>Base de Datos:</b> Conectada (ChromaDB)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-metric">🧠 <b>Modelo principal:</b> Gemini 2.5 Flash (respaldo: Groq Llama 3.1)</div>', unsafe_allow_html=True)

# --- PANEL PRINCIPAL ---
st.markdown('<div class="brand-title">🌐 Azesora AI</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Consultor Inteligente de Inversiones Internacionales y Regulación Local</div>', unsafe_allow_html=True)

# Renderizar mensajes pasados
for idx, msg in enumerate(st.session_state.mensajes):
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user"><b>👤 Tú:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(f'<div class="agent-card-container"><b>🤖 Azesora AI:</b></div>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            
        if "tiempo" in msg:
            modelo_usado = msg.get("modelo", "Gemini 2.5 Flash")
            st.markdown(f'<div class="metadata-caption">⏱️ <b>Tiempo de respuesta:</b> {msg["tiempo"]}s | 🧠 <b>Modelo:</b> {modelo_usado}</div>', unsafe_allow_html=True)
            
        if msg.get("fuentes"):
            st.markdown('<div class="source-container" style="margin-top: -5px; margin-bottom: 10px;">', unsafe_allow_html=True)
            for fuente in msg["fuentes"]:
                st.markdown(f"<span class='source-badge'>📄 {fuente}</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        col_acc1, col_acc2, col_acc3, col_acc4, _ = st.columns([1.5, 2, 0.7, 0.7, 3.1])

        with col_acc1:
            texto_seguro = msg["content"].replace("\n", " ").replace("`", "\\`").replace('"', '\\"')
            boton_html = f"""
            <button onclick="navigator.clipboard.writeText(`{texto_seguro}`)"
                style="background-color: #ffffff; color: #334155; border: 1px solid #CBD5E1;
                padding: 6px 12px; border-radius: 6px; font-size: 14px; font-weight: 500; cursor: pointer; display: inline-flex; align-items: center; gap: 4px;">
                📋 Copiar Respuesta
            </button>
            """
            st.markdown(boton_html, unsafe_allow_html=True)

        with col_acc2:
            texto_descarga = f"--- REPORTE DE CONSULTA - AZESORA AI ---\n\n{msg['content']}\n\nFuentes: {', '.join(msg.get('fuentes', ['Ninguna']))}"
            st.download_button(
                label="📥 Exportar Reporte",
                data=texto_descarga,
                file_name="reporte_azesora.txt",
                mime="text/plain",
                key=f"dl_{msg['tiempo']}"
            )

        pregunta_asociada = st.session_state.mensajes[idx - 1]["content"] if idx > 0 else ""

        if msg.get("feedback"):
            emoji_feedback = "👍" if msg["feedback"] == "positivo" else "👎"
            with col_acc3:
                st.markdown(f"<span title='Feedback registrado'>{emoji_feedback}</span>", unsafe_allow_html=True)
        else:
            with col_acc3:
                if st.button("👍", key=f"fb_pos_{idx}"):
                    msg["feedback"] = "positivo"
                    registrar_feedback(pregunta_asociada, msg["content"], "positivo")
                    st.rerun()
            with col_acc4:
                if st.button("👎", key=f"fb_neg_{idx}"):
                    msg["feedback"] = "negativo"
                    registrar_feedback(pregunta_asociada, msg["content"], "negativo")
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

# Input de consulta
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        nueva_pregunta = st.text_input("Escribe tu consulta financiera...", placeholder="Hazme una pregunta sobre comisiones, impuestos o brokers...", label_visibility="collapsed")
    with col_btn:
        enviar = st.form_submit_button(label="🔍 Consultar")

# Procesamiento de la nueva petición
if enviar and nueva_pregunta:
    historial_previo = list(st.session_state.mensajes)
    
    st.session_state.mensajes.append({"role": "user", "content": nueva_pregunta})
    if nueva_pregunta not in st.session_state.historial_titulos:
        titulo_corto = nueva_pregunta if len(nueva_pregunta) <= 30 else nueva_pregunta[:27] + "..."
        st.session_state.historial_titulos.append(titulo_corto)
        
    pais_enviar = None if filtro_pais == "Todos los países" else filtro_pais
    with st.spinner("🧠 Consultando la base de datos vectorial..."):
        try:
            resultado = consultar_azesora(nueva_pregunta, historial_mensajes=historial_previo, filtro_pais=pais_enviar, modo_experto=modo_experto)
            
            st.session_state.mensajes.append({
                "role": "agent",
                "content": resultado["respuesta"],
                "fuentes": resultado["fuentes"],
                "tiempo": resultado["tiempo"],
                "modelo": resultado["modelo"]
            })
            st.rerun()
        except Exception as e:
            # Control elegante para el límite de cuota (Punto 5)
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                st.error("⚠️ Se ha agotado el límite de cuota diario de la API gratuita de Gemini (Máx 20 consultas/día). Por favor, espera unos minutos o cambia a una clave con plan de pago en tu archivo .env.")
            else:
                st.error(f"Hubo un error con el LLM: {str(e)}")

st.markdown("<br><br><hr><center style='color: #94A3B8; font-size: 11px;'>Azesora AI © 2026 | Desarrollado con arquitectura RAG Local persistida</center>", unsafe_allow_html=True)