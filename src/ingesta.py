import os
import pandas as pd
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def procesar_archivos_corporativos():
    documentos_finales = []
    
    # Configuramos el divisor de texto (Chunking) profesional
    # Dividirá en bloques de 600 caracteres con 100 de superposición
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    
    print("📂 Iniciando extracción y limpieza de documentos...")

    # 1. PROCESAR PDF (Chile)
    pdf_path = "data_source/chile/regulacion_impuestos_chile.pdf"
    if os.path.exists(pdf_path):
        reader = PdfReader(pdf_path)
        texto_pdf = ""
        for i, pagina in enumerate(reader.pages):
            texto_pdf += pagina.extract_text() + "\n"
        
        # Creamos los fragmentos del PDF
        chunks_pdf = text_splitter.split_text(texto_pdf)
        for chunk in chunks_pdf:
            doc = Document(
                page_content=chunk,
                metadata={"pais": "chile", "broker": "todos", "fuente": "regulacion_impuestos_chile.pdf"}
            )
            documentos_finales.append(doc)
        print(f"✅ PDF de Chile procesado. Creados {len(chunks_pdf)} fragmentos.")

    # 2. PROCESAR MARKDOWN (Global)
    md_path = "data_source/global/interactive_brokers_guide.md"
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            texto_md = f.read()
        
        chunks_md = text_splitter.split_text(texto_md)
        for chunk in chunks_md:
            doc = Document(
                page_content=chunk,
                metadata={"pais": "global", "broker": "interactive brokers", "fuente": "interactive_brokers_guide.md"}
            )
            documentos_finales.append(doc)
        print(f"✅ Markdown Global procesado. Creados {len(chunks_md)} fragmentos.")

    # 3. PROCESAR CSV (Tabular - Comparativa)
    csv_path = "data_source/comparativa_comisiones.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Convertimos cada fila del CSV en una frase estructurada comprensible para la IA
        for _, fila in df.iterrows():
            texto_fila = (
                f"El broker {fila['BROKER']} exige un depósito mínimo de {fila['DEPOSITO_MINIMO_USD']} dólares. "
                f"Su comisión por compra es de {fila['COMISION_COMPRA']}. "
                f"El monto mínimo de retiro es de {fila['RETIRO_MINIMO_USD']} dólares y "
                f"¿Tiene soporte en español?: {fila['SOPORTE_ESPANOL']}."
            )
            doc = Document(
                page_content=texto_fila,
                metadata={"pais": "global", "broker": str(fila['BROKER']).lower(), "fuente": "comparativa_comisiones.csv"}
            )
            documentos_finales.append(doc)
        print(f"✅ CSV Comparativo procesado. Creadas {len(df)} filas indexadas.")

    print(f"\n🎉 ¡Procesamiento completado! Total de fragmentos listos para embeddings: {len(documentos_finales)}")
    return documentos_finales

if __name__ == "__main__":
    procesar_archivos_corporativos()