import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def crear_datos_demo():
    print("🚀 Iniciando la generación de la base de conocimientos para Azesora AI...")
    
    # 1. Asegurar la existencia de las carpetas
    os.makedirs("data_source/chile", exist_ok=True)
    os.makedirs("data_source/global", exist_ok=True)
    
    # 2. GENERAR ARCHIVO MARKDOWN (Global)
    md_content = """# Guía Operativa Global: Interactive Brokers (IBKR)
## 1. Resumen Ejecutivo
Interactive Brokers es un broker regulado en EE. UU. (SEC y FINRA) que permite operar a inversores internacionales de múltiples países de Latinoamérica, ofreciendo acceso directo a acciones, ETFs y opciones.

## 2. Estructura de Comisiones
* **Comisión por Operación:** USD 0.005 por acción con un mínimo de USD 1.00 por orden en la estructura fija.
* **Depósito Mínimo:** USD 0.
* **Mantenimiento de Cuenta:** USD 0 (eliminado globalmente).

## 3. Métodos de Fondeo y Retiro
* **Fondeo Internacional:** Transferencia bancaria Wire (SWIFT). Demora entre 24 y 48 horas hábiles.
* **Retiros:** Un retiro gratuito al mes calendario. Los retiros adicionales tienen un costo fijo de USD 10 por transferencia.
"""
    with open("data_source/global/interactive_brokers_guide.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("✅ Archivo Markdown generado en data_source/global/")

    # 3. GENERAR ARCHIVO CSV (Global)
    csv_data = {
        "BROKER": ["Zesty", "Interactive Brokers", "Hapi"],
        "DEPOSITO_MINIMO_USD": [1.0, 0.0, 1.0],
        "COMISION_COMPRA": ["0.0% (Zero Fee)", "USD 0.005 por accion", "USD 0.10 por orden"],
        "RETIRO_MINIMO_USD": [2.0, 10.0, 5.0],
        "SOPORTE_ESPANOL": ["Si", "Si (Limitado)", "Si"]
    }
    df = pd.DataFrame(csv_data)
    df.to_csv("data_source/comparativa_comisiones.csv", index=False, encoding="utf-8")
    print("✅ Archivo CSV generado en data_source/")

    # 4. GENERAR ARCHIVO PDF (Chile)
    pdf_path = "data_source/chile/regulacion_impuestos_chile.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Estilos personalizados
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, spaceAfter=12)
    h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=14, spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=8)
    
    story = []
    story.append(Paragraph("Marco Tributario y Operativo para Inversiones en Chile (2026)", title_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("1. Declaración de Ganancias de Capital", h2_style))
    story.append(Paragraph("Los inversores residentes en Chile que operen en el extranjero a través de plataformas como Zesty o Interactive Brokers deben declarar sus rentas de fuente extranjera. Las ganancias de capital derivadas de la compraventa de acciones en EE. UU. no se acogen al artículo 107 de la LIR chilena, por lo que tributan bajo el Impuesto Global Complementario (IGC) sobre la base de la utilidad percibida.", body_style))
    
    story.append(Paragraph("2. Formulario W-8BEN e Impuestos en EE. UU.", h2_style))
    story.append(Paragraph("Para evitar la doble tributación sobre dividendos de empresas estadounidenses, los usuarios chilenos deben firmar el Formulario W-8BEN a través de su broker. Gracias a esto, la retención en origen (IRS) aplicada a los dividendos se reduce del 30% estándar a una tasa preferencial del 15%, la cual puede utilizarse posteriormente como crédito contra los impuestos en Chile bajo los límites legales del Servicio de Impuestos Internos (SII).", body_style))
    
    story.append(Paragraph("3. Fondeo Local y Cuentas de Origen", h2_style))
    story.append(Paragraph("Para fondear cuentas en brokers digitales ágiles como Zesty, se permiten transferencias electrónicas locales directas en pesos chilenos (CLP), realizando la conversión cambiaria automática a dólares americanos (USD). Los retiros deben ser dirigidos exclusivamente a una cuenta bancaria a nombre del mismo titular de la cuenta de inversión para cumplir con las normativas locales de prevención de lavado de activos.", body_style))
    
    doc.build(story)
    print("✅ Archivo PDF generado en data_source/chile/")
    print("\n🎉 ¡Base de conocimientos inicial creada con éxito!")

if __name__ == "__main__":
    crear_datos_demo()