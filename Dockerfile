FROM python:3.11-slim

WORKDIR /app

# Herramientas de compilación y curl (usado por el HEALTHCHECK). Sin esto, cualquier
# dependencia transitiva sin wheel precompilado para esta plataforma fallaría al instalar.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalamos primero la build CPU-only de torch: el wheel por defecto de PyPI en Linux arrastra
# ~2.7GB de librerias NVIDIA/CUDA inutiles sin GPU (inflaba la imagen a 3.3GB y volvia lentisimo
# cualquier build/transferencia en maquinas pequenas). No usamos GPU, asi que no hace falta.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
