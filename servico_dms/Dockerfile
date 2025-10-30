# ========================================================================
# Documentação: Dockerfile para o Serviço DMS
# Baseado em "bookworm" (Debian 12) para acesso a um CMake moderno.
# ========================================================================

# --- Estágio Base ---
FROM python:3.9-slim-bookworm AS base
WORKDIR /app

# --- Estágio de Dependências ---

# Copia o arquivo de dependências
COPY requirements.txt .
# Instala dependências de sistema (incluindo cmake 3.25.1)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        g++ \
        build-essential \
        curl \
        bzip2 \
        pkg-config \
        libx11-dev \
        libboost-dev \
        libboost-python-dev \
        libopenblas-dev \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get autove -y && \
    rm -rf /var/lib/apt/lists/*

# Verifica a versão do CMake (deve ser >= 3.25.1)
RUN which cmake && cmake --version

# Atualiza pip, setuptools e wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instala dependências do requirements.txt (incluindo ultralytics)
RUN pip install --no-cache-dir -r requirements.txt

# Instala o pybind11 (necessário para compilar o dlib)
RUN pip install --no-cache-dir --no-build-isolation pybind11

# Instala o dlib separadamente com CMake do sistema
RUN CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" \
    pip install --no-cache-dir --no-build-isolation dlib==19.24.2

# --- (REMOVIDO) Estágios de Download (agora locais) ---

# --- Estágio Final ---

# Cria diretório de alertas e copia templates
RUN mkdir /app/alerts
COPY templates/ ./templates/

# Copia os arquivos do projeto
COPY app.py .
COPY dms_base.py .
COPY dms_core.py .
COPY dms_mediapipe.py .
COPY camera_thread.py .
COPY event_handler.py .

# ================== ALTERAÇÃO ==================
# (NOVO) Copia os modelos da pasta local 'models/' para a raiz do app '.'
# O Dockerfile é executado a partir de 'servico_dms', por isso o caminho é 'models/...'
COPY models/shape_predictor_68_face_landmarks.dat .
COPY models/yolov8n.pt .
# ===============================================

# Expõe a porta padrão do Flask
EXPOSE 5000

# Comando de inicialização
CMD ["python", "app.py"]