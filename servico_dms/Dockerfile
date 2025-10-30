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
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Verifica a versão do CMake (deve ser >= 3.25.1)
RUN which cmake && cmake --version

# (NOVO) Atualiza pip, setuptools e wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instala dependências do requirements.txt (SEM dlib, SEM cmake)
# (ALTERADO) O Dlib e o MediaPipe estão agora no requirements.txt (se necessário)
# (MAS o YOLO/ultralytics será instalado)
RUN pip install --no-cache-dir -r requirements.txt

# Instala o pybind11 (necessário para compilar o dlib)
RUN pip install --no-cache-dir --no-build-isolation pybind11

# Instala o dlib separadamente com CMake do sistema
RUN CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" \
    pip install --no-cache-dir --no-build-isolation dlib==19.24.2


# --- Estágio de Download do Modelo (Dlib) ---

# Descarrega o modelo de 68 pontos de referência faciais
RUN curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o landmarks.dat.bz2 && \
    bzip2 -d landmarks.dat.bz2 && \
    mv landmarks.dat shape_predictor_68_face_landmarks.dat

# (NOVO) Pré-download (opcional) do modelo YOLOv8n
# Isto evita que o download ocorra no primeiro arranque do container
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# --- Estágio Final ---

# Cria diretório de alertas e copia templates
RUN mkdir /app/alerts
COPY templates/ ./templates/

# (NOVO) Copia os novos arquivos do projeto
COPY app.py .
COPY dms_base.py .
COPY dms_core.py .
COPY dms_mediapipe.py .
COPY camera_thread.py .
COPY event_handler.py .
# Expõe a porta padrão do Flask
EXPOSE 5000

# Comando de inicialização
CMD ["python", "app.py"]