# Documentação: Dockerfile para o Serviço DMS
# Versão 3: Usa MobileNet-SSD e copia os modelos localmente (em vez de curl)

# --- Estágio Base ---
FROM python:3.9-slim-bullseye AS base
WORKDIR /app

# --- Estágio de Dependências ---
COPY requirements.txt .

# Instala dependências de sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        cmake \
        g++ \
        build-essential \
        curl \
        bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# --- Estágio de Download do Modelo (Dlib) ---
# Descarrega o modelo de 68 pontos de referência faciais
RUN curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o landmarks.dat.bz2 && \
    bzip2 -d landmarks.dat.bz2 && \
    mv landmarks.dat shape_predictor_68_face_landmarks.dat

# --- Estágio de Cópia dos Modelos (MobileNet-SSD) ---
# CORREÇÃO: Copia os modelos descarregados manualmente da pasta local
COPY MobileNetSSD_deploy.prototxt .
COPY MobileNetSSD_deploy.caffemodel .

# --- Estágio Final ---
# Copia os scripts da aplicação
COPY app.py .
COPY dms_core.py .
COPY camera_thread.py .

# Expõe a porta do Flask
EXPOSE 5000

# Comando de execução
CMD ["python", "app.py"]

