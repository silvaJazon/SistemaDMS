# Documentação: Dockerfile para o Serviço DMS
# Versão focada apenas em Dlib (Sonolência/Distração)
# NOVO: Adiciona a pasta de templates e a pasta de alertas

# --- Estágio Base ---
FROM python:3.9-slim-bullseye AS base
WORKDIR /app

# --- Estágio de Dependências ---
COPY requirements.txt .

# Instala dependências de sistema
# Essencial: g++ é necessário para COMPILAR o 'dlib' via pip
# Essencial: bzip2 é necessário para descompactar o modelo de landmarks
# (NOVO) 'cmake' foi removido daqui. Ele agora será instalado via 'pip'
#          pelo 'requirements.txt', o que garante a versão correta.
RUN apt-get update && \
    apt-get remove -y cmake && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        g++ \
        build-essential \
        curl \
        bzip2 \
    && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Instala dependências Python (agora inclui o dlib)

# (NOVO) Passo 1: Instala o CMake PRIMEIRO e sozinho.
# Isto é crucial para garantir que ele esteja disponível para o build do dlib.
RUN pip install --no-cache-dir "cmake==3.27.7"

# (NOVO) Passo 2: Instala o resto das dependências.
# 'cmake' já está instalado (do passo anterior), então o pip vai saltá-lo
# e prosseguir para compilar o 'dlib' usando a versão correta.
RUN pip install --no-cache-dir -r requirements.txt

# --- Estágio de Download do Modelo (Dlib) ---
# Descarrega o modelo de 68 pontos de referência faciais
RUN curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o landmarks.dat.bz2 && \
    bzip2 -d landmarks.dat.bz2 && \
    mv landmarks.dat shape_predictor_68_face_landmarks.dat

# --- Estágio Final ---

# NOVO: Cria a pasta para guardar os alertas
RUN mkdir /app/alerts

# NOVO: Copia a pasta de templates HTML
COPY templates/ ./templates/

# Copia os scripts Python da aplicação
COPY app.py .
COPY dms_core.py .
COPY camera_thread.py .
COPY event_handler.py .
# Expõe a porta do Flask
EXPOSE 5000

# Comando de execução
CMD ["python", "app.py"]

