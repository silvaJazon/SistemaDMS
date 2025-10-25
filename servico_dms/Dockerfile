# Documentação: Dockerfile para o Serviço DMS (Fase 2: Refatorado)
# Instala Python, OpenCV e Dlib (compilando via pip)

# --- Estágio Base ---
FROM python:3.9-slim-bullseye AS base
WORKDIR /app

# --- Estágio de Dependências ---
COPY requirements.txt .

# Instala dependências de sistema
# Essencial: cmake e g++ são necessários para COMPILAR o 'dlib' via pip
# Essencial: bzip2 é necessário para descompactar o modelo de landmarks
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

# Instala dependências Python (agora inclui o dlib)
RUN pip install --no-cache-dir -r requirements.txt

# --- Estágio de Download do Modelo (Dlib) ---
# Descarrega o modelo de 68 pontos de referência faciais
RUN curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o landmarks.dat.bz2 && \
    bzip2 -d landmarks.dat.bz2 && \
    mv landmarks.dat shape_predictor_68_face_landmarks.dat

# --- Estágio Final ---
# Copia os novos scripts da aplicação refatorada
COPY app.py .
COPY dms_core.py .
COPY camera_thread.py .

# Expõe a porta do Flask
EXPOSE 5000

# Comando de execução (agora usa app.py)
CMD ["python", "app.py"]
