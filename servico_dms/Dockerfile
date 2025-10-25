# Documentação: Dockerfile para o Serviço DMS (Fase 3: Refatorado + YOLO)
# Instala Python, OpenCV, Dlib e descarrega modelos YOLO

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

# --- Estágio de Download do Modelo (YOLOv3-tiny) ---
# Descarrega o modelo de Deteção de Objetos (para telemóvel, etc.)
# CORREÇÃO: A usar links diretos do repositório AlexeyAB/darknet (mais estáveis)
RUN curl -L "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names" -o coco.names && \
    curl -L "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg" -o yolov3-tiny.cfg && \
    curl -L "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov3-tiny.weights" -o yolov3-tiny.weights

# --- Estágio Final ---
# Copia os novos scripts da aplicação refatorada
COPY app.py .
COPY dms_core.py .
COPY camera_thread.py .

# Expõe a porta do Flask
EXPOSE 5000

# Comando de execução (agora usa app.py)
CMD ["python", "app.py"]

