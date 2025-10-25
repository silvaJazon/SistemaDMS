# Documentação: Dockerfile para o Serviço DMS (Fase 3.2: MobileNet-SSD)
# Instala Python, OpenCV, Dlib e descarrega os modelos MobileNet-SSD.

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
# O Dlib ainda descarregamos porque é fiável
RUN curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o landmarks.dat.bz2 && \
    bzip2 -d landmarks.dat.bz2 && \
    mv landmarks.dat shape_predictor_68_face_landmarks.dat

# --- Estágio de Download do Modelo (MobileNet-SSD) ---
# Em vez de YOLO, descarregamos o MobileNet-SSD (mais leve)
RUN curl -L "https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/MobileNetSSD_deploy.prototxt" -o MobileNetSSD_deploy.prototxt && \
    curl -L "https://github.com/opencv/opencv_extra/raw/3.4.0/testdata/dnn/MobileNetSSD_deploy.caffemodel" -o MobileNetSSD_deploy.caffemodel

# --- Estágio Final ---
# Copia os scripts da aplicação
COPY app.py .
COPY dms_core.py .
COPY camera_thread.py .

# Expõe a porta do Flask
EXPOSE 5000

# Comando de execução (agora usa app.py)
CMD ["python", "app.py"]

