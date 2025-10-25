# Documentação: Dockerfile para o Serviço DMS (Fase 1: Sonolência)
# Instala Python, OpenCV e Dlib (essencial para landmarks faciais)

# --- Estágio Base ---
FROM python:3.9-slim-bullseye AS base
WORKDIR /app

# --- Estágio de Dependências ---
COPY requirements.txt .

# Instala dependências de sistema
# Essencial: cmake e g++ são necessários para o 'dlib' (se fosse compilado)
# Essencial: python3-dlib instala o dlib otimizado do Debian para o Python 3.9
# Essencial: bzip2 é necessário para descompactar o modelo de landmarks
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        python3-dlib \
        curl \
        bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python (exceto dlib, que já veio do apt)
RUN pip install --no-cache-dir -r requirements.txt

# --- Estágio de Download do Modelo (Dlib) ---
# Descarrega o modelo de 68 pontos de referência faciais
RUN curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o landmarks.dat.bz2 && \
    bzip2 -d landmarks.dat.bz2 && \
    mv landmarks.dat shape_predictor_68_face_landmarks.dat

# --- Estágio Final ---
# Copia apenas o script principal
COPY main.py .

# Expõe a porta do Flask
EXPOSE 5000

# Comando de execução
CMD ["python", "main.py"]
