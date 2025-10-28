# Documentação: Dockerfile para o Serviço DMS
# --- Estágio Base ---
FROM python:3.9-slim-bookworm AS base
WORKDIR /app

# --- Estágio de Dependências ---
COPY requirements.txt .

# Instala dependências de sistema
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
        libglib2.0-0 \
    && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*


# Verifica a instalação do cmake
RUN which cmake && cmake --version

# Instala as dependências Python listadas no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Instala a versão específica do dlib
RUN pip install --no-cache-dir dlib==19.24.2

# --- Estágio de Download do Modelo (Dlib) ---
# Descarrega o modelo de 68 pontos de referência faciais
RUN curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o landmarks.dat.bz2 && \
    bzip2 -d landmarks.dat.bz2 && \
    mv landmarks.dat shape_predictor_68_face_landmarks.dat

# --- Estágio Final ---

# Cria a pasta para guardar os alertas
RUN mkdir /app/alerts

# Copia a pasta de templates HTML
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

