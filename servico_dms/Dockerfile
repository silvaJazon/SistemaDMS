# Documentação: Dockerfile para o Serviço DMS
# Versão focada apenas em Dlib (Sonolência/Distração)
# NOVO: Adiciona a pasta de templates e a pasta de alertas

# --- Estágio Base ---
FROM python:3.9-slim-bullseye AS base
WORKDIR /app

# --- Estágio de Dependências ---
COPY requirements.txt .

# Instala dependências de sistema
# Essencial: g++ (compilador C++) e bzip2 (para o modelo dlib)
# (NOVO) Adicionadas bibliotecas de desenvolvimento C++ cruciais para a compilação do dlib em aarch64
# (NOVO) 'cmake' e 'cmake-data' são removidos para evitar conflitos com a versão do pip.
RUN apt-get update && \
    apt-get purge -y cmake cmake-data && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        g++ \
        build-essential \
        curl \
        bzip2 \
        pkg-config \
        libx11-dev \
        libboost-dev \
        libboost-python-dev \
        libopenblas-dev \
    && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Instala dependências Python (agora inclui o dlib)

# (NOVO) Passo 1: Instala o CMake PRIMEIRO e sozinho.
# Isto é crucial para garantir que ele esteja disponível para o build do dlib.
RUN pip install --no-cache-dir "cmake==3.27.7"

# (NOVO) Passo 2: Força o PATH
# Adiciona o diretório de binários do pip (/usr/local/bin) ao início do PATH.
# Isto força qualquer subprocesso a encontrar o cmake (3.27.7)
# antes de encontrar qualquer versão antiga do sistema (em /usr/bin).
ENV PATH="/usr/local/bin:${PATH}"

# (NOVO) Passo 3: Verifica a versão do CMake (agora com o PATH forçado)
# Este passo vai mostrar-nos no log qual 'cmake' está no PATH e a sua versão.
RUN which cmake && cmake --version

# (NOVO) Passo 4: Instala o resto das dependências.
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

