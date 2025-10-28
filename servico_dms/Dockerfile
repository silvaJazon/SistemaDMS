# Documentação: Dockerfile para o Serviço DMS
# Versão focada apenas em Dlib (Sonolência/Distração)
# NOVO: Adiciona a pasta de templates e a pasta de alertas

# --- Estágio Base ---
FROM python:3.9-slim-bullseye AS base
WORKDIR /app

# --- Estágio de Dependências ---
COPY requirements.txt .

# Instala dependências de sistema
# Adicionado libsm6 e libxext6 para opencv-python-headless
# Removido apt-get purge de cmake e cmake-data, pois não são mais necessários
RUN apt-get update && \
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
        wget \
        libsm6 \
        libxext6 \
    && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Passo 1: Instalação Manual do CMake (A "Bomba Atómica")
# Descarrega o binário pré-compilado para aarch64 (ARM) e extrai-o
# diretamente para /usr/local, que tem prioridade no PATH.
# O problema principal é a arquitetura 'aarch64' (ARM) que o usuário está usando.
# O pip tenta compilar o dlib e precisa do cmake.
# O dlib requer o cmake para compilar.
# Se o usuário está em 'aarch64', o binário de cmake que ele está baixando
# é o correto, mas a maneira de instalar o dlib é o problema.
# Vou manter a instalação manual do cmake, pois o usuário a configurou.
# O problema é que o pip não vai ignorar o 'cmake' no requirements.txt, ele vai tentar instalar.
# A solução é remover 'cmake' do requirements.txt.
RUN wget -qO- "https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-aarch64.tar.gz" | \
    tar --strip-components=1 -xz -C /usr/local

# Passo 2: Força o PATH (para garantir)
ENV PATH="/usr/local/bin:${PATH}"

# Passo 3: Verifica a versão do CMake
RUN which cmake && cmake --version

# Passo 4: Instala o resto das dependências.
# 'cmake' foi removido do requirements.txt.
# A instalação do dlib e opencv-python-headless requerem o cmake que foi instalado manualmente.
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
# Assumindo que a pasta templates existe no contexto de build
COPY templates/ ./templates/

# Copia os scripts Python da aplicação
# Assumindo que os arquivos existem no contexto de build
COPY app.py .
COPY dms_core.py .
COPY camera_thread.py .
COPY event_handler.py .
# Expõe a porta do Flask
EXPOSE 5000

# Comando de execução
CMD ["python", "app.py"]
