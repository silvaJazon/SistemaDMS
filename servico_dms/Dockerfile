# Documentação: Dockerfile para o Serviço DMS
# Versão focada apenas em Dlib (Sonolência/Distração)
# NOVO: Adiciona a pasta de templates e a pasta de alertas

# --- Estágio Base ---
FROM python:3.9-slim-bullseye AS base
WORKDIR /app

# --- Estágio de Dependências ---
COPY requirements.txt .

# Instala dependências de sistema
# (NOVO) Adicionamos 'wget' para descarregar o binário do CMake
# (NOVO) Removemos 'cmake' e 'cmake-data' do apt-get para evitar conflitos
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
        wget \
    && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# (NOVO) Passo 1: Instalação Manual do CMake (A "Bomba Atómica")
# Descarrega o binário pré-compilado para aarch64 (ARM) e extrai-o
# diretamente para /usr/local, que tem prioridade no PATH.
RUN wget -qO- "https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-aarch64.tar.gz" | \
    tar --strip-components=1 -xz -C /usr/local

# (NOVO) Passo 2: Força o PATH (para garantir)
ENV PATH="/usr/local/bin:${PATH}"

# (NOVO) Passo 3: Verifica a versão do CMake
# Isto TEM de mostrar 3.27.7 agora.
RUN which cmake && cmake --version

# (NOVO) Passo 4: Instala o resto das dependências (SEM O DLIB)
# O 'cmake' no requirements.txt será "ignorado" (Requirement already satisfied)
RUN pip install --no-cache-dir -r requirements.txt

# (NOVO) Passo 5: Instala o 'dlib' SEPARADAMENTE
# Forçamos o processo de build do pip a usar o executável correto do cmake
RUN CMAKE_EXECUTABLE=/usr/local/bin/cmake pip install --no-cache-dir dlib==19.24.2

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

