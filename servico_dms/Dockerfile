# Documentação: Dockerfile para o Serviço DMS
# Usando "bookworm" (Debian 12) para ter acesso a um 'cmake' moderno.

# --- Estágio Base ---
FROM python:3.9-slim-bookworm AS base
WORKDIR /app

# --- Estágio de Dependências ---
COPY requirements.txt .

# Instala dependências de sistema (incluindo cmake 3.25.1)
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

# Verifica a versão do CMake (deve ser 3.25.1)
RUN which cmake && cmake --version

# (NOVO) Atualiza pip e setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instala as dependências do requirements.txt (SEM dlib, SEM cmake)
RUN pip install --no-cache-dir -r requirements.txt

# Instala o pybind11 (dependência de build do dlib)
# Nota: Adicionamos --no-build-isolation aqui também por segurança
RUN pip install --no-cache-dir --no-build-isolation pybind11

# Instala o 'dlib' separadamente
# (NOVO) Usamos --no-build-isolation para forçar o uso do cmake do sistema
# Mantemos CMAKE_ARGS por precaução, embora possa não ser necessário agora.
RUN CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install --no-cache-dir --no-build-isolation dlib==19.24.2

# --- Estágio de Download do Modelo (Dlib) ---
# Descarrega o modelo de 68 pontos de referência faciais
RUN curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o landmarks.dat.bz2 && \
    bzip2 -d landmarks.dat.bz2 && \
    mv landmarks.dat shape_predictor_68_face_landmarks.dat

# --- Estágio Final ---
RUN mkdir /app/alerts
COPY templates/ ./templates/
COPY app.py .
COPY dms_core.py .
COPY camera_thread.py .
COPY event_handler.py .
EXPOSE 5000
CMD ["python", "app.py"]

