# ========================================================================
# Documentação: Dockerfile para o Serviço DMS (Versão Multi-Stage)
# (MODIFICADO) Otimizado para cache, separando 'base-requirements.txt'
# ========================================================================

# --- Estágio 1: Builder ---
FROM python:3.9-slim-bookworm AS builder
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Atualiza pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# --- (NOVA LÓGICA DE CACHE) ---

# 1. Instala o numpy primeiro (para o fix aarch64)
RUN pip install --no-cache-dir numpy

# 2. Copia e instala as bibliotecas PESADAS
# Esta camada SÓ será reconstruída se 'base-requirements.txt' mudar
COPY base-requirements.txt .
RUN pip install --no-cache-dir -r base-requirements.txt

# 3. Copia e instala as bibliotecas LEVES
# Esta camada SÓ será reconstruída se 'requirements.txt' mudar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Estágio 2: Final ---
FROM python:3.9-slim-bookworm AS final
WORKDIR /app

# Instala *apenas* as dependências de RUNTIME
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copia as dependências Python instaladas do estágio builder
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
# Copia os executáveis (pytest, flake8, black) do estágio builder
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Cria diretórios necessários
RUN mkdir /app/alerts
RUN mkdir /app/config

# Copia os arquivos do projeto
COPY templates/ ./templates/
COPY models/ ./models/
COPY app.py .
COPY dms_base.py .
COPY dms_mediapipe.py .
COPY camera_thread.py .
COPY event_handler.py .
COPY __init__.py .
COPY tests/ ./tests/
# (NOVO) Copia o novo arquivo de requisitos para a imagem final (apenas por referência)
COPY base-requirements.txt .

# Expõe a porta padrão do Flask
EXPOSE 5000

# Comando de inicialização
CMD ["python", "app.py"]