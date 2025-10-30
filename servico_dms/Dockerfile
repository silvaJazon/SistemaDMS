# ========================================================================
# Documentação: Dockerfile para o Serviço DMS (Versão Simplificada)
# (REMOVIDO Dlib - Apenas MediaPipe + YOLO)
# ========================================================================

# --- Estágio Base ---
FROM python:3.9-slim-bookworm AS base
WORKDIR /app

# --- Estágio de Dependências ---

# Copia o arquivo de dependências
COPY requirements.txt .

# Dependências de Sistema Simplificadas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Atualiza pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instala dependências do requirements.txt (MediaPipe, YOLO, Torch)
RUN pip install --no-cache-dir -r requirements.txt

# --- Estágio Final ---

# Cria diretório de alertas e copia templates
RUN mkdir /app/alerts
COPY templates/ ./templates/

# Copia os arquivos do projeto
COPY app.py .
COPY dms_base.py .
COPY dms_mediapipe.py .
COPY camera_thread.py .
COPY event_handler.py .

# ================== ALTERAÇÃO (Modelo 's') ==================
COPY models/yolov8s.pt .
# ==========================================================

# Expõe a porta padrão do Flask
EXPOSE 5000

# Comando de inicialização
CMD ["python", "app.py"]