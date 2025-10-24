# Documentação: Dockerfile para o Serviço de IA do Projeto FluxoAI (Fase 3)
# Baseado em Python 3.9, otimizado para Raspberry Pi (arm64)

# --- Estágio Base ---
# Usamos a imagem oficial Python 3.9 slim (Debian Bullseye), compatível com arm64.
FROM python:3.9-slim-bullseye AS base

# Define o diretório de trabalho dentro do contentor.
WORKDIR /app

# --- Estágio de Dependências ---
# Copia apenas o ficheiro de requisitos primeiro para aproveitar o cache do Docker.
COPY requirements.txt .

# Documentação:
# Instala as dependências de sistema necessárias ANTES das dependências Python.
# - curl: Ferramenta para descarregar ficheiros (necessária para o modelo).
# - libgl1-mesa-glx, libglib2.0-0: Bibliotecas essenciais para o OpenCV funcionar sem interface gráfica.
# Usamos "--no-install-recommends" para manter a instalação mais leve.
# Limpamos o cache do apt no final para reduzir o tamanho da imagem.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        unzip \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Documentação:
# Instala as bibliotecas Python definidas no requirements.txt.
# Usamos "--no-cache-dir" para manter a imagem final mais pequena.
RUN pip install --no-cache-dir -r requirements.txt

# Documentação:
# Descarrega o modelo TFLite (do TF Hub) e o ficheiro de etiquetas diretamente para o diretório /app.
# A flag -L segue redirecionamentos, -f falha em caso de erro do servidor, -o especifica o nome do ficheiro de saída.
RUN curl -fL -o model.zip https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip && \
    unzip model.zip && \
    mv detect.tflite model.tflite && \
    curl -fL -o labels.txt https://storage.googleapis.com/download.tensorflow.org/data/mscoco_label_map.txt && \
    rm model.zip

# --- Estágio Final ---
# Copia todo o código Python (main.py) para dentro do contentor.
# NOTA: Não precisamos mais copiar model.tflite e labels.txt aqui, pois já foram descarregados.
COPY main.py .

# Documentação:
# Expõe a porta 5000, que o nosso servidor Flask vai usar para o stream de vídeo.
EXPOSE 5000

# Documentação:
# Comando que será executado quando o contentor iniciar.
# Executa o nosso script Python principal.
CMD ["python", "main.py"]

