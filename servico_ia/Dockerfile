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
# - libgl1-mesa-glx, libglib2.0-0: Bibliotecas essenciais para o OpenCV funcionar sem interface gráfica.
# Usamos "--no-install-recommends" para manter a instalação mais leve.
# Limpamos o cache do apt no final para reduzir o tamanho da imagem.
# NOTA: Removemos curl pois não é mais necessário durante o build.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Documentação:
# Instala as bibliotecas Python definidas no requirements.txt.
# Usamos "--no-cache-dir" para manter a imagem final mais pequena.
RUN pip install --no-cache-dir -r requirements.txt

# --- Estágio Final ---
# Copia todo o código E os ficheiros do modelo (que já descarregamos manualmente) para dentro do contentor.
# O Dockerfile assume que model.tflite e labels.txt estão na mesma pasta que ele.
COPY . .

# Documentação:
# Expõe a porta 5000, que o nosso servidor Flask vai usar para o stream de vídeo.
EXPOSE 5000

# Documentação:
# Comando que será executado quando o contentor iniciar.
# Executa o nosso script Python principal.
CMD ["python", "main.py"]

