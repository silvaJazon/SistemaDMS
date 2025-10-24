# Documentação: Dockerfile para o Serviço de Captura de Vídeo (FluxoAI)

# Usa a mesma imagem base Python que os outros serviços para consistência
FROM python:3.9-slim-bullseye

# Define o diretório de trabalho dentro do contentor
WORKDIR /app

# Copia apenas o ficheiro de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências de sistema necessárias para OpenCV e outras bibliotecas
# (libgl1 é necessária mesmo para a versão headless em alguns casos)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o script principal da aplicação
COPY main.py .

# Comando padrão para executar a aplicação quando o contentor iniciar
CMD ["python", "main.py"]
