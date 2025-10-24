# Documentação: Dockerfile para o Serviço Web do FluxoAI

# Usa a mesma imagem base Python dos outros serviços
FROM python:3.9-slim-bullseye

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o ficheiro de requisitos
COPY requirements.txt .

# Instala as dependências de sistema (caso necessário no futuro, por agora vazio)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     # Pacotes adicionais aqui, se necessário \
#     && rm -rf /var/lib/apt/lists/*

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o script principal do serviço web
COPY main.py .

# Expõe a porta que o Flask vai usar (ex: 5000)
EXPOSE 5000

# Comando padrão para executar o script quando o container iniciar
CMD ["python", "main.py"]
