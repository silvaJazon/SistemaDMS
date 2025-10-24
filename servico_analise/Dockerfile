# Documentação: Dockerfile para o Serviço de Análise do FluxoAI

# Usa a mesma imagem base leve do Python que os outros serviços
FROM python:3.9-slim-bullseye

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia apenas o ficheiro de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências Python
# Usamos "--no-cache-dir" para manter a imagem final mais pequena.
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação (o script main.py)
COPY main.py .

# Define o comando padrão para executar quando o container iniciar
CMD ["python", "main.py"]
