# Documentação:
# Etapa 1: Definir a nossa imagem base.
# Usamos uma imagem oficial do Python, versão 3.9, baseada no Debian Bullseye.
# A tag "slim" significa que é uma versão mais leve, ideal para produção.
FROM python:3.9-slim-bullseye

# Documentação:
# Define o diretório de trabalho dentro do contentor.
# Todos os comandos a seguir serão executados a partir desta pasta.
WORKDIR /app

# Documentação:
# Copia o ficheiro de requisitos do nosso PC para dentro do contentor.
COPY requirements.txt .

# Documentação:
# Instala as dependências do sistema necessárias para o OpenCV.
# Bibliotecas como libgl1 são cruciais para o processamento de imagem.
# "--no-install-recommends" torna a instalação mais leve.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Documentação:
# Instala as bibliotecas Python definidas no requirements.txt.
# Usamos "--no-cache-dir" para manter a imagem final mais pequena.
RUN pip install --no-cache-dir -r requirements.txt

# Documentação:
# Copia todo o resto do nosso código (o script Python) do nosso PC para dentro do contentor.
COPY . .

# Documentação:
# Define o comando que será executado quando o contentor iniciar.
# Neste caso, ele vai executar o nosso script principal com o Python.
CMD ["python", "main.py"]

