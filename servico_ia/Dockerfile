# Documentação: Fase 1 - A Base
# Usamos uma imagem base do Python 3.9 na sua versão "slim" para ser mais leve.
# A especificação "-bullseye" refere-se à versão do Debian, garantindo consistência.
FROM python:3.9-slim-bullseye

# Documentação: Define o diretório de trabalho dentro do contentor.
# Todos os comandos a seguir serão executados a partir desta pasta.
WORKDIR /app

# Documentação: Copia o ficheiro de requisitos do nosso PC para dentro do contentor.
COPY requirements.txt .

# Documentação: Instala as dependências do sistema necessárias para o OpenCV.
# O OpenCV precisa destas bibliotecas para manipular vídeo e janelas gráficas.
# "--no-install-recommends" torna a instalação mais leve.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Documentação: Instala as bibliotecas Python definidas no requirements.txt.
# Usamos "--no-cache-dir" para manter a imagem final mais pequena.
RUN pip install --no-cache-dir -r requirements.txt

# Documentação: Copia todo o resto do nosso código (o script Python que vamos criar)
# da pasta local para o diretório de trabalho dentro do contentor.
COPY . .

# Documentação: O comando que será executado quando o contentor arrancar.
# Irá executar o nosso script principal, "main.py".
CMD ["python", "main.py"]
