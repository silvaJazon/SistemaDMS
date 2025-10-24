# Documentação: Dockerfile para o Serviço de IA do Projeto FluxoAI

# Etapa 1: Base
# Usamos uma imagem oficial do Python, versão 3.9, otimizada (slim)
# e baseada no Debian Bullseye, que é compatível com o Raspberry Pi OS.
FROM python:3.9-slim-bullseye

# Documentação:
# Define o diretório de trabalho dentro do contentor.
# Todos os comandos seguintes serão executados a partir daqui.
WORKDIR /app

# Documentação:
# Copia o ficheiro de requisitos do nosso PC para dentro do contentor.
COPY requirements.txt .

# Documentação:
# Instala as dependências do sistema necessárias para o OpenCV e o wget.
# O OpenCV precisa de algumas bibliotecas gráficas (mesmo a versão headless).
# "--no-install-recommends" torna a instalação mais leve.
# "rm -rf /var/lib/apt/lists/*" limpa o cache do apt para reduzir o tamanho da imagem.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Documentação:
# Instala as bibliotecas Python definidas no requirements.txt.
# Usamos "--no-cache-dir" para manter a imagem final mais pequena.
RUN pip install --no-cache-dir -r requirements.txt

# Documentação:
# Descarrega o modelo TFLite e o ficheiro de etiquetas diretamente para o diretório /app.
RUN wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite -O model.tflite && \
    wget https://storage.googleapis.com/download.tensorflow.org/models/object_detection/coco_labels.txt -O labels.txt

# Documentação:
# Copia todo o resto do nosso código (o script main.py)
# da pasta atual no PC para o diretório /app dentro do contentor.
COPY . .

# Documentação:
# Expõe a porta 5000, que o nosso servidor Flask vai usar para o streaming web.
EXPOSE 5000

# Documentação:
# Comando que será executado quando o contentor iniciar.
# Executa o nosso script Python principal.
CMD ["python", "main.py"]

