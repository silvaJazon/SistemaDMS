# Documentação: Dockerfile para o Serviço de IA do Projeto FluxoAI (Monolítico YOLOv8)
# Baseado em Python 3.9, otimizado para Raspberry Pi (arm64)

# --- Estágio Base ---
FROM python:3.9-slim-bullseye AS base
WORKDIR /app

# --- Estágio de Dependências ---
COPY requirements.txt .

# Instala dependências de sistema (OpenCV) e ferramentas (curl, unzip)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# --- Estágio de Download do Modelo (YOLOv8s) ---
# Descarrega o modelo YOLOv8s TFLite (quantizado por inteiros)
RUN curl -L \
    "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-int8.tflite" \
    -o model.tflite

# Descarrega o ficheiro de etiquetas COCO (80 classes)
# Nota: O ID de 'person' neste ficheiro é 0, o que o nosso script espera.
RUN curl -L \
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco8.yaml" \
    -o coco.yaml && \
    # Extrai apenas os nomes das etiquetas do ficheiro yaml e guarda em labels.txt
    grep -oP '(?<=names:\n)([\s\S]*?)(?=\n\n)' coco.yaml | \
    sed -e 's/^[ ]*- //g' -e "s/'//g" > labels.txt && \
    rm coco.yaml

# --- Estágio Final ---
# Copia apenas o script principal (o resto já está na imagem)
COPY main.py .

# Expõe a porta do Flask
EXPOSE 5000

# Comando de execução
CMD ["python", "main.py"]

    

