# Documentação: Dockerfile para o Serviço de IA do Projeto FluxoAI (Monolítico YOLOv8)
# Baseado em Python 3.9, otimizado para Raspberry Pi (arm64)

# --- Estágio Base ---
FROM python:3.9-slim-bullseye AS base
WORKDIR /app

# --- Estágio de Dependências ---
# Copia apenas o ficheiro de requisitos primeiro para aproveitar o cache do Docker.
COPY requirements.txt .

# Instala dependências de sistema (OpenCV)
# NOTA: Removemos curl, pois os modelos são copiados localmente.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# --- Estágio Final ---
# Copia TODO o conteúdo local (main.py, requirements.txt, E os ficheiros .tflite e .txt
# que descarregamos manualmente) para dentro do contentor.
COPY . .

# Expõe a porta do Flask
EXPOSE 5000

# Comando de execução
CMD ["python", "main.py"]

# realize o download manual dos modelos e ficheiros necessários na pasta /fluxoai/servico_ia antes de construir o contentor Docker:
#curl -L "https://huggingface.co/qualcomm/YOLOv8-Detection/resolve/d9cfb77008a49ceae5ea6c023afd6786c2d1b155/Yolo-v8-Detection.tflite" -o model.tflite