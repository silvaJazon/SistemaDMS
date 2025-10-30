# ========================================================================
# Documentação: Dockerfile para o Serviço DMS (Versão Multi-Stage)
# Otimizado para cache de dependências e imagem final limpa.
# ========================================================================

# --- Estágio 1: Builder ---
# Constrói o ambiente com as dependências
FROM python:3.9-slim-bookworm AS builder
WORKDIR /app

# Instala dependências do sistema necessárias para compilar pacotes Python (se houver)
# e as dependências de runtime do OpenCV (libgl1)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Atualiza pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copia e instala *apenas* as dependências
# Isso é cacheado pelo Docker e só será re-executado se requirements.txt mudar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Estágio 2: Final ---
# Cria a imagem final com o código da aplicação
FROM python:3.9-slim-bookworm AS final
WORKDIR /app

# Instala *apenas* as dependências de RUNTIME (muito mais rápido)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copia as dependências Python instaladas do estágio builder
# (AVISO: Este método copia TUDO do site-packages. Uma abordagem
# mais robusta usaria um venv, mas esta é mais simples para este caso.)
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/

# Cria diretórios necessários
RUN mkdir /app/alerts
RUN mkdir /app/config 

# Copia os arquivos do projeto
COPY templates/ ./templates/
COPY models/ ./models/
COPY app.py .
COPY dms_base.py .
COPY dms_mediapipe.py .
COPY camera_thread.py .
COPY event_handler.py .
COPY __init__.py . 

# Expõe a porta padrão do Flask
EXPOSE 5000

# Comando de inicialização
# (Usamos CMD para que possa ser facilmente sobrescrito
# por comandos 'make test' ou 'make lint')
CMD ["python", "app.py"]