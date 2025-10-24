# Makefile para o Projeto FluxoAI

# --- Variáveis ---
IMAGE_NAME=jazonbatista/fluxoai-servico-ia
IMAGE_TAG=latest
SERVICE_DIR=servico_ia
COMPOSE_FILE=docker-compose.yml

# --- Alvos Principais ---

.PHONY: build up down logs prod-up-build clean help

# Alvo Padrão (mostra ajuda)
default: help

# Constrói a imagem do serviço de IA
build:
	@echo ">>> Construindo a imagem $(IMAGE_NAME):$(IMAGE_TAG)..."
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) ./$(SERVICE_DIR)

# Inicia os serviços em background
up:
	@echo ">>> Iniciando os serviços definidos em $(COMPOSE_FILE)..."
	docker compose -f $(COMPOSE_FILE) up -d

# Para os serviços
down:
	@echo ">>> Parando os serviços definidos em $(COMPOSE_FILE)..."
	docker compose -f $(COMPOSE_FILE) down

# Mostra os logs do serviço de IA em tempo real
logs:
	@echo ">>> Mostrando logs do serviço $(SERVICE_DIR) (Pressione Ctrl+C para sair)..."
	docker compose -f $(COMPOSE_FILE) logs -f $(SERVICE_DIR)

# Alvo Combinado: Constrói a imagem e inicia os serviços
prod-up-build: build up
	@echo ">>> Imagem construída e serviços iniciados!"

# Remove imagens Docker penduradas (dangling)
clean:
	@echo ">>> Removendo imagens Docker não utilizadas..."
	docker image prune -f

# Mostra a ajuda
help:
	@echo "Comandos disponíveis:"
	@echo "  make build          - Constrói a imagem Docker do serviço de IA."
	@echo "  make up             - Inicia os serviços definidos no docker-compose.yml em background."
	@echo "  make down           - Para os serviços definidos no docker-compose.yml."
	@echo "  make logs           - Mostra os logs do serviço de IA em tempo real."
	@echo "  make prod-up-build  - Constrói a imagem e inicia os serviços."
	@echo "  make clean          - Remove imagens Docker não utilizadas."
	@echo "  make help           - Mostra esta ajuda."
