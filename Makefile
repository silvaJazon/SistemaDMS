# Documentação: Makefile para simplificar os comandos Docker e Docker Compose do projeto FluxoAI.

# --- Variáveis ---
# Define o nome da imagem Docker. Use o seu username do Docker Hub.
IMAGE_NAME := jazonbatista/fluxoai-servico-ia:latest
# Define o caminho para o ficheiro docker-compose.yml (relativo a este Makefile).
COMPOSE_FILE := docker-compose.yml
# Define o nível de log padrão. Pode ser substituído na linha de comando (ex: make up LOG_LEVEL=DEBUG)
LOG_LEVEL ?= INFO

# --- Alvos Principais ---

# Alvo para construir a imagem Docker do serviço de IA.
.PHONY: build
build:
	@echo ">>> Construindo a imagem $(IMAGE_NAME)..."
	# Executa o build a partir do contexto da pasta 'servico_ia'.
	docker build -t $(IMAGE_NAME) ./servico_ia

# Alvo para iniciar os serviços definidos no docker-compose.yml em background (detached).
.PHONY: up
up:
	@echo ">>> Iniciando os serviços definidos em $(COMPOSE_FILE)... (Nível de Log: $(LOG_LEVEL))"
	# Exporta a variável LOG_LEVEL para que o docker compose a possa usar.
	# O docker compose vai ler LOG_LEVEL e passá-la para o environment do contentor.
	export LOG_LEVEL=$(LOG_LEVEL); docker compose -f $(COMPOSE_FILE) up -d

# Alvo para parar e remover os contentores, redes e volumes definidos no docker-compose.yml.
.PHONY: down
down:
	@echo ">>> Parando e removendo os serviços definidos em $(COMPOSE_FILE)..."
	docker compose -f $(COMPOSE_FILE) down

# Alvo combinado: Para o serviço (se estiver a correr), constrói a imagem e inicia novamente.
.PHONY: prod-up-build
prod-up-build: down build up
	@echo ">>> Imagem reconstruída e serviços reiniciados!"

# --- Alvos Auxiliares ---

# Alvo para ver os logs dos serviços em tempo real.
.PHONY: logs
logs:
	@echo ">>> Mostrando logs em tempo real (Pressione Ctrl+C para sair)..."
	docker compose -f $(COMPOSE_FILE) logs -f

# Alvo para remover a imagem Docker construída localmente.
.PHONY: clean-image
clean-image:
	@echo ">>> Removendo a imagem Docker $(IMAGE_NAME)..."
	docker rmi $(IMAGE_NAME)

# Alvo para remover o cache de build do Docker (útil para forçar reconstrução).
.PHONY: clean-cache
clean-cache:
	@echo ">>> Limpando o cache de build do Docker..."
	docker builder prune -f

# Alvo para uma limpeza mais completa do Docker (remove contentores parados, redes não usadas, etc.).
.PHONY: clean-all
clean-all: down
	@echo ">>> Limpando recursos Docker não utilizados (contentores, redes, imagens penduradas)..."
	docker system prune -f
	@echo ">>> Limpeza concluída."

# --- Ajuda ---
# Alvo para mostrar os comandos disponíveis.
.PHONY: help
help:
	@echo "Comandos disponíveis:"
	@echo "  make build          - Constrói a imagem Docker do serviço de IA."
	@echo "  make up             - Inicia os serviços em background (Nível Log padrão: INFO)."
	@echo "  make up LOG_LEVEL=<NIVEL> - Inicia os serviços com um nível de log específico (DEBUG, WARNING, ERROR)."
	@echo "  make down           - Para e remove os serviços."
	@echo "  make prod-up-build  - Para, reconstrói a imagem e reinicia os serviços."
	@echo "  make logs           - Mostra os logs dos serviços em tempo real."
	@echo "  make clean-image    - Remove a imagem Docker construída localmente."
	@echo "  make clean-cache    - Limpa o cache de build do Docker."
	@echo "  make clean-all      - Para os serviços e limpa recursos Docker não utilizados."
	@echo "  make help           - Mostra esta ajuda."

# Define 'help' como o alvo padrão (o que é executado se apenas 'make' for digitado).
.DEFAULT_GOAL := help

