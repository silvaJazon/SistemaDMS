# Documentação: Makefile para simplificar os comandos Docker e Docker Compose do projeto SistemaDMS.

# --- Variáveis ---
IMAGE_NAME := jazonbatista/sistema-dms:latest
COMPOSE_FILE := docker-compose.yml
LOG_LEVEL ?= INFO # Nível de log padrão (INFO)
# Nome do serviço único no docker-compose
SERVICO := servico_dms

# --- Alvos Principais ---

# Alvo para construir a imagem Docker.
.PHONY: build
build:
	@echo ">>> Construindo a imagem $(IMAGE_NAME)..."
	# Executa o build a partir do contexto da pasta do serviço.
	docker build -t $(IMAGE_NAME) ./$(SERVICO)

# Alvo para construir um serviço específico (make build servico=...)
build-%:
	@echo ">>> Construindo a imagem para o serviço $*..."
	docker build -t jazonbatista/fluxoai-$*:latest ./$*

# Alvo para iniciar os serviços em background.
.PHONY: up
up:
	@echo ">>> Iniciando os serviços definidos em $(COMPOSE_FILE)... (Nível de Log: $(LOG_LEVEL))"
	# Exporta a variável LOG_LEVEL para que o docker compose a possa usar.
	export LOG_LEVEL=$(LOG_LEVEL); \
	docker compose -f $(COMPOSE_FILE) up -d

# Alvo para parar e remover os contentores.
.PHONY: down
down:
	@echo ">>> Parando e removendo os serviços definidos em $(COMPOSE_FILE)..."
	docker compose -f $(COMPOSE_FILE) down

# Alvo combinado: Para, reconstrói e inicia.
.PHONY: prod-up-build
prod-up-build: down build up
	@echo ">>> Imagem reconstruída e serviços reiniciados!"

# --- Alvos Auxiliares ---

# Alvo para ver os logs (make logs) ou (make logs servico=...)
.PHONY: logs
logs:
	@echo ">>> Mostrando logs em tempo real (Pressione Ctrl+C para sair)..."
	docker compose -f $(COMPOSE_FILE) logs -f $(servico)

# --- Ajuda ---
.PHONY: help
help:
	@echo "Comandos disponíveis:"
	@echo "  make build           - Constrói a imagem Docker do serviço."
	@echo "  make up              - Inicia os serviços em background (Log: INFO)."
	@echo "  make up LOG_LEVEL=<NIVEL> - Inicia os serviços com um nível de log específico."
	@echo "  make down            - Para e remove os serviços."
	@echo "  make prod-up-build   - Para, reconstrói a imagem e reinicia os serviços."
	@echo "  make logs            - Mostra os logs de todos os serviços."
	@echo "  make logs servico=<nome> - Mostra os logs de um serviço específico."
	@echo "  make help            - Mostra esta ajuda."

.DEFAULT_GOAL := help
