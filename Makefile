# Makefile para simplificar comandos Docker/Docker Compose do SistemaDMS

# --- Variáveis ---
COMPOSE_FILE := docker-compose.yml
LOG_LEVEL ?= INFO          # Nível de log padrão
SERVICO := servico_dms      # Nome do serviço

# --- Alvos Principais ---

.PHONY: build
build:
	@echo ">>> Construindo a imagem definida em $(COMPOSE_FILE) (sem cache)..."
	docker compose -f $(COMPOSE_FILE) build --no-cache

.PHONY: up
up:
	@echo ">>> Iniciando os serviços definidos em $(COMPOSE_FILE)... (Log: $(LOG_LEVEL))"
	LOG_LEVEL=$(LOG_LEVEL) docker compose -f $(COMPOSE_FILE) up -d

.PHONY: down
down:
	@echo ">>> Parando e removendo os serviços definidos em $(COMPOSE_FILE)..."
	docker compose -f $(COMPOSE_FILE) down

.PHONY: prod-up-build
prod-up-build: down
	@echo ">>> Forçando a reconstrução e reiniciando os serviços..."
	LOG_LEVEL=$(LOG_LEVEL) docker compose -f $(COMPOSE_FILE) up -d --build
	@echo ">>> Imagem reconstruída e serviços reiniciados!"

# --- Alvos Auxiliares ---

.PHONY: logs
logs:
	@echo ">>> Mostrando logs em tempo real (Pressione Ctrl+C para sair)..."
	docker compose -f $(COMPOSE_FILE) logs -f $(SERVICO)

# --- Ajuda ---

.PHONY: help
help:
	@echo "Comandos disponíveis:"
	@echo "  make build                   - Constrói a imagem (sem cache)."
	@echo "  make up                      - Inicia os serviços em background (Log: INFO)."
	@echo "  make up LOG_LEVEL=<NIVEL>    - Inicia os serviços com nível de log específico."
	@echo "  make down                     - Para e remove os serviços."
	@echo "  make prod-up-build           - Para, força a reconstrução e reinicia os serviços."
	@echo "  make logs                     - Mostra os logs de todos os serviços."
	@echo "  make logs SERVICO=<nome>      - Mostra os logs de um serviço específico."
	@echo "  make help                     - Mostra esta ajuda."

.DEFAULT_GOAL := help
