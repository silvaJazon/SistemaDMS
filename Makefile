#Documentação: Makefile para simplificar os comandos Docker e Docker Compose do projeto SistemaDMS.

#VERSÃO CORRIGIDA: Usa 'docker compose build' e 'up --build' para evitar problemas de cache.

#--- Variáveis ---

COMPOSE_FILE := docker-compose.yml
LOG_LEVEL ?= INFO # Nível de log padrão (INFO)

#Nome do serviço único no docker-compose

SERVICO := servico_dms

#--- Alvos Principais ---

#Alvo para construir a imagem Docker (força a não usar cache para descarregar modelos)

.PHONY: build
build:
@echo ">>> Construindo a imagem definida em $(COMPOSE_FILE) (sem cache)..."
export LOG_LEVEL=$(LOG_LEVEL);

docker compose -f $(COMPOSE_FILE) build --no-cache

#Alvo para iniciar os serviços em background.

.PHONY: up
up:
@echo ">>> Iniciando os serviços definidos em $(COMPOSE_FILE)... (Nível de Log: $(LOG_LEVEL))"

#Exporta a variável LOG_LEVEL para que o docker compose a possa usar.

export LOG_LEVEL=$(LOG_LEVEL);

docker compose -f $(COMPOSE_FILE) up -d

#Alvo para parar e remover os contentores.

.PHONY: down
down:
@echo ">>> Parando e removendo os serviços definidos em $(COMPOSE_FILE)..."
docker compose -f $(COMPOSE_FILE) down

#Alvo combinado: Para, reconstrói (forçado) e inicia.

.PHONY: prod-up-build
prod-up-build: down
@echo ">>> Forçando a reconstrução e reiniciando os serviços..."
# CORREÇÃO: Usar 'up -d --build' para forçar a reconstrução
export LOG_LEVEL=$(LOG_LEVEL);

docker compose -f $(COMPOSE_FILE) up -d --build
@echo ">>> Imagem reconstruída e serviços reiniciados!"

#--- Alvos Auxiliares ---

#Alvo para ver os logs (make logs) ou (make logs servico=...)

.PHONY: logs
logs:
@echo ">>> Mostrando logs em tempo real (Pressione Ctrl+C para sair)..."
docker compose -f $(COMPOSE_FILE) logs -f $(servico)

#--- Ajuda ---

.PHONY: help
help:
@echo "Comandos disponíveis:"
@echo "  make build          - Constrói a imagem (sem cache)."
@echo "  make up             - Inicia os serviços em background (Log: INFO)."
@echo "  make up LOG_LEVEL=<NIVEL> - Inicia os serviços com um nível de log específico."
@echo "  make down           - Para e remove os serviços."
@echo "  make prod-up-build  - Para, FORÇA a reconstrução e reinicia os serviços."
@echo "  make logs           - Mostra os logs de todos os serviços."
@echo "  make logs servico=<nome> - Mostra os logs de um serviço específico."

    @echo "  make help            - Mostra esta ajuda."



.DEFAULT_GOAL := help