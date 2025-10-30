# ==========================================
# Makefile — SistemaDMS
# Simplifica comandos Docker e Docker Compose
# ==========================================

# --- Variáveis ---

COMPOSE_FILE := docker-compose.yml
SERVICO := servico_dms   # Nome do serviço principal

# Valores padrão (podem ser sobrescritos na linha de comando)
LOG_LEVEL ?= INFO
ROTATE_FRAME ?= 0

# --- Variáveis Internas ---
DOCKER_COMPOSE := docker compose -f $(COMPOSE_FILE)

# --- Alvos Principais ---

.PHONY: build
build:
	@echo ">>> Construindo a imagem de $(SERVICO) (sem cache)..."
	$(DOCKER_COMPOSE) build --no-cache $(SERVICO)

.PHONY: up
up:
	@echo ">>> Iniciando os serviços... (Log: $(LOG_LEVEL), Rotação: $(ROTATE_FRAME))"
	LOG_LEVEL=$(LOG_LEVEL) ROTATE_FRAME=$(ROTATE_FRAME) $(DOCKER_COMPOSE) up -d

.PHONY: down
down:
	@echo ">>> Parando e removendo os serviços..."
	$(DOCKER_COMPOSE) down

.PHONY: prod-up-build
prod-up-build: down
	@echo ">>> Forçando a reconstrução e reiniciando os serviços..."
	LOG_LEVEL=$(LOG_LEVEL) ROTATE_FRAME=$(ROTATE_FRAME) $(DOCKER_COMPOSE) up -d --build
	@echo ">>> Imagem reconstruída e serviços reiniciados!"

# --- Alvos Auxiliares ---

.PHONY: logs
logs:
	@echo ">>> Mostrando logs em tempo real de $(SERVICO)... (Ctrl+C para sair)"
	$(DOCKER_COMPOSE) logs -f $(SERVICO)

.PHONY: restart
restart:
	@echo ">>> Reiniciando o serviço $(SERVICO)..."
	$(DOCKER_COMPOSE) restart $(SERVICO)
	@echo ">>> Serviço reiniciado com sucesso!"

.PHONY: status
status:
	@echo ">>> Status atual dos containers:"
	$(DOCKER_COMPOSE) ps

.PHONY: clean
clean:
	@echo ">>> Removendo containers parados, imagens órfãs e volumes não utilizados..."
	docker system prune -f
	@echo ">>> Limpeza leve concluída!"

.PHONY: prune
prune:
	@echo "⚠️  ATENÇÃO: Esta ação remove TUDO (containers, imagens, volumes e redes)."
	@read -p 'Deseja realmente continuar? (digite SIM): ' CONFIRM && [ "$$CONFIRM" = "SIM" ] && docker system prune -a --volumes -f || echo 'Operação cancelada.'
	@echo ">>> Limpeza completa concluída (ou cancelada pelo usuário)."

# --- (NOVOS) Alvos de Teste e Qualidade ---

.PHONY: test
test:
	@echo ">>> Executando testes (pytest)..."
	# Executa o pytest dentro de um novo container (run --rm)
	$(DOCKER_COMPOSE) run --rm $(SERVICO) pytest

.PHONY: lint
lint:
	@echo ">>> Verificando estilo de código (flake8)..."
	$(DOCKER_COMPOSE) run --rm $(SERVICO) flake8 .

.PHONY: format
format:
	@echo ">>> Formatando código (black)..."
	$(DOCKER_COMPOSE) run --rm $(SERVICO) black .


# --- Ajuda ---

.PHONY: help
help:
	@echo ""
	@echo "==========================================="
	@echo "        Makefile — SistemaDMS"
	@echo "==========================================="
	@echo ""
	@echo "Comandos de Serviço:"
	@echo "  make build               - Constrói a imagem (sem cache)."
	@echo "  make up                  - Inicia os serviços em background."
	@echo "  make down                - Para e remove os serviços."
	@echo "  make prod-up-build       - Para, reconstrói e reinicia tudo."
	@echo "  make restart             - Reinicia o serviço principal."
	@echo ""
	@echo "Comandos de Diagnóstico e Qualidade:"
	@echo "  make status              - Mostra o status dos containers."
	@echo "  make logs                - Mostra os logs em tempo real."
	@echo "  make test                - (NOVO) Executa os testes automatizados (pytest)."
	@echo "  make lint                - (NOVO) Verifica o estilo do código (flake8)."
	@echo "  make format              - (NOVO) Formata o código (black)."
	@echo ""
	@echo "Comandos de Limpeza:"
	@echo "  make clean               - Faz uma limpeza leve (sem apagar tudo)."
	@echo "  make prune               - ⚠️ Limpeza pesada de TUDO (confirmação manual)."
	@echo ""
	@echo "Exemplos com parâmetros:"
	@echo "  make prod-up-build ROTATE_FRAME=180"
	@echo "  make up LOG_LEVEL=DEBUG ROTATE_FRAME=90"
	@echo ""
	@echo "  make help                - Mostra esta ajuda."
	@echo ""

.DEFAULT_GOAL := help