# SistemaDMS - Sistema de Monitoramento de Motorista

Este projeto é um **Sistema de Monitoramento de Motorista (Driver Monitoring System - DMS)** que utiliza visão computacional para detetar sinais de sonolência, bocejo e distração (uso de celular) em tempo real.

Ele é construído usando **Python**, **Flask**, **MediaPipe** e **YOLOv8**, e é totalmente containerizado com **Docker** para fácil implantação e execução. O sistema fornece uma interface web para visualização ao vivo e ajuste de parâmetros.

## Funcionalidades Principais

*   **Deteção de Sonolência**: Monitora a Abertura Ocular (**EAR - Eye Aspect Ratio**) para detetar olhos fechados por períodos prolongados.
*   **Deteção de Bocejo**: Monitora a Abertura da Boca (**MAR - Mouth Aspect Ratio**) para detetar bocejos frequentes ou prolongados.
*   **Deteção de Distração (Celular)**: Utiliza um método híbrido otimizado:
    1.  Deteta mãos em tempo real usando **MediaPipe Hands** (rápido).
    2.  Se uma mão é detetada, executa o modelo **YOLOv8s** apenas no recorte da imagem da mão para confirmar a presença de um celular ("cell phone").
    3.  Um alerta é gerado se o celular for detetado continuamente por um número configurável de segundos.
*   **Interface Web**: Fornece um stream de vídeo ao vivo (na porta 5000) com as deteções sobrepostas e uma interface para ajustar os parâmetros (como thresholds de EAR/MAR, rotação, etc.).
*   **Gestão de Alertas**: Salva todos os eventos (Sonolência, Bocejo, Distração) num banco de dados local e captura imagens de evidência, que podem ser visualizadas na página `/alerts`.
*   **Persistência de Configuração**: As configurações ajustadas pela interface web (ex: thresholds, rotação da câmara) são salvas automaticamente no volume `config_dms/settings.json` e recarregadas ao reiniciar o serviço.

## Tecnologias Utilizadas

| Categoria | Tecnologias |
| :--- | :--- |
| Backend | Python 3, Flask, Waitress |
| Visão Computacional | OpenCV, MediaPipe (FaceMesh, Hands), YOLOv8s (Ultralytics, PyTorch) |
| Containerização | Docker, Docker Compose |
| Frontend (Interface) | HTML, JavaScript (implícito pelos templates e app.py) |
| Qualidade de Código | Pytest, Flake8, Black (configurados no Makefile) |

## Como Executar

Este projeto é desenhado para ser executado com **Docker** e **Docker Compose**. O `Makefile` fornecido simplifica todos os comandos necessários.

### Pré-requisitos

1.  Docker e Docker Compose instalados.
2.  Uma câmara USB conectada (ou outra fonte de vídeo). O serviço tentará aceder a `/dev/video0` por padrão.

### Instalação e Execução

O `Makefile` é a forma recomendada de interagir com o projeto.

**Construir e Iniciar (Recomendado para a primeira vez)**: Este comando pára qualquer versão antiga, força a reconstrução da imagem Docker (sem cache) e inicia os serviços em background.

\`\`\`bash
make prod-up-build
\`\`\`

**Iniciar Serviços (Uso normal)**: Se a imagem já estiver construída, este comando apenas inicia os contentores.

\`\`\`bash
make up
\`\`\`

**Parar Serviços**: Este comando pára e remove os contentores da aplicação.

\`\`\`bash
make down
\`\`\`

## Utilização

Após iniciar os serviços (`make up`), podes aceder à aplicação:

*   **Interface Principal e Stream**: Abre o teu navegador e acede a: [http://localhost:5000](http://localhost:5000)
*   **Página de Alertas**: Para ver o histórico de eventos detetados: [http://localhost:5000/alerts](http://localhost:5000/alerts)

### Parâmetros de Execução (Opcional)

Podes passar variáveis de ambiente ao `Makefile` para alterar o comportamento do serviço:

*   `ROTATE_FRAME`: Define a rotação da câmara (em graus). Padrão: `0`.
*   `LOG_LEVEL`: Define o nível de log (ex: `DEBUG`, `INFO`, `WARNING`). Padrão: `INFO`.

**Exemplos**:

\`\`\`bash
# Iniciar o serviço rodando a câmara em 180 graus
make up ROTATE_FRAME=180

# Iniciar com logs de debug e rotação de 90 graus
make up LOG_LEVEL=DEBUG ROTATE_FRAME=90
\`\`\`

## Comandos de Diagnóstico e Desenvolvimento

O `Makefile` também inclui ferramentas úteis para desenvolvimento e diagnóstico:

*   **Ver Logs (em tempo real)**:
    \`\`\`bash
    make logs
    \`\`\`
*   **Verificar Status dos Serviços**:
    \`\`\`bash
    make status
    \`\`\`
*   **Formatar o Código (Black)**:
    \`\`\`bash
    make format
    \`\`\`
*   **Verificar Estilo do Código (Flake8)**:
    \`\`\`bash
    make lint
    \`\`\`
*   **Executar Testes (Pytest)**:
    \`\`\`bash
    make test
    \`\`\`

## Estrutura de Pastas (Volumes)

O `docker-compose.yml` mapeia duas pastas locais para dentro do contentor, permitindo a persistência de dados:

*   `./alertas_dms/`: Esta pasta é mapeada para `/app/alerts` no contentor. Ela armazena a base de dados SQLite dos alertas e as imagens capturadas.
*   `./config_dms/`: Esta pasta é mapeada para `/app/config`. Ela armazena o ficheiro `settings.json`, que guarda as tuas configurações da interface web.
