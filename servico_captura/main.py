# Documentação: Serviço de Captura de Vídeo para o FluxoAI (Microsserviços)
# Responsabilidade: Capturar frames da fonte de vídeo e publicá-los no MQTT.

import cv2
import paho.mqtt.client as mqtt
import time
import os
import logging
import sys
import numpy as np

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - Captura - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Configurações da Aplicação (via Variáveis de Ambiente) ---
MQTT_BROKER_HOST = os.environ.get('MQTT_BROKER_HOST', 'localhost')
MQTT_BROKER_PORT = int(os.environ.get('MQTT_BROKER_PORT', 1883))
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH = int(os.environ.get('FRAME_WIDTH', 640))
FRAME_HEIGHT = int(os.environ.get('FRAME_HEIGHT', 480))
PUBLISH_TOPIC = os.environ.get('PUBLISH_TOPIC', 'fluxoai/frames/raw')
FPS_LIMIT = 10 # Limita o número de frames publicados por segundo

# --- Variáveis Globais ---
mqtt_client = None
connected_to_mqtt = False

# --- Funções MQTT ---

def on_connect(client, userdata, flags, rc, properties=None): # Adicionado properties para v2
    """Callback chamado quando a ligação ao broker MQTT é estabelecida."""
    global connected_to_mqtt
    # Verifica o código de retorno (rc) para sucesso (0)
    # Veja: https://github.com/eclipse/paho.mqtt.python?tab=readme-ov-file#return-codes--reasons-codes
    if rc == 0:
        logging.info(f"Conectado ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        connected_to_mqtt = True
    else:
        # Tenta obter a mensagem de erro correspondente ao código
        error_string = mqtt.connack_string(rc)
        logging.error(f"Falha ao conectar ao Broker MQTT, código: {rc}, erro: {error_string}")
        connected_to_mqtt = False
        # Considerar sair ou tentar reconectar manualmente aqui se a reconexão automática falhar

def on_disconnect(client, userdata, rc, properties=None): # Adicionado properties para v2
    """Callback chamado quando a ligação ao broker MQTT é perdida."""
    global connected_to_mqtt
    logging.warning(f"Desconectado do Broker MQTT (código: {rc}). A biblioteca tentará reconectar...")
    connected_to_mqtt = False

def setup_mqtt():
    """Configura e inicia o cliente MQTT."""
    global mqtt_client
    # Usa a versão 2 da API de callback para compatibilidade futura
    client_id = f"fluxoai-captura-{os.getpid()}" # ID único para o cliente
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect

    logging.info(f"A tentar conectar ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    try:
        # Tenta conectar com um timeout inicial de 60 segundos
        mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        # loop_start() inicia uma thread para gerir a rede MQTT (reconexão, etc.)
        mqtt_client.loop_start()
    except Exception as e:
        logging.error(f"Erro CRÍTICO ao tentar conectar inicialmente ao MQTT: {e}", exc_info=True)
        # Se a conexão inicial falhar, sair pode ser a melhor opção
        # pois indica um problema de configuração ou de rede fundamental.
        sys.exit(1)

# --- Função Principal de Captura ---

def start_capture():
    """Inicia a captura de vídeo e publica frames no MQTT."""
    global mqtt_client, connected_to_mqtt

    logging.info(">>> Serviço de Captura a iniciar...")
    logging.info(f">>> Versão do OpenCV: {cv2.__version__}")

    is_rtsp = VIDEO_SOURCE.startswith("rtsp://")
    source_description = f"stream de rede: {VIDEO_SOURCE}" if is_rtsp else f"câmara local no índice: {VIDEO_SOURCE}"
    try:
        video_source_arg = VIDEO_SOURCE if is_rtsp else int(VIDEO_SOURCE)
    except ValueError:
        logging.error(f"!!! ERRO FATAL: VIDEO_SOURCE inválido para câmara local: {VIDEO_SOURCE}. Deve ser um número.")
        sys.exit(1)


    logging.info(f">>> A tentar conectar a: {source_description}...")
    cap = cv2.VideoCapture(video_source_arg)

    # Verifica se a fonte de vídeo abriu corretamente
    # Adiciona uma pequena espera e nova verificação, útil para algumas câmaras/streams
    if not cap.isOpened():
        logging.warning("!!! Fonte de vídeo não abriu imediatamente. A aguardar 2 segundos...")
        time.sleep(2)
        cap = cv2.VideoCapture(video_source_arg) # Tenta abrir novamente

    if not cap.isOpened():
        logging.error(f"!!! ERRO FATAL: Não foi possível abrir a fonte de vídeo após nova tentativa: {VIDEO_SOURCE}")
        sys.exit(1)

    logging.info(">>> Fonte de vídeo conectada com sucesso!")
    logging.info(f">>> A publicar frames no tópico MQTT: {PUBLISH_TOPIC} (max {FPS_LIMIT} FPS)")

    frame_count = 0
    last_publish_time = 0

    while True:
        try: # Adiciona try..except para capturar erros durante a leitura do frame
            ret, frame = cap.read()
        except Exception as e:
            logging.error(f"!!! Erro ao ler frame da fonte de vídeo: {e}", exc_info=True)
            ret = False # Força a lógica de reconexão/saída

        if not ret:
            logging.warning("!!! Frame não recebido. A verificar ligação...")
            # Lógica de reconexão ou término
            if is_rtsp:
                logging.info(">>> A tentar reconectar ao stream RTSP...")
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(video_source_arg)
                if not cap.isOpened():
                    logging.error("!!! Falha ao reconectar. A terminar.")
                    break
                else:
                    logging.info(">>> Reconectado com sucesso!")
                    continue
            else:
                logging.error("!!! Falha ao ler frame da câmara local. A terminar.")
                break # Sai do loop se for câmara local

        current_time = time.time()
        # Limita o FPS de publicação
        time_since_last_publish = current_time - last_publish_time
        if time_since_last_publish < (1.0 / FPS_LIMIT):
            # logging.debug(f"A saltar frame {frame_count+1}. Tempo desde último: {time_since_last_publish:.3f}s")
            time.sleep((1.0 / FPS_LIMIT) - time_since_last_publish) # Espera o tempo restante para atingir o FPS
            continue # Salta este frame

        frame_count += 1
        last_publish_time = time.time() # Atualiza o tempo ANTES de publicar

        try:
            # Redimensiona o frame
            frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Codifica o frame como JPEG
            ret_encode, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret_encode:
                logging.warning(f"Frame {frame_count}: Falha ao codificar como JPEG.")
                continue

            frame_bytes = buffer.tobytes()

            # Publica no MQTT apenas se estiver conectado
            if connected_to_mqtt and mqtt_client and mqtt_client.is_connected():
                # Publica a mensagem. QoS 0 é o mais rápido, mas sem garantia de entrega.
                # Para vídeo, perder um frame ocasional geralmente não é crítico.
                result, mid = mqtt_client.publish(PUBLISH_TOPIC, payload=frame_bytes, qos=0)
                if result == mqtt.MQTT_ERR_SUCCESS:
                     logging.debug(f"Frame {frame_count} publicado (MID: {mid}, {len(frame_bytes)} bytes).")
                elif result == mqtt.MQTT_ERR_NO_CONN:
                    logging.warning(f"Frame {frame_count} não publicado (sem conexão MQTT no momento).")
                    # A reconexão é gerida pela thread loop_start()
                else:
                    # Outros erros (ex: fila cheia se QoS > 0)
                    logging.warning(f"Falha ao publicar frame {frame_count}. Código Paho: {result}")
            elif not connected_to_mqtt:
                 logging.debug(f"Frame {frame_count} descartado (MQTT desconectado).")

        except cv2.error as e:
             logging.error(f"Erro OpenCV no frame {frame_count}: {e}", exc_info=False) # Não mostra traceback completo para erros comuns de CV
        except Exception as e:
            logging.error(f"Erro inesperado no loop principal (frame {frame_count}): {e}", exc_info=True)
            time.sleep(1) # Pausa para evitar spam de logs

    # --- Limpeza ao Sair ---
    logging.info(">>> A terminar o serviço de captura...")
    if cap.isOpened():
        cap.release()
        logging.info(">>> Fonte de vídeo libertada.")
    if mqtt_client:
        logging.info(">>> A parar a thread MQTT...")
        mqtt_client.loop_stop() # Para a thread MQTT
        logging.info(">>> A desconectar do broker MQTT...")
        mqtt_client.disconnect() # Desconecta explicitamente
        logging.info(">>> Cliente MQTT desconectado.")
    logging.info(">>> Serviço de captura terminado.")


# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    logging.info("--- Iniciando Serviço de Captura FluxoAI ---")
    # Adiciona uma pausa inicial para dar tempo ao broker MQTT iniciar
    STARTUP_DELAY = 3 # Segundos
    logging.info(f"A aguardar {STARTUP_DELAY} segundos para o broker MQTT iniciar...")
    time.sleep(STARTUP_DELAY)

    setup_mqtt() # Configura e conecta ao MQTT
    start_capture() # Inicia o loop principal de captura e publicação

