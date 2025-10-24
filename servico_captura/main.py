# Documentação: Serviço de Captura de Vídeo para o FluxoAI (Microsserviços)
# Responsabilidade: Capturar frames da fonte de vídeo e publicá-los no MQTT.

import cv2
import paho.mqtt.client as mqtt
import time
import os
import logging
import sys
import numpy as np # Necessário para codificar/decodificar frame

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
FPS_LIMIT = 10 # Limita o número de frames publicados por segundo para não sobrecarregar

# --- Variáveis Globais ---
mqtt_client = None
connected_to_mqtt = False

# --- Funções MQTT ---

def on_connect(client, userdata, flags, rc):
    """Callback chamado quando a ligação ao broker MQTT é estabelecida."""
    global connected_to_mqtt
    if rc == 0:
        logging.info(f"Conectado ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        connected_to_mqtt = True
    else:
        logging.error(f"Falha ao conectar ao Broker MQTT, código de retorno: {rc}")
        connected_to_mqtt = False

def on_disconnect(client, userdata, rc):
    """Callback chamado quando a ligação ao broker MQTT é perdida."""
    global connected_to_mqtt
    logging.warning(f"Desconectado do Broker MQTT (código: {rc}). A tentar reconectar...")
    connected_to_mqtt = False
    # A biblioteca Paho trata da reconexão automaticamente se loop_start() for usado

def setup_mqtt():
    """Configura e inicia o cliente MQTT."""
    global mqtt_client
    client_id = f"fluxoai-captura-{os.getpid()}" # ID único para o cliente
    mqtt_client = mqtt.Client(client_id=client_id)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect

    logging.info(f"A tentar conectar ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    try:
        mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        mqtt_client.loop_start() # Inicia a thread de rede do MQTT em background
    except Exception as e:
        logging.error(f"Erro ao conectar ao MQTT: {e}", exc_info=True)
        sys.exit(1) # Termina se não conseguir conectar inicialmente

# --- Função Principal de Captura ---

def start_capture():
    """Inicia a captura de vídeo e publica frames no MQTT."""
    global mqtt_client, connected_to_mqtt

    logging.info(">>> Serviço de Captura a iniciar...")
    logging.info(f">>> Versão do OpenCV: {cv2.__version__}")

    is_rtsp = VIDEO_SOURCE.startswith("rtsp://")
    source_description = f"stream de rede: {VIDEO_SOURCE}" if is_rtsp else f"câmara local no índice: {VIDEO_SOURCE}"
    video_source_arg = VIDEO_SOURCE if is_rtsp else int(VIDEO_SOURCE)

    logging.info(f">>> A tentar conectar a: {source_description}...")
    cap = cv2.VideoCapture(video_source_arg)
    time.sleep(2)

    if not cap.isOpened():
        logging.error(f"!!! ERRO FATAL: Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")
        sys.exit(1)

    logging.info(">>> Fonte de vídeo conectada com sucesso!")
    logging.info(f">>> A publicar frames no tópico MQTT: {PUBLISH_TOPIC}")

    frame_count = 0
    last_publish_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("!!! Frame não recebido. A verificar ligação...")
            # Lógica de reconexão (igual à anterior)
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
                break

        current_time = time.time()
        # Limita o FPS de publicação
        if current_time - last_publish_time < (1.0 / FPS_LIMIT):
            continue # Salta este frame

        frame_count += 1
        last_publish_time = current_time

        try:
            # Redimensiona o frame para o tamanho desejado
            frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Codifica o frame como JPEG para enviar via MQTT
            ret, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80]) # Qualidade 80%
            if not ret:
                logging.warning("Falha ao codificar frame como JPEG.")
                continue

            frame_bytes = buffer.tobytes()

            # Publica no MQTT apenas se estiver conectado
            if connected_to_mqtt and mqtt_client:
                result = mqtt_client.publish(PUBLISH_TOPIC, payload=frame_bytes, qos=0) # QoS 0: Envia no máximo uma vez
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                     logging.debug(f"Frame {frame_count} publicado ({len(frame_bytes)} bytes).")
                else:
                    logging.warning(f"Falha ao publicar frame {frame_count}. Código: {result.rc}")
            elif not connected_to_mqtt:
                 logging.debug(f"Frame {frame_count} descartado (MQTT desconectado).")


        except Exception as e:
            logging.error(f"Erro no loop principal de captura: {e}", exc_info=True)
            time.sleep(1) # Evita spam de logs em caso de erro contínuo

    cap.release()
    logging.info(">>> Loop de captura terminado.")
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    logging.info(">>> Cliente MQTT desconectado.")


# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    setup_mqtt()
    start_capture()
