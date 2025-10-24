# Documentação: Serviço de Análise para o FluxoAI (Microsserviços)
# Responsabilidade: Receber dados de deteção e frames, aplicar tracking/vadiagem, publicar frame analisado.

import paho.mqtt.client as mqtt
import os
import time
import logging
import sys
import json
import numpy as np
import cv2 # Importa OpenCV
import threading # Para gerir o último frame de forma segura

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - Análise - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Configurações da Aplicação ---
MQTT_BROKER_HOST = os.environ.get('MQTT_BROKER_HOST', 'localhost')
MQTT_BROKER_PORT = int(os.environ.get('MQTT_BROKER_PORT', 1883))
SUBSCRIBE_TOPIC_DATA = os.environ.get('SUBSCRIBE_TOPIC_DATA', 'fluxoai/detections')
SUBSCRIBE_TOPIC_RAW_IMG = os.environ.get('SUBSCRIBE_TOPIC_RAW_IMG', 'fluxoai/frames/raw') # Novo tópico de subscrição
PUBLISH_TOPIC_ANALYZED_IMG = os.environ.get('PUBLISH_TOPIC_ANALYZED_IMG', 'fluxoai/frames/analyzed') # Novo tópico de publicação
# PUBLISH_TOPIC_STATUS = os.environ.get('PUBLISH_TOPIC_STATUS', 'fluxoai/status/loitering') # (Opcional, para depois)

LOITERING_THRESHOLD_SECONDS = int(os.environ.get('LOITERING_THRESHOLD_SECONDS', 10))
LOITERING_MAX_DISTANCE = int(os.environ.get('LOITERING_MAX_DISTANCE', 30))
TARGET_LABEL = 'person'
JPEG_QUALITY = int(os.environ.get('JPEG_QUALITY', 50)) # Qualidade para publicar imagem

# --- Variáveis Globais ---
mqtt_client = None
connected_to_mqtt = False
tracked_persons = {} # {track_id: {'box': [], 'center': (x,y), 'start_time': t, 'is_loitering': False, 'last_seen': t}}
next_track_id = 0
last_frame_lock = threading.Lock() # Lock para acesso seguro ao último frame
last_frame_data = {'frame': None, 'timestamp': 0} # Guarda o último frame raw recebido

# --- Funções MQTT ---

def on_connect_analysis(client, userdata, flags, rc, properties=None):
    """Callback de conexão."""
    global connected_to_mqtt
    if rc == 0:
        logging.info(f"Conectado ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        logging.info(f"A subscrever ao tópico de deteções: {SUBSCRIBE_TOPIC_DATA}")
        client.subscribe(SUBSCRIBE_TOPIC_DATA, qos=0)
        logging.info(f"A subscrever ao tópico de frames crus: {SUBSCRIBE_TOPIC_RAW_IMG}")
        client.subscribe(SUBSCRIBE_TOPIC_RAW_IMG, qos=0) # Subscreve aos frames crus
        connected_to_mqtt = True
    else:
        logging.error(f"Falha ao conectar ao Broker MQTT, código: {rc}, erro: {mqtt.connack_string(rc)}")
        connected_to_mqtt = False

def on_disconnect_analysis(client, userdata, rc, properties=None):
    """Callback de desconexão."""
    global connected_to_mqtt
    logging.warning(f"Desconectado do Broker MQTT (código: {rc}).")
    connected_to_mqtt = False

def calculate_center(xmin, ymin, xmax, ymax):
    """Calcula o ponto central de uma caixa (em pixels)."""
    return int((xmin + xmax) / 2), int((ymin + ymax) / 2)

def update_tracking_and_loitering(detections_data):
    """Processa os dados de deteção recebidos para atualizar o tracking e a vadiagem."""
    global tracked_persons, next_track_id

    current_time = detections_data.get('timestamp', time.time())
    current_person_detections = [det for det in detections_data.get('detections', []) if det.get('label') == TARGET_LABEL]
    matched_track_ids = set()

    logging.debug(f"Recebidas {len(current_person_detections)} deteções de '{TARGET_LABEL}'.")

    # 1. Associa deteções atuais a tracks existentes
    for det in current_person_detections:
        # A caixa já vem em pixels do frame original (esperamos)
        xmin, ymin, xmax, ymax = det['box_pixels']
        center_x, center_y = calculate_center(xmin, ymin, xmax, ymax)
        current_center = (center_x, center_y)

        best_match_id = -1
        min_distance = float('inf')

        # Encontra o track mais próximo
        for track_id, data in tracked_persons.items():
            if track_id not in matched_track_ids:
                distance = np.linalg.norm(np.array(data['center']) - np.array(current_center))
                if distance < LOITERING_MAX_DISTANCE * 2: # Limiar de associação
                    if distance < min_distance:
                        min_distance = distance
                        best_match_id = track_id

        if best_match_id != -1:
            # Atualiza track existente
            logging.debug(f"Associando deteção atual ao Track ID {best_match_id} (Dist: {min_distance:.1f})")
            tracked_persons[best_match_id]['box'] = det['box_pixels']
            tracked_persons[best_match_id]['last_seen'] = current_time
            matched_track_ids.add(best_match_id)

            # Verifica vadiagem
            distance_moved = np.linalg.norm(np.array(tracked_persons[best_match_id]['center']) - np.array(current_center))
            if distance_moved > LOITERING_MAX_DISTANCE:
                # Moveu-se, reinicia
                tracked_persons[best_match_id]['center'] = current_center
                tracked_persons[best_match_id]['start_time'] = current_time
                if tracked_persons[best_match_id]['is_loitering']:
                     logging.info(f"Pessoa ID {best_match_id} deixou de vadiar.")
                     # Publicar evento "fim de vadiagem"?
                tracked_persons[best_match_id]['is_loitering'] = False
            else:
                # Parado, verifica tempo
                time_stopped = current_time - tracked_persons[best_match_id]['start_time']
                if not tracked_persons[best_match_id]['is_loitering'] and time_stopped > LOITERING_THRESHOLD_SECONDS:
                    tracked_persons[best_match_id]['is_loitering'] = True
                    logging.info(f"Pessoa ID {best_match_id} DETETADA vadiando (tempo: {time_stopped:.1f}s)")
                    # Publicar evento "início de vadiagem"?
        else:
            # Cria novo track
            tracked_persons[next_track_id] = {
                'box': det['box_pixels'],
                'center': current_center,
                'start_time': current_time,
                'is_loitering': False,
                'last_seen': current_time
            }
            logging.info(f"Novo Track ID {next_track_id} criado.")
            next_track_id += 1

    # 2. Remove tracks antigos
    ids_to_remove = []
    for track_id, data in tracked_persons.items():
        if current_time - data['last_seen'] > 5: # Remove se não visto por 5 segundos
             ids_to_remove.append(track_id)
             if data['is_loitering']:
                 logging.info(f"Pessoa ID {track_id} (que estava vadiando) desapareceu.")
                 # Publicar evento "fim de vadiagem"?
             else:
                 logging.debug(f"Track ID {track_id} removido por inatividade.")

    for track_id in ids_to_remove:
        del tracked_persons[track_id]

    # 3. Desenha e publica o frame analisado
    with last_frame_lock:
        if last_frame_data['frame'] is not None:
             # Garante que estamos a usar o frame correspondente ao timestamp (aproximado)
             # Idealmente, a mensagem de deteção incluiria um ID de frame ou timestamp mais preciso
             # Para simplificar, usamos o último frame recebido
            frame_to_draw = last_frame_data['frame'].copy()
            draw_analyzed_detections(frame_to_draw, tracked_persons)
            publish_analyzed_frame(frame_to_draw)
        else:
             logging.warning("Recebidos dados de deteção, mas não há frame raw recente para desenhar.")

def draw_analyzed_detections(frame, current_tracks):
    """Desenha as caixas de deteção no frame com cores baseadas no estado de vadiagem."""
    for track_id, data in current_tracks.items():
        xmin, ymin, xmax, ymax = data['box'] # A caixa já está em pixels
        is_loitering = data['is_loitering']

        color = (0, 0, 255) if is_loitering else (0, 255, 0) # Vermelho para vadiagem, Verde normal
        label_text = f'ID:{track_id}'
        if is_loitering:
             label_text += " (Vadiando)"

        # Garante coordenadas válidas (embora já devam estar)
        frame_height, frame_width, _ = frame.shape
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(frame_width - 1, xmax)
        ymax = min(frame_height - 1, ymax)

        if xmax > xmin and ymax > ymin:
            # Desenha o retângulo
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Prepara o texto da etiqueta
            label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_ymin = max(ymin, label_size[1] + 7)

            # Desenha fundo e texto da etiqueta
            cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 7),
                          (xmin + label_size[0], label_ymin - base_line - 7), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label_text, (xmin, label_ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame

def publish_analyzed_frame(frame):
    """Codifica o frame como JPEG e publica no MQTT."""
    try:
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ret:
            mqtt_client.publish(PUBLISH_TOPIC_ANALYZED_IMG, buffer.tobytes(), qos=0)
            logging.debug(f"Frame analisado publicado ({len(buffer)} bytes)")
        else:
            logging.warning("Falha ao codificar frame analisado como JPEG.")
    except Exception as e:
        logging.error(f"Erro ao publicar frame analisado: {e}", exc_info=True)


def on_message_analysis(client, userdata, msg):
    """Callback para mensagens recebidas (dados de deteção E frames crus)."""
    global last_frame_data
    current_time = time.time()

    if msg.topic == SUBSCRIBE_TOPIC_DATA:
        try:
            payload_str = msg.payload.decode('utf-8')
            detections_data = json.loads(payload_str)
            logging.debug(f"Dados de deteção recebidos.")
            update_tracking_and_loitering(detections_data)
        except json.JSONDecodeError:
            logging.warning("Erro ao descodificar JSON da mensagem de deteção.")
        except Exception as e:
            logging.error(f"Erro ao processar mensagem de deteção: {e}", exc_info=True)

    elif msg.topic == SUBSCRIBE_TOPIC_RAW_IMG:
        try:
            logging.debug(f"Frame raw recebido ({len(msg.payload)} bytes)")
            # Descodifica o JPEG recebido
            frame_bytes = np.frombuffer(msg.payload, dtype=np.uint8)
            frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
            if frame is not None:
                with last_frame_lock:
                    last_frame_data['frame'] = frame
                    last_frame_data['timestamp'] = current_time
            else:
                logging.warning("Falha ao descodificar frame raw recebido.")
        except Exception as e:
            logging.warning(f"Erro ao processar frame raw: {e}", exc_info=True)


def setup_mqtt_analysis():
    """Configura e inicia o cliente MQTT."""
    global mqtt_client
    client_id = f"fluxoai-analise-{os.getpid()}"
    # Usa a API v2 para compatibilidade com a nova assinatura dos callbacks
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    mqtt_client.on_connect = on_connect_analysis
    mqtt_client.on_disconnect = on_disconnect_analysis
    mqtt_client.on_message = on_message_analysis

    logging.info(f"A tentar conectar ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    try:
        retry_count = 0
        max_retries = 5
        while not connected_to_mqtt and retry_count < max_retries:
             try:
                 mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
                 mqtt_client.loop_start() # Inicia loop em background
                 time.sleep(2)
                 if connected_to_mqtt:
                     break
             except Exception as conn_e:
                 logging.warning(f"Tentativa {retry_count+1}/{max_retries} falhou ao conectar ao MQTT: {conn_e}")
                 retry_count += 1
                 time.sleep(5)

        if not connected_to_mqtt:
             logging.error(f"Não foi possível conectar ao Broker MQTT após {max_retries} tentativas. A sair.")
             sys.exit(1)

        # Mantém script vivo
        while True:
             time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Interrupção recebida, a desligar...")
    except Exception as e:
        logging.error(f"Erro CRÍTICO no cliente MQTT: {e}", exc_info=True)
    finally:
        if mqtt_client and mqtt_client.is_connected():
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            logging.info("Cliente MQTT desconectado.")
        sys.exit(0)

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    logging.info("--- Iniciando Serviço de Análise FluxoAI ---")
    setup_mqtt_analysis()
    logging.info(">>> Serviço de Análise terminado.")

