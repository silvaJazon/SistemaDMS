# Documentação: Serviço de Análise de Comportamento para o FluxoAI (Microsserviços)
# Responsabilidade: Receber dados de deteção e frames crus via MQTT,
#                   aplicar lógica de tracking e vadiagem, desenhar caixas
#                   e publicar o frame analisado.

import paho.mqtt.client as mqtt
import os
import time
import logging
import sys
import numpy as np
import cv2 # Para descodificar e desenhar
import json # Para processar dados de deteção
import threading

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - Análise - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Configurações da Aplicação ---
MQTT_BROKER_HOST = os.environ.get('MQTT_BROKER_HOST', 'localhost')
MQTT_BROKER_PORT = int(os.environ.get('MQTT_BROKER_PORT', 1883))
SUBSCRIBE_TOPIC_DATA = os.environ.get('SUBSCRIBE_TOPIC_DATA', 'fluxoai/detections')
SUBSCRIBE_TOPIC_RAW_IMG = os.environ.get('SUBSCRIBE_TOPIC_RAW_IMG', 'fluxoai/frames/raw')
PUBLISH_TOPIC_ANALYZED_IMG = os.environ.get('PUBLISH_TOPIC_ANALYZED_IMG', 'fluxoai/frames/analyzed')
# PUBLISH_TOPIC_STATUS = os.environ.get('PUBLISH_TOPIC_STATUS', 'fluxoai/status/loitering') # Para o futuro

LOITERING_THRESHOLD_SECONDS = int(os.environ.get('LOITERING_THRESHOLD_SECONDS', 10))
LOITERING_MAX_DISTANCE = int(os.environ.get('LOITERING_MAX_DISTANCE', 30))
JPEG_QUALITY = int(os.environ.get('JPEG_QUALITY', 50))
TARGET_LABEL = 'person' # Assumimos que o servico_ia já filtra por 'person' nos dados

# --- Variáveis Globais ---
mqtt_client = None
connected_to_mqtt = False
last_raw_frame_lock = threading.Lock()
last_raw_frame = None # Guarda o último frame cru recebido
tracked_persons = {} # Dicionário para guardar pessoas seguidas {track_id: {'box': [], 'center': (x,y), 'start_time': t, 'is_loitering': False, 'last_seen': t}}
next_track_id = 0

# --- Funções Auxiliares ---

def calculate_center(xmin, ymin, xmax, ymax):
    """Calcula o ponto central de uma caixa."""
    return int((xmin + xmax) / 2), int((ymin + ymax) / 2)

def update_tracking_and_loitering(detections_data):
    """Atualiza o tracking e o estado de vadiagem com base nos dados de deteção recebidos."""
    global tracked_persons, next_track_id

    current_time = time.time()
    current_detections_centers = [] # Lista de centros das deteções atuais
    current_detections_map = {} # Mapeia centro para dados completos da deteção

    # Processa as deteções recebidas
    for det in detections_data.get('detections', []):
        if det['label'] == TARGET_LABEL: # Foca apenas em pessoas
            ymin, xmin, ymax, xmax = det['box'] # Coordenadas já estão em pixels
            center_x, center_y = calculate_center(xmin, ymin, xmax, ymax)
            center = (center_x, center_y)
            current_detections_centers.append(center)
            current_detections_map[center] = det # Guarda dados completos associados ao centro

    matched_track_ids = set()
    newly_created_tracks = {}

    # 1. Tenta associar centros atuais a tracks existentes
    if current_detections_centers: # Apenas se houver deteções
        centers_array = np.array(current_detections_centers)
        for track_id, data in tracked_persons.items():
            if not current_detections_centers: # Otimização: Se já associámos todos, sair
                break
            
            last_center = np.array(data['center'])
            distances = np.linalg.norm(centers_array - last_center, axis=1)
            min_dist_idx = np.argmin(distances)
            min_distance = distances[min_dist_idx]

            if min_distance < LOITERING_MAX_DISTANCE * 2: # Se encontrou um match próximo
                matched_center = tuple(centers_array[min_dist_idx])
                matched_data = current_detections_map[matched_center]

                # Atualiza track
                tracked_persons[track_id]['box'] = matched_data['box']
                tracked_persons[track_id]['last_seen'] = current_time
                matched_track_ids.add(track_id)

                # Verifica vadiagem
                distance_moved = np.linalg.norm(np.array(tracked_persons[track_id]['center']) - np.array(matched_center))
                if distance_moved > LOITERING_MAX_DISTANCE:
                    # Pessoa moveu-se, reinicia contador
                    if tracked_persons[track_id]['is_loitering']:
                         logging.info(f"Pessoa ID {track_id} deixou de vadiar.")
                    tracked_persons[track_id]['center'] = matched_center
                    tracked_persons[track_id]['start_time'] = current_time
                    tracked_persons[track_id]['is_loitering'] = False
                else:
                    # Pessoa parada, verifica tempo
                    time_stopped = current_time - tracked_persons[track_id]['start_time']
                    if time_stopped > LOITERING_THRESHOLD_SECONDS and not tracked_persons[track_id]['is_loitering']:
                        tracked_persons[track_id]['is_loitering'] = True
                        logging.info(f"Pessoa ID {track_id} DETETADA vadiando (tempo: {time_stopped:.1f}s)")
                        # Futuro: Publicar alerta no tópico PUBLISH_TOPIC_STATUS

                # Remove centro associado da lista para não associar de novo
                current_detections_centers.pop(min_dist_idx)
                centers_array = np.delete(centers_array, min_dist_idx, axis=0)
                del current_detections_map[matched_center]
                
                # Se centers_array ficar vazio, sair do loop de tracks
                if centers_array.size == 0:
                    current_detections_centers = [] # Garante que a lista também fica vazia
                    break


    # 2. Cria novos tracks para deteções não associadas
    for center, det_data in current_detections_map.items():
        tracked_persons[next_track_id] = {
            'box': det_data['box'],
            'center': center,
            'start_time': current_time,
            'is_loitering': False,
            'last_seen': current_time
        }
        logging.info(f"Novo Track ID {next_track_id} criado.")
        newly_created_tracks[next_track_id] = tracked_persons[next_track_id] # Guarda para desenho inicial
        next_track_id += 1

    # 3. Remove tracks antigos
    timeout_threshold = 5 # Segundos sem ver para remover
    ids_to_remove = []
    for track_id, data in tracked_persons.items():
        if current_time - data['last_seen'] > timeout_threshold:
             ids_to_remove.append(track_id)
             if data['is_loitering']:
                  logging.info(f"Pessoa ID {track_id} (que estava vadiando) desapareceu.")
             else:
                  logging.debug(f"Track ID {track_id} removido por inatividade.")

    for track_id in ids_to_remove:
        del tracked_persons[track_id]

    return newly_created_tracks # Retorna os tracks recém-criados


def draw_tracked_persons(frame):
    """Desenha as caixas das pessoas seguidas no frame."""
    frame_copy = frame.copy() # Trabalha numa cópia para não alterar o original
    
    for track_id, data in tracked_persons.items():
        ymin, xmin, ymax, xmax = data['box']
        is_loitering = data['is_loitering']
        score = data.get('score', 1.0) # Usa score se disponível, senão assume 1.0

        color = (0, 0, 255) if is_loitering else (0, 255, 0) # Vermelho para vadiagem, Verde normal
        label_text = f'P:{track_id}' # Mostra ID do track
        # label_text = f'P:{track_id} {int(score*100)}%' # Alternativa com score
        if is_loitering:
             label_text += " (Vadiando)"

        # Desenha o retângulo
        cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), color, 2)

        # Prepara o texto da etiqueta
        label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_ymin = max(ymin, label_size[1] + 10)

        # Desenha fundo e texto da etiqueta
        cv2.rectangle(frame_copy, (xmin, label_ymin - label_size[1] - 10),
                      (xmin + label_size[0], label_ymin - base_line - 10), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame_copy, label_text, (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # Preto

    return frame_copy

# --- Funções MQTT Callbacks ---

def on_connect_analise(client, userdata, flags, rc, properties=None):
    """Callback de conexão."""
    global connected_to_mqtt
    if rc == 0:
        logging.info(f"Conectado ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        logging.info(f"A subscrever ao tópico de deteções: {SUBSCRIBE_TOPIC_DATA}")
        client.subscribe(SUBSCRIBE_TOPIC_DATA, qos=0)
        logging.info(f"A subscrever ao tópico de frames crus: {SUBSCRIBE_TOPIC_RAW_IMG}")
        client.subscribe(SUBSCRIBE_TOPIC_RAW_IMG, qos=0)
        connected_to_mqtt = True
    else:
        logging.error(f"Falha ao conectar ao Broker MQTT, código: {rc}, erro: {mqtt.connack_string(rc)}")
        connected_to_mqtt = False

def on_disconnect_analise(client, userdata, rc, properties=None):
    """Callback de desconexão."""
    global connected_to_mqtt
    logging.warning(f"Desconectado do Broker MQTT (código: {rc}).")
    connected_to_mqtt = False

def on_message_analise(client, userdata, msg):
    """Callback para mensagens recebidas (frames crus e dados de deteção)."""
    global last_raw_frame

    # Guarda o último frame cru recebido
    if msg.topic == SUBSCRIBE_TOPIC_RAW_IMG:
        try:
            logging.debug(f"Frame cru recebido ({len(msg.payload)} bytes)")
            frame_bytes = np.frombuffer(msg.payload, dtype=np.uint8)
            frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
            if frame is not None:
                with last_raw_frame_lock:
                    last_raw_frame = frame
            else:
                 logging.warning("Falha ao descodificar frame cru.")
        except Exception as e:
            logging.warning(f"Erro ao processar frame cru: {e}", exc_info=False) # Diminui verbosidade

    # Processa os dados de deteção quando chegam
    elif msg.topic == SUBSCRIBE_TOPIC_DATA:
        try:
            logging.debug(f"Dados de deteção recebidos ({len(msg.payload)} bytes)")
            detections_data = json.loads(msg.payload.decode('utf-8'))

            # Atualiza o tracking e o estado de vadiagem
            update_tracking_and_loitering(detections_data)

            # Pega no último frame cru correspondente (se disponível)
            current_frame = None
            with last_raw_frame_lock:
                if last_raw_frame is not None:
                    current_frame = last_raw_frame.copy()

            if current_frame is not None:
                # Desenha as caixas das pessoas seguidas (com estado de vadiagem)
                frame_with_boxes = draw_tracked_persons(current_frame)

                # Codifica o frame final como JPEG
                (flag, encodedImage) = cv2.imencode(".jpg", frame_with_boxes, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

                if flag:
                    # Publica o frame analisado
                    frame_final_bytes = bytearray(encodedImage)
                    publish_result = client.publish(PUBLISH_TOPIC_ANALYZED_IMG, payload=frame_final_bytes, qos=0)
                    publish_result.wait_for_publish(timeout=1.0) # Espera confirmação (ou timeout)
                    if publish_result.rc == mqtt.MQTT_ERR_SUCCESS:
                        logging.info(f"Frame analisado publicado ({len(frame_final_bytes)} bytes)")
                    else:
                        logging.warning(f"Falha ao publicar frame analisado (rc: {publish_result.rc})")
                else:
                    logging.warning("Falha ao codificar frame final para publicação.")
            else:
                logging.debug("Ainda não há frame cru para desenhar as deteções.")

        except json.JSONDecodeError:
            logging.warning("Erro ao descodificar JSON dos dados de deteção.")
        except Exception as e:
            logging.warning(f"Erro ao processar dados de deteção: {e}", exc_info=True)


def setup_mqtt_analise():
    """Configura e inicia o cliente MQTT."""
    global mqtt_client
    client_id = f"fluxoai-analise-{os.getpid()}"
    # Usa a API de Callback v2 para compatibilidade futura e clareza
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    mqtt_client.on_connect = on_connect_analise
    mqtt_client.on_disconnect = on_disconnect_analise
    mqtt_client.on_message = on_message_analise

    logging.info(f"A tentar conectar ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    try:
        mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        mqtt_client.loop_forever() # Mantém a conexão ativa e processa callbacks
    except ConnectionRefusedError:
         logging.error(f"!!! ERRO FATAL: Conexão ao Broker MQTT recusada. Verifique se o Mosquitto está a correr e acessível.")
         sys.exit(1)
    except Exception as e:
        logging.error(f"Erro inesperado no loop MQTT: {e}", exc_info=True)
        # Tentar reconectar ou sair? Por agora, apenas loga.
    finally:
         logging.info("Loop MQTT terminado.")
         if mqtt_client and mqtt_client.is_connected():
              mqtt_client.disconnect()


# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    logging.info("--- Iniciando Serviço de Análise FluxoAI ---")

    # A função setup_mqtt_analise agora contém o loop principal
    setup_mqtt_analise()

    logging.info("--- Serviço de Análise Terminado ---")

