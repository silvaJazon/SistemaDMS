# Documentação: Serviço de Análise de Comportamento para o Projeto FluxoAI
# Responsabilidade: Receber dados de deteção e frames crus, aplicar lógica de Atitude Suspeita (Vadiagem) e publicar frame final continuamente.

import cv2
import time
import os
import sys
import numpy as np
import json
import logging
import threading
from paho.mqtt import client as mqtt

# Habilita otimizações do OpenCV (SIMD/SSE, etc.)
cv2.setUseOptimized(True)

# --- Configuração do Logging ---
# O nível é lido do environment, default é INFO
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - Análise - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Configurações da Aplicação ---
MQTT_BROKER_HOST = os.environ.get('MQTT_BROKER_HOST', 'mosquitto')
MQTT_BROKER_PORT = int(os.environ.get('MQTT_BROKER_PORT', 1883))
SUBSCRIBE_TOPIC_DATA = os.environ.get('SUBSCRIBE_TOPIC_DATA', 'fluxoai/detections')
SUBSCRIBE_TOPIC_RAW_IMG = os.environ.get('SUBSCRIBE_TOPIC_RAW_IMG', 'fluxoai/frames/raw')
PUBLISH_TOPIC_ANALYZED_IMG = os.environ.get('PUBLISH_TOPIC_ANALYZED_IMG', 'fluxoai/frames/analyzed')

# Tracking e Vadiagem
LOITERING_THRESHOLD_SECONDS = int(os.environ.get('LOITERING_THRESHOLD_SECONDS', 10))
LOITERING_MAX_DISTANCE = int(os.environ.get('LOITERING_MAX_DISTANCE', 30))
JPEG_QUALITY = int(os.environ.get('JPEG_QUALITY', 35)) # Qualidade baixa (35) para reduzir o trabalho de encode no RPi
MIN_BOX_SIZE = 30 # Tamanho mínimo (em pixels) para a caixa de deteção (para filtrar falsos positivos)
DETECTION_THRESHOLD = 0.55 # Limite de confiança para considerar uma deteção válida (Adicionado aqui para o Canvas)

# --- Variáveis Globais de Microsserviço ---
mqtt_client = None
# Guarda apenas os dados de deteção para serem desenhados
latest_detections_for_draw = [] 
# {track_id: {'box': [ymin, xmin, ymax, xmax], 'center': (x,y), 'start_time': t, 'is_loitering': False, 'last_seen': t}}
tracked_persons = {} 
next_track_id = 0
latest_raw_frame = None # O último frame cru (numpy array)
lock = threading.Lock() # Para acesso seguro às variáveis globais
TARGET_LABEL = 'person' # O objeto que estamos a seguir

# --- Funções Auxiliares de Tracking e Desenho ---

def calculate_center(xmin, ymin, xmax, ymax):
    """Calcula o ponto central de uma caixa."""
    return int((xmin + xmax) / 2), int((ymin + ymax) / 2)

def update_tracking(detections):
    """Associa deteções atuais a tracks existentes e atualiza o estado de vadiagem."""
    global tracked_persons, next_track_id, DETECTION_THRESHOLD
    
    current_time = time.time()
    matched_track_ids = set()
    new_detections = [] # Deteções com status de tracking atualizado

    # 1. Tenta associar deteções atuais a tracks existentes
    for det in detections:
        if det['label'] != TARGET_LABEL or det['score'] < DETECTION_THRESHOLD:
            # FILTRO: Ignora deteções que não são 'person' ou estão abaixo do limiar
            continue
            
        # O servico_ia envia a box em pixels do seu frame de detecção (640x480)
        # Assumimos [ymin, xmin, ymax, xmax] como fornecido pelo modelo SSD
        ymin, xmin, ymax, xmax = det['box_pixels']
        
        # FILTRO: Ignora deteções minúsculas (falsos positivos)
        if (xmax - xmin) < MIN_BOX_SIZE or (ymax - ymin) < MIN_BOX_SIZE:
            logging.debug("Detecção ignorada: Tamanho muito pequeno (<30px).")
            continue
        
        # CORREÇÃO CRÍTICA: As caixas já estão em pixels (640x480), não precisa de desnormalização.
        # Apenas converte para inteiros para o tracking/desenho.
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

        center_x, center_y = calculate_center(xmin, ymin, xmax, ymax)
        
        best_match_id = -1
        min_distance = float('inf')

        # Encontra o track mais próximo
        for track_id, data in tracked_persons.items():
            distance = np.linalg.norm(np.array(data['center']) - np.array((center_x, center_y)))
            if distance < LOITERING_MAX_DISTANCE * 2 and track_id not in matched_track_ids:
                if distance < min_distance:
                    min_distance = distance
                    best_match_id = track_id

        if best_match_id != -1:
            # Atualiza track existente
            track_id = best_match_id
            tracked_persons[track_id]['last_seen'] = current_time

            # Verifica vadiagem
            distance_moved = np.linalg.norm(np.array(tracked_persons[track_id]['center']) - np.array((center_x, center_y)))
            
            if distance_moved > LOITERING_MAX_DISTANCE:
                # Pessoa moveu-se, reinicia contador de vadiagem
                tracked_persons[track_id]['center'] = (center_x, center_y)
                tracked_persons[track_id]['start_time'] = current_time
                if tracked_persons[track_id]['is_loitering']:
                    tracked_persons[track_id]['is_loitering'] = False
                    logging.info(f"Pessoa ID {track_id} deixou de ter Atitude Suspeita (Moveu).")
            else:
                # Pessoa está parada, verifica tempo
                time_stopped = current_time - tracked_persons[track_id]['start_time']
                if time_stopped > LOITERING_THRESHOLD_SECONDS and not tracked_persons[track_id]['is_loitering']:
                    tracked_persons[track_id]['is_loitering'] = True
                    logging.info(f"Pessoa ID {track_id} DETETADA com ATITUDE SUSPEITA (tempo: {time_stopped:.1f}s)")
            
            # Adiciona aos resultados atuais
            new_detections.append({'box': [ymin, xmin, ymax, xmax], 'score': det['score'], 'is_loitering': tracked_persons[track_id]['is_loitering'], 'track_id': track_id})
            matched_track_ids.add(track_id)

        else:
            # Cria novo track
            track_id = next_track_id
            tracked_persons[track_id] = {
                'box': [ymin, xmin, ymax, xmax],
                'center': (center_x, center_y),
                'start_time': current_time,
                'is_loitering': False,
                'last_seen': current_time
            }
            logging.info(f"Novo Track ID {track_id} criado.")
            next_track_id += 1
            new_detections.append({'box': [ymin, xmin, ymax, xmax], 'score': det['score'], 'is_loitering': False, 'track_id': track_id})

    # 2. Remove tracks antigos (que não foram vistos recentemente)
    ids_to_remove = [track_id for track_id, data in tracked_persons.items() if current_time - data['last_seen'] > 3] # Remove se não visto por 3 segundos
    for track_id in ids_to_remove:
        if tracked_persons[track_id]['is_loitering']:
             logging.info(f"Pessoa ID {track_id} (que tinha Atitude Suspeita) desapareceu.")
        del tracked_persons[track_id]
        logging.debug(f"Track ID {track_id} removido por inatividade.")
        
    return new_detections

def draw_detections(frame, detections_to_draw):
    """Desenha as caixas de deteção no frame, usando o estado de vadiagem."""
    
    loitering_count = 0

    for det in detections_to_draw:
        # As caixas já estão em coordenadas de pixel
        ymin, xmin, ymax, xmax = det['box']
        score = det['score']
        is_loitering = det['is_loitering']
        track_id = det['track_id']

        # CORREÇÃO: Garante que as coordenadas são inteiras antes de desenhar
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

        color = (0, 0, 255) if is_loitering else (0, 255, 0) # Vermelho para Suspeita, Verde normal
        loitering_count += 1 if is_loitering else 0

        # Desenha o retângulo
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Prepara o texto da etiqueta
        label_text = f'ID:{track_id} {int(score*100)}%'
        if is_loitering:
             label_text += " (SUSPEITA)" # CORREÇÃO: Nova Terminologia

        # Prepara as coordenadas para o texto
        label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(ymin, label_size[1] + 10)

        # Desenha fundo e texto da etiqueta
        cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                      (xmin + label_size[0], label_ymin - base_line - 10), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label_text, (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
    return frame, loitering_count

# --- Thread de Publicação Contínua (Heartbeat) ---

def publish_heartbeat():
    """Publica o frame final analisado continuamente para garantir o stream do servico_web."""
    global latest_raw_frame, latest_detections_for_draw, lock, mqtt_client
    
    # Placeholder Frame (para exibição enquanto espera)
    # Codifica o placeholder APENAS UMA VEZ
    placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8) # 640x480 padrão
    cv2.putText(placeholder_frame, "Aguardando frames e analise...", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    (flag, encodedImage) = cv2.imencode(".jpg", placeholder_frame)
    placeholder_bytes = bytearray(encodedImage) if flag else None
    
    if placeholder_bytes is None:
         logging.error("Falha ao codificar imagem de placeholder. Terminando thread de publicação.")
         return 

    while True:
        frame_to_publish = None
        bytes_to_publish = None
        current_loitering_count = 0
        
        # OTIMIZAÇÃO CRÍTICA: Não usar hash para evitar o alto consumo de CPU
        # Publicar sempre que a thread acordar (30 FPS)
        
        with lock:
            if latest_raw_frame is not None:
                # 1. Copia o frame cru mais recente
                frame_to_publish = latest_raw_frame.copy()
                # 2. Desenha as ÚLTIMAS detecções/tracks conhecidos (mesmo que tenham 1s)
                frame_to_publish, current_loitering_count = draw_detections(frame_to_publish, latest_detections_for_draw)
            else:
                 # 3. Usa o placeholder se não houver frame cru
                 bytes_to_publish = placeholder_bytes

        if frame_to_publish is not None:
            # Codifica o frame analisado para JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_publish, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            
            if flag:
                bytes_to_publish = bytearray(encodedImage)
            else:
                logging.warning("Falha ao codificar o frame analisado para JPEG no heartbeat.")
                bytes_to_publish = placeholder_bytes # Usa o placeholder em caso de falha de codificação
        
        # Publica o frame APENAS se houver bytes para publicar (o que garante o stream contínuo)
        if bytes_to_publish is not None and mqtt_client is not None and mqtt_client.is_connected():
             mqtt_client.publish(PUBLISH_TOPIC_ANALYZED_IMG, bytes_to_publish, qos=0)
             logging.debug(f"Frame (heartbeat) publicado ({len(bytes_to_publish)} bytes). Suspeita: {current_loitering_count}")
        
        # CRITICAL: Controla o FPS do stream para 30 FPS FIXOS
        time.sleep(1/30) 

# --- Funções de Conexão MQTT e Callbacks ---

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        logging.info(f"Conectado ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        # Subscreve aos tópicos de dados e de frames crus
        client.subscribe(SUBSCRIBE_TOPIC_DATA, qos=0)
        client.subscribe(SUBSCRIBE_TOPIC_RAW_IMG, qos=0)
        logging.info(f"A subscrever aos tópicos de dados: {SUBSCRIBE_TOPIC_DATA} e frames crus: {SUBSCRIBE_TOPIC_RAW_IMG}")
    else:
        logging.error(f"Falha na conexão ao MQTT, código: {reason_code}")

def on_message_raw_img(client, userdata, msg):
    """Callback ao receber um frame cru do servico_captura."""
    global latest_raw_frame, lock
    
    # Decodifica o payload (JPEG bytes)
    np_arr = np.frombuffer(msg.payload, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if frame is not None:
        with lock:
            latest_raw_frame = frame
        logging.debug("Frame cru recebido para análise.")
    else:
        logging.warning("Frame cru recebido, mas falha ao descodificar.")

def on_message_data(client, userdata, msg):
    """Callback ao receber dados de deteção (JSON) do servico_ia)."""
    global latest_detections_for_draw, lock
    
    # 1. Deserializa os dados de deteção
    try:
        detections_data = json.loads(msg.payload.decode('utf-8'))
        detections = detections_data.get('detections', [])
        logging.debug(f"Dados de deteção recebidos. Total: {len(detections)}")
    except json.JSONDecodeError:
        logging.error("Erro ao descodificar payload JSON do servico_ia.")
        return

    # 2. Aplica a lógica de tracking e vadiagem
    detections_with_tracking = update_tracking(detections)
    
    # 3. CRITICAL: Atualiza a variável global de desenho
    with lock:
        # A thread de heartbeat irá ler esta variável e desenhar no frame cru mais recente
        latest_detections_for_draw = detections_with_tracking
        logging.info(f"Análise concluída. Pessoas: {len(latest_detections_for_draw)}. Suspeita: {len([d for d in latest_detections_for_draw if d['is_loitering']])}")


def setup_mqtt():
    """Configura e inicia o cliente MQTT."""
    global mqtt_client
    client_id = f'fluxoai-analise-1'
    # Utiliza a API v2 para resolver o DeprecationWarning
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    client.on_connect = on_connect
    
    # Atribui callbacks de mensagem (usa a mesma função para os dois tópicos)
    client.message_callback_add(SUBSCRIBE_TOPIC_DATA, on_message_data)
    client.message_callback_add(SUBSCRIBE_TOPIC_RAW_IMG, on_message_raw_img)

    logging.info(f"A tentar conectar ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    
    client.loop_start() # Inicia o loop de processamento em background
    return client

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    logging.info("--- Iniciando Serviço de Análise FluxoAI ---")
    
    # Configuração e início do MQTT
    mqtt_client = setup_mqtt()

    # Inicia a thread de publicação contínua
    heartbeat_thread = threading.Thread(target=publish_heartbeat)
    heartbeat_thread.daemon = True
    heartbeat_thread.start()

    try:
        # Mantém o programa a correr indefinidamente
        while True:
            time.sleep(1) 
    except KeyboardInterrupt:
        logging.info("Serviço de Análise terminado pelo utilizador.")
    finally:
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
