# Documentação: Serviço de IA para o FluxoAI (Microsserviços)
# Responsabilidade: Receber frames via MQTT, executar deteção, publicar resultados.

import cv2
import time
import os
import numpy as np
import logging
import sys
import paho.mqtt.client as mqtt
import tflite_runtime.interpreter as tflite
import json # Para publicar os resultados da deteção

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - IA - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Configurações da Aplicação (via Variáveis de Ambiente) ---
MQTT_BROKER_HOST = os.environ.get('MQTT_BROKER_HOST', 'localhost')
MQTT_BROKER_PORT = int(os.environ.get('MQTT_BROKER_PORT', 1883))
SUBSCRIBE_TOPIC = os.environ.get('SUBSCRIBE_TOPIC', 'fluxoai/frames/raw')
PUBLISH_TOPIC_DATA = os.environ.get('PUBLISH_TOPIC_DATA', 'fluxoai/detections') # Tópico para dados JSON
PUBLISH_TOPIC_IMG = os.environ.get('PUBLISH_TOPIC_IMG', 'fluxoai/frames/detected') # Tópico para imagem com caixas
MODEL_PATH = 'model.tflite'
LABELS_PATH = 'labels.txt'
DETECTION_THRESHOLD = float(os.environ.get('DETECTION_THRESHOLD', 0.55)) # Limite de confiança
TARGET_LABEL = 'person'

# --- Variáveis Globais ---
mqtt_client = None
connected_to_mqtt = False
labels = None
person_class_id = -1
interpreter = None
input_details = None
output_details = None
model_height = 0
model_width = 0
floating_model = False

# --- Funções Auxiliares (load_labels, initialize_model - como antes) ---

def load_labels_and_find_person(path):
    """Carrega as etiquetas e encontra o ID da classe 'person'."""
    global person_class_id
    try:
        loaded_labels = {}
        with open(path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                # Ignora linhas vazias ou comentários
                clean_line = line.strip()
                if clean_line and not clean_line.startswith('#'):
                     # Assume formato "index name" ou apenas "name"
                     parts = clean_line.split(maxsplit=1)
                     label_name = parts[-1] # Pega o nome
                     # Tenta obter o índice se houver, senão usa a ordem da linha
                     try:
                        label_index = int(parts[0]) if len(parts) > 1 else i
                     except ValueError:
                         label_index = i # Usa ordem da linha se o primeiro item não for número

                     loaded_labels[label_index] = label_name
                     if label_name == TARGET_LABEL:
                         person_class_id = label_index

        logging.info(f">>> Etiquetas carregadas ({len(loaded_labels)}): {list(loaded_labels.values())[:5]}...")
        if person_class_id != -1:
            logging.info(f">>> ID da classe '{TARGET_LABEL}': {person_class_id}")
        else:
            logging.error(f"!!! ERRO FATAL: A etiqueta alvo '{TARGET_LABEL}' não foi encontrada em {path}")
            sys.exit(1) # Termina se não encontrar a etiqueta 'person'
        return loaded_labels
    except FileNotFoundError:
        logging.error(f"!!! ERRO FATAL: Ficheiro de etiquetas não encontrado em {path}")
        sys.exit(1)

def initialize_model():
    """Carrega o modelo TFLite e aloca tensores."""
    global interpreter, input_details, output_details, model_height, model_width, floating_model
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        model_height = input_details[0]['shape'][1]
        model_width = input_details[0]['shape'][2]
        floating_model = (input_details[0]['dtype'] == np.float32)

        logging.info(f">>> Modelo TFLite carregado: {MODEL_PATH}")
        logging.debug(f">>> Input Shape: {input_details[0]['shape']}, Input Type: {input_details[0]['dtype']}")
        logging.debug(f">>> Modelo espera input flutuante: {floating_model}")
        logging.debug(f">>> Detalhes dos Outputs: {output_details}")
    except Exception as e:
        logging.error(f"!!! ERRO FATAL ao carregar o modelo TFLite ({MODEL_PATH}): {e}", exc_info=True)
        sys.exit(1) # Termina se não conseguir carregar o modelo

def detect_objects_on_frame(frame):
    """Executa a deteção de objetos num frame."""
    global interpreter, input_details, output_details, model_height, model_width, floating_model

    if interpreter is None:
        logging.error("!!! Modelo TFLite não inicializado.")
        return [], [], [], 0

    image_resized = cv2.resize(frame, (model_width, model_height))
    input_data = np.expand_dims(image_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = np.uint8(input_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    # Ordem padrão esperada para SSD MobileNet TFLite (pode precisar de ajuste)
    # 0: locations (boxes), 1: classes, 2: scores, 3: num_detections
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        # num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0]) # Opcional
        logging.debug(f"Outputs obtidos na ordem padrão. Scores[0]: {scores[0]:.2f}, Classe[0]: {int(classes[0])}")
    except IndexError:
        # Tenta ordem alternativa ou menos outputs
        try:
             # Ordem comum diferente: 0:scores, 1:boxes, 2:count, 3:classes (ou similar)
             # Ajuste os índices conforme os 'output_details' impressos no log DEBUG
            scores = interpreter.get_tensor(output_details[0]['index'])[0]
            boxes = interpreter.get_tensor(output_details[1]['index'])[0]
            classes = interpreter.get_tensor(output_details[3]['index'])[0] # Exemplo, ajuste se necessário
            logging.warning("Aviso: Ordem alternativa de outputs do modelo TFLite detetada/utilizada.")
        except IndexError as e:
            logging.error(f"!!! ERRO ao obter outputs do modelo. Detalhes: {output_details}. Erro: {e}", exc_info=True)
            return [], [], [], 0 # Retorna vazio se não conseguir interpretar

    return boxes, classes, scores, inference_time

def draw_detections(frame, boxes, classes, scores):
    """Desenha as caixas das pessoas detetadas no frame."""
    frame_height, frame_width, _ = frame.shape
    detections_list = [] # Lista para guardar dados das deteções

    for i in range(len(scores)):
        if scores[i] > DETECTION_THRESHOLD and int(classes[i]) == person_class_id:
            ymin, xmin, ymax, xmax = boxes[i]
            # Converte para pixels
            xmin_pix = int(xmin * frame_width)
            xmax_pix = int(xmax * frame_width)
            ymin_pix = int(ymin * frame_height)
            ymax_pix = int(ymax * frame_height)

            # Garante limites
            xmin_pix = max(0, xmin_pix)
            ymin_pix = max(0, ymin_pix)
            xmax_pix = min(frame_width - 1, xmax_pix)
            ymax_pix = min(frame_height - 1, ymax_pix)

            if xmax_pix > xmin_pix and ymax_pix > ymin_pix:
                # Guarda dados da deteção
                detection_data = {
                    'label': TARGET_LABEL,
                    'score': float(scores[i]),
                    'box_pixels': [xmin_pix, ymin_pix, xmax_pix, ymax_pix], # [x_min, y_min, x_max, y_max]
                    'box_normalized': [float(xmin), float(ymin), float(xmax), float(ymax)] # [x_min, y_min, x_max, y_max] normalizado
                }
                detections_list.append(detection_data)

                # Desenha no frame
                cv2.rectangle(frame, (xmin_pix, ymin_pix), (xmax_pix, ymax_pix), (0, 255, 0), 2) # Verde
                label_text = f'{TARGET_LABEL}: {int(scores[i]*100)}%'
                label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin_pix, label_size[1] + 10)
                cv2.rectangle(frame, (xmin_pix, label_ymin - label_size[1] - 10),
                              (xmin_pix + label_size[0], label_ymin - base_line - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label_text, (xmin_pix, label_ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return frame, detections_list


# --- Funções MQTT ---

def on_connect_ia(client, userdata, flags, rc, properties=None):
    """Callback de conexão para o serviço de IA."""
    global connected_to_mqtt
    if rc == 0:
        logging.info(f"Conectado ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        logging.info(f"A subscrever ao tópico de frames: {SUBSCRIBE_TOPIC}")
        client.subscribe(SUBSCRIBE_TOPIC, qos=0) # QoS 0 para vídeo é geralmente suficiente
        connected_to_mqtt = True
    else:
        error_string = mqtt.connack_string(rc)
        logging.error(f"Falha ao conectar ao Broker MQTT, código: {rc}, erro: {error_string}")
        connected_to_mqtt = False
        # Considerar retry ou sys.exit aqui

def on_disconnect_ia(client, userdata, rc, properties=None):
    """Callback de desconexão para o serviço de IA."""
    global connected_to_mqtt
    logging.warning(f"Desconectado do Broker MQTT (código: {rc}). A biblioteca tentará reconectar...")
    connected_to_mqtt = False

def on_message(client, userdata, msg):
    """Callback chamado quando uma mensagem (frame) é recebida."""
    global mqtt_client # Necessário para publicar
    start_time = time.time()
    try:
        logging.debug(f"Mensagem recebida no tópico {msg.topic} ({len(msg.payload)} bytes)")

        # Descodifica a imagem JPEG recebida
        frame_bytes = np.frombuffer(msg.payload, dtype=np.uint8)
        frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            logging.warning("Falha ao descodificar frame JPEG recebido.")
            return

        # Executa a deteção de objetos
        boxes, classes, scores, inference_time = detect_objects_on_frame(frame)

        # Desenha as deteções no frame
        frame_with_detections, detections_data = draw_detections(frame, boxes, classes, scores)

        # Publica os dados da deteção (JSON)
        if detections_data and mqtt_client and mqtt_client.is_connected():
            payload_json = json.dumps({"detections": detections_data, "timestamp": time.time()})
            result_data, mid_data = mqtt_client.publish(PUBLISH_TOPIC_DATA, payload=payload_json, qos=0)
            if result_data == mqtt.MQTT_ERR_SUCCESS:
                logging.debug(f"Dados de deteção publicados (MID: {mid_data})")
            else:
                logging.warning(f"Falha ao publicar dados de deteção. Código Paho: {result_data}")

        # Publica a imagem com as deteções (JPEG)
        if mqtt_client and mqtt_client.is_connected():
            ret_encode, buffer = cv2.imencode('.jpg', frame_with_detections, [int(cv2.IMWRITE_JPEG_QUALITY), 85]) # Qualidade ligeiramente maior
            if ret_encode:
                frame_bytes_out = buffer.tobytes()
                result_img, mid_img = mqtt_client.publish(PUBLISH_TOPIC_IMG, payload=frame_bytes_out, qos=0)
                if result_img == mqtt.MQTT_ERR_SUCCESS:
                    logging.debug(f"Frame processado publicado (MID: {mid_img}, {len(frame_bytes_out)} bytes)")
                else:
                    logging.warning(f"Falha ao publicar frame processado. Código Paho: {result_img}")
            else:
                 logging.warning("Falha ao codificar frame processado para JPEG.")

        processing_time = time.time() - start_time
        logging.info(f"Frame processado. Pessoas: {len(detections_data)}. Inferência: {inference_time:.3f}s. Total: {processing_time:.3f}s")

    except cv2.error as e:
        logging.error(f"Erro OpenCV ao processar frame recebido: {e}", exc_info=False)
    except Exception as e:
        logging.error(f"Erro inesperado ao processar mensagem MQTT: {e}", exc_info=True)


def setup_mqtt_ia():
    """Configura e inicia o cliente MQTT para o serviço de IA."""
    global mqtt_client
    client_id = f"fluxoai-ia-{os.getpid()}"
    # Usa a API v2 para compatibilidade com futuras versões do Paho e melhor tratamento de erros
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    mqtt_client.on_connect = on_connect_ia
    mqtt_client.on_disconnect = on_disconnect_ia
    mqtt_client.on_message = on_message

    logging.info(f"A tentar conectar ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    try:
        # Tenta conectar repetidamente em caso de falha inicial (ex: broker ainda a iniciar)
        retry_count = 0
        max_retries = 5
        while not connected_to_mqtt and retry_count < max_retries:
             try:
                 mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
                 mqtt_client.loop_start() # Inicia o loop em background
                 # Espera um pouco para a conexão ser estabelecida
                 time.sleep(2)
                 if connected_to_mqtt:
                     break # Sai do loop se conectado
             except Exception as conn_e:
                 logging.warning(f"Tentativa {retry_count+1}/{max_retries} falhou ao conectar ao MQTT: {conn_e}")
                 retry_count += 1
                 time.sleep(5) # Espera antes de tentar novamente

        if not connected_to_mqtt:
             logging.error(f"Não foi possível conectar ao Broker MQTT após {max_retries} tentativas. A sair.")
             sys.exit(1)

        # Mantém o script principal vivo enquanto o loop MQTT corre em background
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
    logging.info("--- Iniciando Serviço de IA FluxoAI ---")
    initialize_model() # Carrega o modelo TFLite
    labels = load_labels_and_find_person(LABELS_PATH) # Carrega as etiquetas

    # Não precisa mais de pausa aqui, a lógica de retry cuida disso
    # STARTUP_DELAY = 4
    # logging.info(f"A aguardar {STARTUP_DELAY} segundos para o broker MQTT iniciar...")
    # time.sleep(STARTUP_DELAY)

    setup_mqtt_ia() # Inicia o cliente MQTT e entra no loop

    logging.info(">>> Serviço de IA terminado.")

