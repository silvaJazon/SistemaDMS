import os
import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import json
import threading
import logging

# =====================================
# CONFIGURAÇÕES GERAIS
# =====================================
cv2.setUseOptimized(True)  # ativa otimizações SIMD do OpenCV

# Variáveis globais
lock = threading.Lock()
latest_raw_frame = None
latest_detections_for_draw = []
latest_tracking_data = {}

# MQTT
MQTT_BROKER = "servico_broker"
MQTT_PORT = 1883
MQTT_CLIENT_NAME = "ServicoAnalise"
SUBSCRIBE_TOPIC_RAW_IMG = "fluxoai/imagem/bruta"
SUBSCRIBE_TOPIC_DETECTIONS = "fluxoai/deteccoes"
PUBLISH_TOPIC_ANALYZED_IMG = "fluxoai/imagem/analisada"

# Parâmetros visuais
JPEG_QUALITY = 40
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
BOUNDING_BOX_COLORS = {
    "person": (0, 255, 0),
    "suspeito": (0, 0, 255)
}

# Lógica de suspeição
LOITERING_TIME_THRESHOLD = 10  # segundos parado para ser considerado suspeito
TARGET_LABEL = "person"

# =====================================
# LOGGING
# =====================================
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# =====================================
# FUNÇÕES AUXILIARES
# =====================================

def calculate_center(xmin, ymin, xmax, ymax):
    """Calcula o centro de uma bounding box (x, y)."""
    return int((xmin + xmax) / 2), int((ymin + ymax) / 2)


def is_stationary(old_center, new_center, threshold=15):
    """Verifica se o objeto está praticamente parado."""
    dx = abs(old_center[0] - new_center[0])
    dy = abs(old_center[1] - new_center[1])
    return dx < threshold and dy < threshold


def update_tracking(detections):
    """Atualiza o rastreamento e detecta pessoas paradas (suspeitas)."""
    global latest_tracking_data, latest_detections_for_draw

    current_time = time.time()
    new_tracking_data = {}
    updated_detections = []

    for det in detections:
        if det["label"] != TARGET_LABEL or det["score"] < 0.5:
            continue

        # Garante a ordem correta das coordenadas (ymin, xmin, ymax, xmax)
        ymin, xmin, ymax, xmax = det["box_pixels"]
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

        # Ignora detecções muito pequenas (ruído)
        if (xmax - xmin) < 30 or (ymax - ymin) < 30:
            continue

        center = calculate_center(xmin, ymin, xmax, ymax)
        matched_id = None

        # Associa com o tracking anterior
        for obj_id, data in latest_tracking_data.items():
            if is_stationary(data["center"], center):
                matched_id = obj_id
                break

        if matched_id:
            obj_data = latest_tracking_data[matched_id]
            obj_data["center"] = center
            obj_data["last_seen"] = current_time

            # Detecta ociosidade
            if is_stationary(obj_data["initial_center"], center):
                if (current_time - obj_data["start_time"]) > LOITERING_TIME_THRESHOLD:
                    obj_data["suspeito"] = True
            else:
                obj_data["initial_center"] = center
                obj_data["start_time"] = current_time
                obj_data["suspeito"] = False
        else:
            matched_id = len(latest_tracking_data) + len(new_tracking_data) + 1
            obj_data = {
                "center": center,
                "initial_center": center,
                "start_time": current_time,
                "last_seen": current_time,
                "suspeito": False
            }

        new_tracking_data[matched_id] = obj_data

        updated_detections.append({
            "id": matched_id,
            "label": "suspeito" if obj_data["suspeito"] else TARGET_LABEL,
            "score": det["score"],
            "box_pixels": [ymin, xmin, ymax, xmax]
        })

    # Remove rastros antigos (sem atualização há 10s)
    for obj_id, data in list(latest_tracking_data.items()):
        if current_time - data["last_seen"] < 10:
            new_tracking_data[obj_id] = data

    latest_tracking_data = new_tracking_data
    latest_detections_for_draw = updated_detections


def draw_detections(frame, detections):
    """Desenha boxes e contagens na imagem."""
    loitering_count = 0

    for det in detections:
        ymin, xmin, ymax, xmax = det["box_pixels"]
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

        label = det["label"]
        color = BOUNDING_BOX_COLORS.get(label, (255, 255, 255))
        if label == "suspeito":
            loitering_count += 1

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, f"{label} {det['score']:.2f}", (xmin, max(15, ymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"Suspeitos: {loitering_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, loitering_count


# =====================================
# MQTT CALLBACKS
# =====================================

def on_message_raw_img(client, userdata, msg):
    """Recebe frames brutos via MQTT."""
    global latest_raw_frame, lock
    np_arr = np.frombuffer(msg.payload, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    with lock:
        latest_raw_frame = frame


def on_message_detections(client, userdata, msg):
    """Recebe detecções e atualiza o tracking."""
    detections = json.loads(msg.payload.decode())
    with lock:
        update_tracking(detections)


def publish_heartbeat():
    """Publica frames analisados periodicamente, com uso de CPU reduzido."""
    global latest_raw_frame, latest_detections_for_draw, lock, mqtt_client

    placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, "Aguardando frames e análise...", (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    _, encodedImage = cv2.imencode(".jpg", placeholder_frame)
    placeholder_bytes = bytearray(encodedImage)

    last_frame_hash = None
    last_pub_time = 0
    min_interval = 0.2  # 5 FPS máximo

    while True:
        frame_to_publish = None
        bytes_to_publish = None
        current_loitering_count = 0

        with lock:
            if latest_raw_frame is not None:
                frame_to_publish = latest_raw_frame.copy()
                frame_to_publish, current_loitering_count = draw_detections(frame_to_publish, latest_detections_for_draw)

        if frame_to_publish is not None:
            frame_hash = hash(frame_to_publish.tobytes())
            if frame_hash != last_frame_hash or (time.time() - last_pub_time) > 1:
                _, encodedImage = cv2.imencode(".jpg", frame_to_publish, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                bytes_to_publish = bytearray(encodedImage)
                last_frame_hash = frame_hash
                last_pub_time = time.time()
        else:
            bytes_to_publish = placeholder_bytes

        if bytes_to_publish and mqtt_client and mqtt_client.is_connected():
            mqtt_client.publish(PUBLISH_TOPIC_ANALYZED_IMG, bytes_to_publish, qos=0)
            logging.debug(f"Frame publicado. Suspeitos: {current_loitering_count}")

        time.sleep(min_interval)


# =====================================
# INICIALIZAÇÃO DO MQTT
# =====================================

def start_mqtt():
    global mqtt_client
    mqtt_client = mqtt.Client(MQTT_CLIENT_NAME)
    mqtt_client.on_message = lambda c, u, m: (
        on_message_raw_img(c, u, m) if m.topic == SUBSCRIBE_TOPIC_RAW_IMG else on_message_detections(c, u, m)
    )
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.subscribe(SUBSCRIBE_TOPIC_RAW_IMG)
    mqtt_client.subscribe(SUBSCRIBE_TOPIC_DETECTIONS)
    mqtt_client.loop_start()


# =====================================
# MAIN
# =====================================
if __name__ == "__main__":
    logging.info("Iniciando ServicoAnalise otimizado...")
    start_mqtt()
    publish_heartbeat()
