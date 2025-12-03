# Documentação: Aplicação Principal Flask para o SistemaDMS
# (Atualizado para incluir Gestor MQTT)

import cv2
import time
import os
import numpy as np
import threading
import logging
from flask import (
    Flask,
    Response,
    render_template,
    jsonify,
    request,
    send_from_directory,
)
import queue
import json
import signal

# Importa os nossos módulos
from camera_thread import CameraThread
from dms_base import BaseMonitor
from dms_mediapipe import MediaPipeMonitor
from event_handler import EventHandler
from mqtt_uploader import MQTTUploader # <-- NOVO IMPORT

try:
    from waitress import serve
    HAS_WAITRESS = True
except ImportError:
    HAS_WAITRESS = False

cv2.setUseOptimized(True)

# --- Configuração do Logging ---
default_log_level_str = os.environ.get("LOG_LEVEL", "WARNING").upper()
log_levels_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
log_level = log_levels_map.get(default_log_level_str, logging.WARNING)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - DMS - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)


CONFIG_DIR = "/app/config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "settings.json")


def load_config():
    """Carrega o arquivo settings.json se ele existir."""
    if not os.path.exists(CONFIG_FILE):
        logging.warning(f"Arquivo '{CONFIG_FILE}' não encontrado. Usando padrões.")
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            logging.info(f"Configuração carregada de '{CONFIG_FILE}'.")
            return config
    except Exception as e:
        logging.error(f"Erro ao carregar '{CONFIG_FILE}': {e}. Usando padrões.")
        return {}


def save_config(settings_dict):
    """Salva o dicionário de configurações em settings.json."""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(settings_dict, f, indent=4)
        logging.info(f"Configuração salva em '{CONFIG_FILE}'.")
    except Exception as e:
        logging.error(f"Erro ao salvar '{CONFIG_FILE}': {e}", exc_info=True)


# --- Carrega Configurações ---
config_from_file = load_config()


# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "0")
FRAME_WIDTH_DISPLAY = 640
FRAME_HEIGHT_DISPLAY = 480
JPEG_QUALITY = 60
TARGET_FPS = 30
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
EVENT_QUEUE_MAX_SIZE = 100

INITIAL_ROTATION = int(
    os.environ.get("ROTATE_FRAME", config_from_file.get("rotation", "0"))
)
DETECTION_BACKEND = "MEDIAPIPE"

# --- NOVA VARIAVEL DE CONTROLO ---
# Converte "0" ou "false" para False, e "1" ou "true" para True
ENABLE_VIDEO_STREAM = os.environ.get("ENABLE_VIDEO_STREAM", "0").lower() in ("true", "1", "t", "y")
if not ENABLE_VIDEO_STREAM:
    logging.warning("!!! ATENCAO: O stream de vídeo está DESABILITADO via ENABLE_VIDEO_STREAM=0.")
    logging.warning("!!! A rota /video_feed servirá apenas um placeholder estático.")
# -----------------------------------

# Padrões DMS
DEFAULT_EAR_THRESHOLD = config_from_file.get("ear_threshold", 0.30)
DEFAULT_EAR_FRAMES = config_from_file.get("ear_frames", 2)
DEFAULT_EAR_CALIB_FACTOR = config_from_file.get("ear_calibration_factor", 0.80)
DEFAULT_MAR_THRESHOLD = config_from_file.get("mar_threshold", 0.40)
DEFAULT_MAR_FRAMES = config_from_file.get("mar_frames", 2)
DEFAULT_PHONE_ENABLED = config_from_file.get("phone_detection_enabled", True)
DEFAULT_PHONE_CONF = config_from_file.get("phone_confidence", 0.30)
DEFAULT_PHONE_FRAMES = config_from_file.get("phone_frames", 1)

# --- NOVO: Padrões MQTT ---
DEFAULT_MQTT_ENABLED = config_from_file.get("mqtt_enabled", False)
DEFAULT_MQTT_BROKER = config_from_file.get("mqtt_broker", "broker.hivemq.com")
DEFAULT_MQTT_PORT = config_from_file.get("mqtt_port", 1883)
# Gera um ID de dispositivo único se não estiver configurado
DEFAULT_MQTT_DEVICE_ID = config_from_file.get(
    "mqtt_device_id", f"dms_device_{int(time.time()) % 10000}"
)
DEFAULT_MQTT_FLEET_ID = config_from_file.get("mqtt_fleet_id", "default_fleet")
DEFAULT_MQTT_USERNAME = config_from_file.get("mqtt_username", "")
DEFAULT_MQTT_PASSWORD = config_from_file.get("mqtt_password", "")
DEFAULT_MQTT_RETENTION_DAYS = config_from_file.get("mqtt_retention_days", 10)


# --- Gerenciador de Brilho Automático (AutoBrightnessManager) ---
class AutoBrightnessManager:
    """
    Gerencia o ajuste automático de brilho com base no sucesso da deteção.
    """
    def __init__(self, cam_thread_ref: CameraThread):
        self.cam_thread = cam_thread_ref
        self.enabled = False
        self.consecutive_failures = 0
        
        # --- Constantes de Ajuste ---
        self.FAILURE_THRESHOLD = 50  # Nº de frames sem rosto antes de agir
        self.BRIGHTNESS_STEP = 5.0   # O "tamanho" do passo de ajuste
        self.BRIGHTNESS_MIN = 0.0    # Valor mínimo de brilho
        self.BRIGHTNESS_MAX = 60.0   # Valor máximo de brilho (ajuste!)
        # ---------------------------

        self.current_brightness = 17.0 # Padrão inicial
        self.search_direction = +1     # Começa aumentando
        
        if self.cam_thread and not self.cam_thread.is_rtsp:
            try:
                self.current_brightness = self.cam_thread.get_brightness()
                logging.info(f"AutoBrightnessManager: Brilho inicial lido da câmara: {self.current_brightness}")
            except Exception as e:
                logging.warning(f"AutoBrightnessManager: Não foi possível ler brilho inicial. Usando padrão. {e}")
        else:
            logging.warning("AutoBrightnessManager: Câmara RTSP ou indisponível. Auto-brilho não funcionará.")


    def set_enabled(self, enabled: bool):
        """Ativa ou desativa o modo automático."""
        if self.cam_thread.is_rtsp:
             self.enabled = False # Garante que está desligado para RTSP
             if enabled:
                 logging.warning("AutoBrightness: Não pode ser ativado, fonte é RTSP.")
             return

        if enabled == self.enabled:
            return # Sem mudança

        self.enabled = enabled
        if self.enabled:
            # Ao ligar, lê o brilho atual como ponto de partida
            self.current_brightness = self.cam_thread.get_brightness()
            self.consecutive_failures = 0
            self.search_direction = +1
            logging.info(f"AutoBrightness: ATIVADO. Iniciando do brilho atual: {self.current_brightness}")
        else:
            logging.info("AutoBrightness: DESATIVADO.")

    def is_enabled(self) -> bool:
        """Verifica se o modo automático está ativo."""
        return self.enabled

    def update_status(self, face_found: bool):
        """
        Método principal chamado a cada frame pelo detection_loop.
        """
        if not self.enabled:
            return # Não faz nada se estiver desligado

        if face_found:
            # Sucesso! Reseta o contador de falhas.
            self.consecutive_failures = 0
            return

        # --- Falha na Deteção ---
        self.consecutive_failures += 1

        # Se ainda não atingimos o limite, espera mais
        if self.consecutive_failures < self.FAILURE_THRESHOLD:
            return
            
        # --- HORA DE AGIR ---
        self.consecutive_failures = 0 
        
        self.current_brightness += (self.search_direction * self.BRIGHTNESS_STEP)

        if self.current_brightness >= self.BRIGHTNESS_MAX:
            self.current_brightness = self.BRIGHTNESS_MAX
            self.search_direction = -1 # Inverte a direção
            logging.debug("AutoBrightness: Atingiu BRILHO MÁXIMO. Invertendo direção.")

        elif self.current_brightness <= self.BRIGHTNESS_MIN:
            self.current_brightness = self.BRIGHTNESS_MIN
            self.search_direction = +1 # Inverte a direção
            logging.debug("AutoBrightness: Atingiu BRILHO MÍNIMO. Invertendo direção.")

        logging.info(f"AutoBrightness: Rosto não detectado. Ajustando brilho para {self.current_brightness}")
        
        try:
            self.cam_thread.update_brightness(self.current_brightness)
        except Exception as e:
             logging.error(f"AutoBrightness: Erro ao definir brilho: {e}")


# --- Variáveis Globais ---
output_frame_display = None
output_frame_lock = threading.Lock()
status_data_global = {"ear": "-", "mar": "-", "yaw": "-", "pitch": "-", "roll": "-"}
status_data_lock = threading.Lock()
stop_event = threading.Event()

cam_thread = None
detection_thread = None
event_handler = None
event_queue = None
dms_monitor: BaseMonitor = None
brightness_manager: "AutoBrightnessManager" = None
mqtt_thread: "MQTTUploader" = None # <-- NOVO GLOBAL

app = Flask(__name__)


# --- Funções Auxiliares (create_placeholder_frame) ---
def create_placeholder_frame(text="Aguardando camera..."):
    frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        cv2.putText(
            frame, text, (30, FRAME_HEIGHT_DISPLAY // 2), font, 1, (255, 255, 255), 2
        )
    except cv2.error as e:
        logging.warning(f"Erro ao desenhar texto no placeholder: {e}")
    return frame


# --- Threads Principais (detection_loop) ---
def detection_loop(cam_thread_ref, dms_monitor_ref: BaseMonitor, event_queue_ref):
    global output_frame_display, status_data_global, brightness_manager
    logging.info(
        f">>> Loop de deteção (Backend: {DETECTION_BACKEND}) "
        f"iniciado (Alvo: {TARGET_FPS} FPS)."
    )
    frame_count = 0

    while not stop_event.is_set():
        start_time = time.time()
        logging.debug("DetectionLoop: Topo do loop.")

        if not cam_thread_ref or not cam_thread_ref.is_alive():
            logging.error("!!! Thread da câmara não ativa. A parar.")
            break

        logging.debug("DetectionLoop: A chamar get_frame()...")
        frame = cam_thread_ref.get_frame()

        if frame is None:
            if not stop_event.is_set():
                logging.debug("Frame não recebido.")
            stop_event.wait(timeout=0.1)
            continue
        logging.debug("DetectionLoop: get_frame() retornou frame.")

        try:
            logging.debug("DetectionLoop: A converter BGR p/ RGB...")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            logging.debug("DetectionLoop: A chamar process_frame()...")
            if dms_monitor_ref is None:
                logging.error("!!! dms_monitor_ref (BaseMonitor) não inicializado!")
                stop_event.wait(timeout=1.0)
                continue

            processed_frame, events, status_data, face_found = dms_monitor_ref.process_frame(
                frame.copy(), frame_rgb
            )
            logging.debug("DetectionLoop: process_frame() retornou.")

            if brightness_manager:
                brightness_manager.update_status(face_found)

            logging.debug("DetectionLoop: A adquirir output_frame_lock...")
            with output_frame_lock:
                logging.debug("DetectionLoop: output_frame_lock adquirido.")
                output_frame_display = processed_frame.copy()
            logging.debug("DetectionLoop: output_frame_lock libertado.")

            frame_count += 1
            if frame_count % 100 == 0:
                logging.debug(f"Loop deteção: Frame {frame_count}.")

            logging.debug("DetectionLoop: A adquirir status_data_lock...")
            with status_data_lock:
                logging.debug("DetectionLoop: status_data_lock adquirido.")
                status_data_global = status_data.copy()
            logging.debug("DetectionLoop: status_data_lock libertado.")

            if events:
                logging.debug(f"DetectionLoop: A processar {len(events)} eventos...")
                for event in events:
                    try:
                        event_queue_ref.put(
                            {"event_data": event, "frame": frame.copy()},
                            block=False,
                            timeout=0.1,
                        )
                    except queue.Full:
                        logging.warning("!!! Fila cheia.")
                    except Exception as q_err:
                        logging.error(f"Erro fila: {q_err}")

        except cv2.error as cv_err:
            logging.error(f"Erro OpenCV: {cv_err}", exc_info=True)
            stop_event.wait(timeout=1.0)
        except Exception as e:
            logging.error(f"!!! Erro no process_frame: {e}", exc_info=True)
            stop_event.wait(timeout=1.0)

        processing_time = time.time() - start_time
        wait_time = TARGET_FRAME_TIME - processing_time
        logging.debug(
            f"Tempo: {processing_time:.3f}s, Espera: {max(0, wait_time):.3f}s"
        )

        if wait_time > 0:
            logging.debug(f"DetectionLoop: A esperar {wait_time:.3f}s...")
            stop_event.wait(timeout=wait_time)
        else:
            logging.debug("DetectionLoop: Loop lento, pausa (0.01s).")
            stop_event.wait(timeout=0.01)

    logging.info(">>> Loop de deteção terminado.")


# --- Servidor Web Flask (Rotas: /, /alerts, generate_video_stream, video_feed) ---
@app.route("/")
def index():
    cam_source_desc = cam_thread.source_description if cam_thread else "Indisponível"
    return render_template(
        "index.html",
        source_desc=cam_source_desc,
        width=FRAME_WIDTH_DISPLAY,
        height=FRAME_HEIGHT_DISPLAY,
        active_backend=DETECTION_BACKEND,
        video_stream_enabled=ENABLE_VIDEO_STREAM
    )


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")


def generate_video_stream():
    global output_frame_display
    placeholder = create_placeholder_frame()
    last_frame_time = time.time()
    frame_yield_count = 0
    logging.debug("generate_video_stream: Iniciando.")

    while not stop_event.is_set():
        frame_to_encode = None
        use_placeholder = False
        logging.debug("generate_video_stream: A adquirir output_frame_lock...")
        with output_frame_lock:
            logging.debug("generate_video_stream: output_frame_lock adquirido.")
            if output_frame_display is not None:
                frame_to_encode = output_frame_display.copy()
                logging.debug("generate_video_stream: Usando frame processado.")
            else:
                frame_to_encode = placeholder.copy()
                use_placeholder = True
                logging.debug("generate_video_stream: Usando placeholder.")
        logging.debug("generate_video_stream: output_frame_lock libertado.")

        if frame_to_encode is None:
            frame_to_encode = placeholder.copy()
            use_placeholder = True
        try:
            if not isinstance(frame_to_encode, np.ndarray) or frame_to_encode.size == 0:
                frame_to_encode = placeholder.copy()
                use_placeholder = True

            logging.debug("generate_video_stream: A codificar frame...")
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, encode_param)

            if not flag:
                (flag, encodedImage) = cv2.imencode(
                    ".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                )
                if not flag:
                    logging.error("generate_video_stream: Falha codificar placeholder.")
                    stop_event.wait(timeout=0.1)
                    continue
            frame_bytes = bytearray(encodedImage)
            logging.debug(
                f"generate_video_stream: A enviar frame {frame_yield_count} "
                f"({len(frame_bytes)} bytes)."
            )
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            frame_yield_count += 1
        except GeneratorExit:
            logging.info("generate_video_stream: Cliente desconectou.")
            break
        except cv2.error as e:
            logging.error(f"generate_video_stream: Erro OpenCV: {e}", exc_info=True)
            stop_event.wait(timeout=0.5)
        except Exception as e:
            logging.error(f"generate_video_stream: Erro inesperado: {e}", exc_info=True)
            break

        target_stream_time = 1 / 15
        current_time = time.time()
        sleep_time = target_stream_time - (current_time - last_frame_time)
        if sleep_time > 0:
            stop_event.wait(timeout=sleep_time)
        last_frame_time = time.time()

    logging.info(f"generate_video_stream: Terminado após {frame_yield_count} frames.")


@app.route("/video_feed")
def video_feed():
    logging.debug("Rota /video_feed acedida.")

    # --- MUDANÇA PRINCIPAL ---
    if not ENABLE_VIDEO_STREAM:
        # Se o stream estiver desabilitado, não inicie o gerador.
        # Em vez disso, sirva um único frame de placeholder e saia.
        logging.debug("Rota /video_feed: Stream desabilitado. Servindo placeholder estático.")

        placeholder = create_placeholder_frame(text="Stream desabilitado")
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        (flag, encodedImage) = cv2.imencode(".jpg", placeholder, encode_param)

        if not flag:
             logging.error("Falha ao codificar placeholder estático.")
             return "Error encoding placeholder", 500

        # Retorna o JPEG estático
        return Response(encodedImage.tobytes(), mimetype="image/jpeg")
    # --- FIM DA MUDANÇA ---

    # Comportamento original (só executa se ENABLE_VIDEO_STREAM=1)
    if not cam_thread or not cam_thread.is_alive():
        logging.error("Rota /video_feed: Thread câmara não ativa.")
        return "Camera thread not running", 503

    return Response(
        generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --- Rotas da API (api_config, api_alerts, serve_alert_image) ---

@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    global dms_monitor, brightness_manager, mqtt_thread # <-- mqtt_thread adicionado
    
    logging.debug(f"Rota /api/config (Método: {request.method})")
    
    # Verifica se os serviços estão prontos
    if (
        dms_monitor is None 
        or not cam_thread 
        or not event_queue 
        or brightness_manager is None
        or mqtt_thread is None # <-- NOVO
    ):
        logging.warning("/api/config: Serviço não inicializado.")
        return jsonify({"error": "Service not fully initialized"}), 503

    if request.method == "GET":
        try:
            current_settings = dms_monitor.get_settings()
            
            # Configurações da Câmara
            current_settings["brightness"] = cam_thread.get_brightness()
            current_settings["rotation"] = cam_thread.get_rotation()
            current_settings["active_backend"] = DETECTION_BACKEND
            current_settings["auto_brightness"] = brightness_manager.is_enabled()
            
            # --- NOVO: Configurações MQTT ---
            with mqtt_thread.config_lock:
                current_settings["mqtt_enabled"] = mqtt_thread.config.get("mqtt_enabled")
                current_settings["mqtt_broker"] = mqtt_thread.config.get("mqtt_broker")
                current_settings["mqtt_port"] = mqtt_thread.config.get("mqtt_port")
                current_settings["mqtt_device_id"] = mqtt_thread.config.get("mqtt_device_id")
                current_settings["mqtt_fleet_id"] = mqtt_thread.config.get("mqtt_fleet_id")
                current_settings["mqtt_username"] = mqtt_thread.config.get("mqtt_username")
                current_settings["mqtt_password"] = mqtt_thread.config.get("mqtt_password")
                current_settings["mqtt_retention_days"] = mqtt_thread.config.get("mqtt_retention_days")
            # -------------------------------

            # Status Runtime
            logging.debug("api_config GET: Lock status...")
            with status_data_lock:
                logging.debug("api_config GET: Lock status OK.")
                current_settings["status"] = status_data_global.copy()
            logging.debug("api_config GET: Lock status libertado.")
            
            try:
                current_settings["queue_depth"] = event_queue.qsize()
                current_settings["queue_max_size"] = event_queue.maxsize
            except Exception as e:
                logging.warning(f"Erro obter tamanho fila: {e}")
                current_settings["queue_depth"] = -1
                current_settings["queue_max_size"] = EVENT_QUEUE_MAX_SIZE

            logging.debug(f"/api/config GET: Retornando {current_settings}")
            return jsonify(current_settings)
            
        except Exception as e:
            logging.error(f"Erro inesperado /api/config GET: {e}", exc_info=True)
            return jsonify({"error": "Internal server error reading config"}), 500

    elif request.method == "POST":
        try:
            new_settings = request.json
            logging.debug(f"/api/config POST: Recebido {new_settings}")
            if not new_settings:
                return jsonify({"success": False, "error": "No data received"}), 400

            # --- Gestão Auto-Brilho ---
            if "auto_brightness" in new_settings:
                try:
                    brightness_manager.set_enabled(bool(new_settings["auto_brightness"]))
                except Exception as e:
                     logging.error(f"Erro ao definir auto_brightness: {e}")
                if new_settings["auto_brightness"]:
                    new_settings.pop("brightness", None)
            
            # --- Atualiza Módulos ---
            
            # 1. Configurações DMS
            dms_success = dms_monitor.update_settings(new_settings)

            # 2. Configurações Câmara
            cam_success = True
            try:
                if "brightness" in new_settings:
                    cam_thread.update_brightness(new_settings["brightness"])
                if "rotation" in new_settings:
                    cam_thread.update_rotation(new_settings["rotation"])
            except Exception as e:
                logging.error(f"Erro atualizar conf câmara: {e}")
                cam_success = False

            # --- NOVO: 3. Configurações MQTT ---
            # Filtra apenas as chaves MQTT que foram recebidas
            mqtt_keys = [
                "mqtt_enabled", "mqtt_broker", "mqtt_port", "mqtt_device_id",
                "mqtt_fleet_id", "mqtt_username", "mqtt_password", "mqtt_retention_days"
            ]
            mqtt_updates = {k: new_settings[k] for k in mqtt_keys if k in new_settings}
            
            if mqtt_updates:
                try:
                    # Converte tipos de dados
                    if "mqtt_port" in mqtt_updates:
                        mqtt_updates["mqtt_port"] = int(mqtt_updates["mqtt_port"])
                    if "mqtt_retention_days" in mqtt_updates:
                        mqtt_updates["mqtt_retention_days"] = int(mqtt_updates["mqtt_retention_days"])
                    if "mqtt_enabled" in mqtt_updates:
                        mqtt_updates["mqtt_enabled"] = bool(mqtt_updates["mqtt_enabled"])
                        
                    mqtt_thread.update_config(mqtt_updates)
                except Exception as e:
                    logging.error(f"Erro ao atualizar config MQTT: {e}", exc_info=True)
            # ------------------------------------

            if dms_success and cam_success:
                logging.info("/api/config POST: Configurações atualizadas.")

                # --- Salva a configuração persistente ---
                try:
                    # Pega todas as configurações atuais de todos os módulos
                    all_current_settings = dms_monitor.get_settings()
                    all_current_settings["brightness"] = cam_thread.get_brightness()
                    all_current_settings["rotation"] = cam_thread.get_rotation()
                    all_current_settings["auto_brightness"] = brightness_manager.is_enabled()
                    
                    # Adiciona configs MQTT
                    with mqtt_thread.config_lock:
                        for k in mqtt_keys:
                            all_current_settings[k] = mqtt_thread.config.get(k)
                    
                    save_config(all_current_settings)
                except Exception as e:
                    logging.error(
                        f"Falha ao salvar config persistente: {e}", exc_info=True
                    )
                # -----------------------------------------------

                return jsonify({"success": True})
            else:
                error_msg = f"Failed (DMS: {dms_success}, Cam: {cam_success})"
                logging.warning(f"/api/config POST: Falha: {error_msg}")
                return jsonify({"success": False, "error": error_msg}), 500
        except Exception as e:
            logging.error(f"Erro inesperado /api/config POST: {e}", exc_info=True)
            return jsonify({"error": "Internal server error updating config"}), 500


@app.route("/api/alerts", methods=["GET"])
def api_alerts():
    logging.debug("Rota /api/alerts acedida.")
    if not event_handler:
        logging.warning("/api/alerts: Gestor eventos não init.")
        return jsonify({"error": "Event handler not initialized"}), 503
    try:
        alerts_list = event_handler.get_alerts(limit=50)
        logging.debug(f"/api/alerts: Retornando {len(alerts_list)} alertas.")
        return jsonify(alerts_list)
    except Exception as e:
        logging.error(f"Erro ler alertas SQLite: {e}", exc_info=True)
        return jsonify({"error": "Failed to read alerts from database"}), 500


@app.route("/alerts/images/<path:filepath>")
def serve_alert_image(filepath):
    logging.debug(f"Rota /alerts/images: {filepath}")
    if not event_handler:
        logging.warning(f"/alerts/images: Gestor eventos não init ({filepath}).")
        return "Event handler not initialized", 503
    image_base_path = os.path.join(event_handler.save_path, "images")
    safe_path = os.path.abspath(os.path.join(image_base_path, filepath))
    if not safe_path.startswith(image_base_path):
        logging.warning(f"Acesso inválido /alerts/images: {filepath}")
        return "Invalid path", 400
    if not os.path.isfile(safe_path):
        logging.warning(f"Imagem não encontrada /alerts/images: {safe_path}")
        return "Image not found", 404
    try:
        logging.debug(f"A servir imagem: {safe_path}")
        return send_from_directory(
            os.path.dirname(safe_path), os.path.basename(safe_path)
        )
    except Exception as e:
        logging.error(f"Erro servir imagem '{filepath}': {e}", exc_info=True)
        return "Internal server error", 500


@app.route("/alerts/videos/<path:filepath>")
def serve_alert_video(filepath):
    """Rota para servir ficheiros de vídeo."""
    # ... (verificações de segurança mantêm-se iguais) ...
    video_base_path = os.path.join(event_handler.save_path, "videos")
    safe_path = os.path.abspath(os.path.join(video_base_path, filepath))

    if not safe_path.startswith(video_base_path): return "Invalid path", 400
    if not os.path.isfile(safe_path): return "Video not found", 404

    try:
        # Detecta extensão para decidir o MIME type correto
        if filepath.endswith('.webm'):
            mimetype = 'video/webm'
        else:
            mimetype = 'video/mp4'

        return send_from_directory(
            os.path.dirname(safe_path),
            os.path.basename(safe_path),
            mimetype=mimetype
        )
    except Exception as e:
        logging.error(f"Erro servir vídeo: {e}")
        return "Internal error", 500

# --- Encerramento Gracioso (shutdown_handler) ---
def shutdown_handler(signum, frame):
    if not stop_event.is_set():
        logging.info(f">>> Sinal {signal.Signals(signum).name} recebido. A encerrar...")
        stop_event.set()


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        logging.info(
            f">>> Serviço DMS (Backend: {DETECTION_BACKEND}) a iniciar... "
            f"(Log: {logging.getLevelName(logging.getLogger().level)})"
        )

        event_queue = queue.Queue(maxsize=EVENT_QUEUE_MAX_SIZE)
        event_handler = EventHandler(queue=event_queue, stop_event=stop_event)
        event_handler.start()

        # --- NOVO: Inicia o MQTT Uploader ---
        logging.info("A inicializar o MQTTUploader...")
        mqtt_initial_config = {
            "mqtt_enabled": DEFAULT_MQTT_ENABLED,
            "mqtt_broker": DEFAULT_MQTT_BROKER,
            "mqtt_port": DEFAULT_MQTT_PORT,
            "mqtt_device_id": DEFAULT_MQTT_DEVICE_ID,
            "mqtt_fleet_id": DEFAULT_MQTT_FLEET_ID,
            "mqtt_username": DEFAULT_MQTT_USERNAME,
            "mqtt_password": DEFAULT_MQTT_PASSWORD,
            "mqtt_retention_days": DEFAULT_MQTT_RETENTION_DAYS,
        }
        # A var global 'mqtt_thread' é definida aqui
        mqtt_thread = MQTTUploader(
            event_handler_ref=event_handler,
            stop_event=stop_event,
            initial_config=mqtt_initial_config
        )
        mqtt_thread.start()
        # -----------------------------------

        frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)

        default_dms_settings = {
            "ear_threshold": DEFAULT_EAR_THRESHOLD,
            "ear_frames": DEFAULT_EAR_FRAMES,
            "ear_calibration_factor": DEFAULT_EAR_CALIB_FACTOR,
            "mar_threshold": DEFAULT_MAR_THRESHOLD,
            "mar_frames": DEFAULT_MAR_FRAMES,
            "phone_detection_enabled": DEFAULT_PHONE_ENABLED,
            "phone_confidence": DEFAULT_PHONE_CONF,
            "phone_frames": DEFAULT_PHONE_FRAMES,
            "calibration_state": config_from_file.get("calibration_state", "IDLE")
        }

        logging.info("A carregar o MediaPipeMonitor...")
        dms_monitor = MediaPipeMonitor(
            frame_size=frame_size,
            stop_event=stop_event,
            default_settings=default_dms_settings,
        )

        cam_thread = CameraThread(
            VIDEO_SOURCE,
            frame_width=FRAME_WIDTH_DISPLAY,
            frame_height=FRAME_HEIGHT_DISPLAY,
            rotation_degrees=INITIAL_ROTATION,
            stop_event=stop_event,
        )
        event_handler.set_camera_thread(cam_thread)
        cam_thread.start()

        logging.info("A aguardar o primeiro frame...")
        start_wait_cam = time.time()
        while cam_thread.get_frame() is None and cam_thread.is_alive():
            if stop_event.wait(timeout=0.2):
                raise SystemExit("Encerrado init câmara.")
            if time.time() - start_wait_cam > 15:
                raise RuntimeError("Timeout câmara.")
        if not cam_thread.is_alive():
            raise RuntimeError("Thread câmara terminou.")
        logging.info(">>> Primeiro frame recebido!")

        logging.info("A inicializar o AutoBrightnessManager...")
        brightness_manager = AutoBrightnessManager(cam_thread)
        if config_from_file.get("auto_brightness", False):
             brightness_manager.set_enabled(True)

        detection_thread = threading.Thread(
            target=detection_loop,
            args=(cam_thread, dms_monitor, event_queue),
            name="DetectionThread",
        )
        detection_thread.daemon = True
        detection_thread.start()

        logging.info(">>> A iniciar servidor web porta 5000...")
        if HAS_WAITRESS:
            logging.info("A usar Waitress.")
            serve(app, host="0.0.0.0", port=5000, threads=8)
        else:
            logging.warning("Waitress não encontrado. A usar Flask dev server.")
            try:
                app.run(
                    host="0.0.0.0",
                    port=5000,
                    debug=False,
                    threaded=True,
                    use_reloader=False,
                )
            except OSError as e:
                logging.error(f"!!! ERRO FATAL Flask: {e}", exc_info=True)
                stop_event.set()
            except Exception as e:
                logging.error(f"!!! ERRO FATAL Flask: {e}", exc_info=True)
                stop_event.set()

    except (KeyboardInterrupt, SystemExit) as e:
        logging.info(f">>> {type(e).__name__} recebido. A encerrar...")
    except RuntimeError as e:
        logging.error(f"!!! ERRO FATAL init: {e}")
    except Exception as e:
        logging.error(f"!!! ERRO FATAL não capturado: {e}", exc_info=True)
    finally:
        if not stop_event.is_set():
            logging.warning("stop_event não estava definido no finally, definindo agora.")
            stop_event.set()

        logging.info(">>> A iniciar encerramento final...")
        threads_to_join = []

        if (
            "detection_thread" in locals()
            and detection_thread
            and detection_thread.is_alive()
        ):
            threads_to_join.append(detection_thread)
        if "cam_thread" in locals() and cam_thread and cam_thread.is_alive():
            threads_to_join.append(cam_thread)
        if "event_handler" in locals() and event_handler and event_handler.is_alive():
            threads_to_join.append(event_handler)
        if "mqtt_thread" in locals() and mqtt_thread and mqtt_thread.is_alive(): # <-- NOVO
            threads_to_join.append(mqtt_thread)

        if (
            "dms_monitor" in locals()
            and dms_monitor
            and hasattr(dms_monitor, "phone_thread")
            and dms_monitor.phone_thread.is_alive()
        ):
            threads_to_join.append(dms_monitor.phone_thread)

        for t in threads_to_join:
            logging.info(f"A aguardar thread '{t.name}'...")
            timeout = 2 if getattr(t, "daemon", False) else 5
            t.join(timeout=timeout)

            if t.is_alive():
                logging.warning(f"!!! Timeout ao esperar thread '{t.name}'.")

        logging.info(">>> Serviço DMS terminado.")