# Documentação: Aplicação Principal Flask para o SistemaDMS
# (VERSÃO: MediaPipe + YOLO)
# (MODIFICADO: Persistência de config e graceful shutdown)

import cv2
import time
import os
import numpy as np
import threading
import logging
# import sys (F401 - Removido)
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
# from datetime import datetime (F401 - Removido)
import signal

# Importa os nossos módulos
from camera_thread import CameraThread
from dms_base import BaseMonitor
from dms_mediapipe import MediaPipeMonitor
from event_handler import EventHandler

try:
    from waitress import serve

    HAS_WAITRESS = True
except ImportError:
    HAS_WAITRESS = False

cv2.setUseOptimized(True)

# --- Configuração do Logging ---
default_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = logging.DEBUG if default_log_level == "DEBUG" else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - DMS - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)


# --- (NOVO) Lógica de Persistência de Config (Roadmap 1.1) ---
CONFIG_DIR = "/app/config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "settings.json")


def load_config():
    """Carrega o arquivo settings.json se ele existir."""
    if not os.path.exists(CONFIG_FILE):
        logging.warning(
            f"Arquivo de configuração '{CONFIG_FILE}' não encontrado. Usando padrões."
        )
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            logging.info(f"Configuração carregada de '{CONFIG_FILE}'.")
            return config
    except Exception as e:
        logging.error(
            f"Erro ao carregar '{CONFIG_FILE}': {e}. Usando padrões.", exc_info=True
        )
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
JPEG_QUALITY = 75
TARGET_FPS = 5
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
EVENT_QUEUE_MAX_SIZE = 100

# (MODIFICADO) Usa valores do arquivo salvo ou do env, com fallback para padrões
INITIAL_ROTATION = int(
    os.environ.get("ROTATE_FRAME", config_from_file.get("rotation", "0"))
)
DETECTION_BACKEND = "MEDIAPIPE"

# (MODIFICADO) Padrões são usados se NADA for encontrado no arquivo de config
DEFAULT_EAR_THRESHOLD = config_from_file.get("ear_threshold", 0.25)
DEFAULT_EAR_FRAMES = config_from_file.get("ear_frames", 7)
DEFAULT_MAR_THRESHOLD = config_from_file.get("mar_threshold", 0.40)
DEFAULT_MAR_FRAMES = config_from_file.get("mar_frames", 10)
# (Adiciona padrões de celular aqui também, se existirem no config)
DEFAULT_PHONE_ENABLED = config_from_file.get("phone_detection_enabled", True)
DEFAULT_PHONE_CONF = config_from_file.get("phone_confidence", 0.20)
DEFAULT_PHONE_FRAMES = config_from_file.get("phone_frames", 5)  # (Segundos)

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
        cv2.rectangle(
            frame,
            (10, FRAME_HEIGHT_DISPLAY // 2 - 20),
            (FRAME_WIDTH_DISPLAY - 10, FRAME_HEIGHT_DISPLAY // 2 + 20),
            (50, 50, 50),
            -1,
        )
    return frame


# --- Threads Principais (detection_loop) ---
def detection_loop(cam_thread_ref, dms_monitor_ref: BaseMonitor, event_queue_ref):
    global output_frame_display, status_data_global
    logging.info(
        f">>> Loop de deteção (Backend: {DETECTION_BACKEND}) "
        f"iniciado (Alvo: {TARGET_FPS} FPS)."
    )
    # last_process_time = time.time() (F841 - Removido)
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
            logging.debug("DetectionLoop: A converter p/ cinzento...")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            logging.debug("DetectionLoop: A chamar process_frame()...")
            if dms_monitor_ref is None:
                logging.error("!!! dms_monitor_ref (BaseMonitor) não inicializado!")
                stop_event.wait(timeout=1.0)
                continue

            processed_frame, events, status_data = dms_monitor_ref.process_frame(
                frame.copy(), gray
            )
            logging.debug("DetectionLoop: process_frame() retornou.")

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
            logging.error(
                f"Erro OpenCV (shape: {frame.shape if frame is not None else 'None'}): {cv_err}",
                exc_info=True,
            )
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
            logging.warning("generate_video_stream: frame_to_encode é None, usando placeholder.")
            frame_to_encode = placeholder.copy()
            use_placeholder = True
        try:
            if not isinstance(frame_to_encode, np.ndarray) or frame_to_encode.size == 0:
                logging.error(
                    f"generate_video_stream: Frame inválido (tipo: {type(frame_to_encode)}). "
                    "Usando placeholder."
                )
                frame_to_encode = placeholder.copy()
                use_placeholder = True

            logging.debug("generate_video_stream: A codificar frame...")
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, encode_param)
            logging.debug(
                f"generate_video_stream: Codificação {'bem-sucedida' if flag else 'falhou'}."
            )

            if not flag:
                logging.warning(
                    f"generate_video_stream: Falha codificar (ph={use_placeholder}). "
                    "Tentando placeholder."
                )
                (flag, encodedImage) = cv2.imencode(
                    ".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                )
                if not flag:
                    logging.error(
                        "generate_video_stream: Falha codificar placeholder. Saltando frame."
                    )
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
            logging.error(
                f"generate_video_stream: Erro OpenCV codificar (ph={use_placeholder}, "
                f"shape={frame_to_encode.shape}): {e}",
                exc_info=True,
            )
            stop_event.wait(timeout=0.5)
        except Exception as e:
            logging.error(f"generate_video_stream: Erro inesperado: {e}", exc_info=True)
            break

        target_stream_time = 1 / 20
        current_time = time.time()
        sleep_time = target_stream_time - (current_time - last_frame_time)
        if sleep_time > 0:
            stop_event.wait(timeout=sleep_time)
        last_frame_time = time.time()

    logging.info(f"generate_video_stream: Terminado após {frame_yield_count} frames.")


@app.route("/video_feed")
def video_feed():
    logging.debug("Rota /video_feed acedida.")
    if not cam_thread or not cam_thread.is_alive():
        logging.error("Rota /video_feed: Thread câmara não ativa.")
        return "Camera thread not running", 503
    return Response(
        generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --- Rotas da API (api_config, api_alerts, serve_alert_image) ---
@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    global dms_monitor
    logging.debug(f"Rota /api/config (Método: {request.method})")
    if dms_monitor is None or not cam_thread or not event_queue:
        logging.warning("/api/config: Serviço não inicializado.")
        return jsonify({"error": "Service not fully initialized"}), 503

    if request.method == "GET":
        try:
            current_settings = dms_monitor.get_settings()
            current_settings["brightness"] = cam_thread.get_brightness()
            current_settings["rotation"] = cam_thread.get_rotation()
            current_settings["active_backend"] = DETECTION_BACKEND

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

            dms_success = dms_monitor.update_settings(new_settings)

            cam_success = True
            try:
                if "brightness" in new_settings:
                    cam_thread.update_brightness(new_settings["brightness"])
                if "rotation" in new_settings:
                    cam_thread.update_rotation(new_settings["rotation"])
            except Exception as e:
                logging.error(f"Erro atualizar conf câmara: {e}")
                cam_success = False

            if dms_success and cam_success:
                logging.info("/api/config POST: Configurações atualizadas.")

                # --- (NOVO) Salva a configuração persistente ---
                try:
                    all_current_settings = dms_monitor.get_settings()
                    all_current_settings["brightness"] = cam_thread.get_brightness()
                    all_current_settings["rotation"] = cam_thread.get_rotation()
                    save_config(all_current_settings)
                except Exception as e:
                    logging.error(
                        f"Falha ao salvar config persistente: {e}", exc_info=True
                    )
                # -----------------------------------------------

                return jsonify({"success": True})
            else:
                error_msg = (
                    "Failed settings"
                    + (" (DMS)" if not dms_success else "")
                    + (" (Cam)" if not cam_success else "")
                )
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

        frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)

        # (MODIFICADO) Usa os padrões carregados (do arquivo ou os defaults)
        default_dms_settings = {
            "ear_threshold": DEFAULT_EAR_THRESHOLD,
            "ear_frames": DEFAULT_EAR_FRAMES,
            "mar_threshold": DEFAULT_MAR_THRESHOLD,
            "mar_frames": DEFAULT_MAR_FRAMES,
            "phone_detection_enabled": DEFAULT_PHONE_ENABLED,
            "phone_confidence": DEFAULT_PHONE_CONF,
            "phone_frames": DEFAULT_PHONE_FRAMES,
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

        try:
            dms_monitor.start_yolo_thread(cam_thread)
            logging.info(">>> Thread de deteção de celular (YOLO) iniciada.")
        except AttributeError as e:
            logging.warning(f"Não foi possível iniciar o thread YOLO: {e}")
        except Exception as e:
            logging.error(f"Erro ao iniciar thread YOLO: {e}", exc_info=True)

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
            logging.warning("Waitress não encontrado.")
            logging.warning("A usar Flask dev server.")
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