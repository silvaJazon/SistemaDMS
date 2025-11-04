# Documentação: Aplicação Principal Flask para o SistemaDMS
# (ROADMAP 2.0: Implementado Multiprocessing)
# (CORRIGIDO: Separado 'threading.Event' de 'multiprocessing.Event' para prevenir crash 'cannot pickle lock')
# (CORRIGIDO: Adicionada verificação de 'isinstance' para prevenir crash 'TypeError: str object')

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
import queue # Fila da thread de eventos (interna)
import multiprocessing as mp # (NOVO)
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

# A 'detection_loop' é super leve, 15FPS é fácil
TARGET_FPS = 15
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

# --- (CORRIGIDO) Eventos de Paragem Separados ---
thread_stop_event = threading.Event() # Para threads (Câmara, DetectionLoop, EventHandler)
mp_stop_event = mp.Event()         # Para processos (CV Worker)
# -----------------------------------------------

cam_thread = None
detection_thread = None
event_handler = None
dms_monitor = None # (MODIFICADO) Agora será o 'MediaPipeMonitorProcess'
cv_process = None # (NOVO) O processo de CV

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


# --- (MODIFICADO) Threads Principais (detection_loop) ---
def detection_loop(cam_thread_ref, dms_monitor_ref, event_queue_ref):
    """
    Esta loop agora é 100% LEVE.
    1. Pega o frame da câmara.
    2. Envia o frame para o Processo de CV (não-bloqueante).
    3. Pega os últimos resultados do Processo de CV (não-bloqueante).
    4. Desenha os resultados no frame.
    5. Envia o frame para o stream de vídeo.
    """
    global output_frame_display, status_data_global
    logging.info(
        f">>> Loop de deteção (Modo: Multiprocess) "
        f"iniciado (Alvo: {TARGET_FPS} FPS)."
    )
    
    frame_count = 0

    # (CORRIGIDO) Usa o 'thread_stop_event'
    while not thread_stop_event.is_set():
        start_time = time.time()
        logging.debug("DetectionLoop: Topo do loop.")

        if not cam_thread_ref or not cam_thread_ref.is_alive():
            logging.error("!!! Thread da câmara não ativa. A parar.")
            break
        
        if not dms_monitor_ref or not dms_monitor_ref.is_alive():
            logging.error("!!! Processo de CV não ativo. A parar.")
            thread_stop_event.set() # Termina a aplicação
            break

        logging.debug("DetectionLoop: A chamar get_frame()...")
        frame = cam_thread_ref.get_frame()

        if frame is None:
            # (CORRIGIDO) Usa o 'thread_stop_event'
            if not thread_stop_event.is_set():
                logging.debug("Frame não recebido.")
            thread_stop_event.wait(timeout=0.1)
            continue
        logging.debug("DetectionLoop: get_frame() retornou frame.")

        try:
            # --- (MODIFICADO) PASSO 1: Envia o frame para o processo de CV ---
            # 'process_frame' agora é só um 'put_nowait' para a fila de entrada
            dms_monitor_ref.process_frame(frame.copy())
            
            # --- PASSO 2: Pega os últimos resultados e eventos ---
            # 'get_results' é um 'get_nowait' da fila de saída
            # 'results' pode ter um atraso (ex: 1.0s), mas é o resultado mais recente
            results = dms_monitor_ref.get_results()
            
            processed_frame = frame # Começa com o frame original
            events = []
            status_data = status_data_global # Mantém o status antigo se não houver novo
            
            if results:
                # Se o processo de CV enviou novos dados, atualiza
                status_data, events, annotations = results
                
                logging.debug("DetectionLoop: Novos resultados recebidos do Processo CV.")

                # --- PASSO 3: Desenha os resultados (aqui na thread principal) ---
                # A thread principal (leve) faz o desenho, não o processo (pesado)
                processed_frame = dms_monitor_ref.draw_annotations(frame, annotations)

                # Atualiza o status global
                with status_data_lock:
                    status_data_global = status_data.copy()
            else:
                # Se não há resultados novos (worker está ocupado),
                # apenas desenha as anotações antigas
                logging.debug("DetectionLoop: Sem resultados, a desenhar anotações antigas.")
                processed_frame = dms_monitor_ref.draw_annotations(frame, None)


            # --- PASSO 4: Atualiza o stream de vídeo ---
            with output_frame_lock:
                output_frame_display = processed_frame.copy()

            frame_count += 1
            if frame_count % 100 == 0:
                logging.debug(f"Loop deteção: Frame {frame_count}.")

            # --- PASSO 5: Envia eventos para a fila de eventos (SQLite) ---
            if events:
                logging.debug(f"DetectionLoop: A processar {len(events)} eventos...")
                for event in events:
                    try:
                        # (O frame original 'frame' é usado para a evidência)
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
            thread_stop_event.wait(timeout=1.0)
        except Exception as e:
            logging.error(f"!!! Erro no process_frame: {e}", exc_info=True)
            thread_stop_event.wait(timeout=1.0)

        processing_time = time.time() - start_time
        wait_time = TARGET_FRAME_TIME - processing_time
        logging.debug(
            f"Tempo: {processing_time:.3f}s, Espera: {max(0, wait_time):.3f}s"
        )

        if wait_time > 0:
            logging.debug(f"DetectionLoop: A esperar {wait_time:.3f}s...")
            # (CORRIGIDO) Usa o 'thread_stop_event'
            thread_stop_event.wait(timeout=wait_time)
        else:
            logging.debug("DetectionLoop: Loop lento, pausa (0.01s).")
            # (CORRIGIDO) Usa o 'thread_stop_event'
            thread_stop_event.wait(timeout=0.01)

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

    # (CORRIGIDO) Usa o 'thread_stop_event'
    while not thread_stop_event.is_set():
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
                    thread_stop_event.wait(timeout=0.1)
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
            thread_stop_event.wait(timeout=0.5)
        except Exception as e:
            logging.error(f"generate_video_stream: Erro inesperado: {e}", exc_info=True)
            break

        target_stream_time = 1 / 20
        current_time = time.time()
        sleep_time = target_stream_time - (current_time - last_frame_time)
        if sleep_time > 0:
            # (CORRIGIDO) Usa o 'thread_stop_event'
            thread_stop_event.wait(timeout=sleep_time)
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


# --- (MODIFICADO) Rotas da API (api_config) ---
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
            
            # --- (CORRIGIDO) Verifica se o 'get_settings' falhou ---
            if not isinstance(current_settings, dict):
                 logging.warning(f"api_config GET: Não obteve settings (recebeu {type(current_settings)}). A usar config do ficheiro.")
                 current_settings = load_config() # Fallback 1: Ficheiro
                 if not isinstance(current_settings, dict) or not current_settings:
                    logging.warning("api_config GET: Ficheiro de config vazio. A usar defaults.")
                    # Fallback 2: Defaults
                    current_settings = {
                        "ear_threshold": DEFAULT_EAR_THRESHOLD, "ear_frames": DEFAULT_EAR_FRAMES,
                        "mar_threshold": DEFAULT_MAR_THRESHOLD, "mar_frames": DEFAULT_MAR_FRAMES,
                        "phone_detection_enabled": DEFAULT_PHONE_ENABLED,
                        "phone_confidence": DEFAULT_PHONE_CONF, "phone_frames": DEFAULT_PHONE_FRAMES,
                    }
            # ----------------------------------------------------

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

            # (MODIFICADO) 'update_settings' agora envia para uma fila
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
                    # (MODIFICADO) Pega nos settings que acabámos de enviar
                    settings_to_save = dms_monitor.get_last_sent_settings()
                    if settings_to_save:
                        settings_to_save["brightness"] = cam_thread.get_brightness()
                        settings_to_save["rotation"] = cam_thread.get_rotation()
                        save_config(settings_to_save)
                    else:
                        logging.warning("Não foi possível salvar settings, 'get_last_sent_settings' falhou.")
                        
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


# --- (CORRIGIDO) Encerramento Gracioso (shutdown_handler) ---
def shutdown_handler(signum, frame):
    if not thread_stop_event.is_set():
        logging.info(f">>> Sinal {signal.Signals(signum).name} recebido. A encerrar...")
        thread_stop_event.set()
        mp_stop_event.set() # Também para o processo


# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        logging.info(
            f">>> Serviço DMS (Backend: {DETECTION_BACKEND}) a iniciar... "
            f"(Log: {logging.getLevelName(logging.getLogger().level)})"
        )
        
        # (NOVO) Define o método de arranque para 'spawn' ou 'fork'
        # 'spawn' é mais seguro e compatível com CUDA/GPU
        try:
            mp.set_start_method('spawn') 
            logging.info(">>> Método de multiprocessing definido para 'spawn'.")
        except RuntimeError:
            logging.warning(">>> Método de multiprocessing já definido, a ignorar.")
            pass


        event_queue = queue.Queue(maxsize=EVENT_QUEUE_MAX_SIZE)
        # (CORRIGIDO) Passa o 'thread_stop_event'
        event_handler = EventHandler(queue=event_queue, stop_event=thread_stop_event)
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

        logging.info("A carregar o MediaPipeMonitor (Processo)...")
        # (CORRIGIDO) Passa o 'mp_stop_event'
        dms_monitor = MediaPipeMonitor(
            frame_size=frame_size,
            stop_event=mp_stop_event, # <--- EVENTO DE PROCESSO
            default_settings=default_dms_settings,
        )

        # (CORRIGIDO) Passa o 'thread_stop_event'
        cam_thread = CameraThread(
            VIDEO_SOURCE,
            frame_width=FRAME_WIDTH_DISPLAY,
            frame_height=FRAME_HEIGHT_DISPLAY,
            rotation_degrees=INITIAL_ROTATION,
            stop_event=thread_stop_event, # <--- EVENTO DE THREAD
        )
        cam_thread.start()

        logging.info("A aguardar o primeiro frame...")
        start_wait_cam = time.time()

        while cam_thread.get_frame() is None and cam_thread.is_alive():
            # (CORRIGIDO) Usa o 'thread_stop_event'
            if thread_stop_event.wait(timeout=0.2):
                raise SystemExit("Encerrado init câmara.")
            if time.time() - start_wait_cam > 15:
                raise RuntimeError("Timeout câmara.")

        if not cam_thread.is_alive():
            raise RuntimeError("Thread câmara terminou.")

        logging.info(">>> Primeiro frame recebido!")

        try:
            # (MODIFICADO) 'start_yolo_thread' agora chama-se 'start_process'
            cv_process = dms_monitor.start_process()
            logging.info(">>> Processo de CV (FaceMesh+Hands+YOLO) iniciado.")
        except Exception as e:
            logging.error(f"Erro ao iniciar processo CV: {e}", exc_info=True)
            raise # Erro fatal se o processo de CV não arrancar

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
                thread_stop_event.set()
                mp_stop_event.set()
            except Exception as e:
                logging.error(f"!!! ERRO FATAL Flask: {e}", exc_info=True)
                thread_stop_event.set()
                mp_stop_event.set()

    except (KeyboardInterrupt, SystemExit) as e:
        logging.info(f">>> {type(e).__name__} recebido. A encerrar...")
    except RuntimeError as e:
        logging.error(f"!!! ERRO FATAL init: {e}")
    except Exception as e:
        logging.error(f"!!! ERRO FATAL não capturado: {e}", exc_info=True)
    finally:
        # (CORRIGIDO) Define ambos os eventos
        if not thread_stop_event.is_set():
            logging.warning("thread_stop_event não estava definido no finally, definindo agora.")
            thread_stop_event.set()
        if not mp_stop_event.is_set():
            logging.warning("mp_stop_event não estava definido no finally, definindo agora.")
            mp_stop_event.set()


        logging.info(">>> A iniciar encerramento final...")
        
        # (MODIFICADO) Termina o processo de CV primeiro
        if "dms_monitor" in locals() and dms_monitor:
            logging.info("A enviar sinal de paragem para o processo de CV...")
            dms_monitor.stop() # Isto chama mp_stop_event.set()

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
            
        for t in threads_to_join:
            logging.info(f"A aguardar thread '{t.name}'...")
            t.join(timeout=2.0)
            if t.is_alive():
                logging.warning(f"!!! Timeout ao esperar thread '{t.name}'.")

        # (MODIFICADO) Agora espera pelo 'join' do processo de CV
        if "cv_process" in locals() and cv_process and cv_process.is_alive():
            logging.info(f"A aguardar processo '{cv_process.name}'...")
            cv_process.join(timeout=5.0)
            if cv_process.is_alive():
                logging.warning(f"!!! Timeout ao esperar processo '{cv_process.name}'. A forçar terminação.")
                cv_process.terminate()

        logging.info(">>> Serviço DMS terminado.")