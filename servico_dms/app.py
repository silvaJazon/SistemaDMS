# Documentação: Aplicação Principal Flask para o SistemaDMS
# (VERSÃO: MediaPipe + YOLO)

import cv2
import time
import os
import numpy as np
import threading
import logging
import sys
from flask import Flask, Response, render_template, jsonify, request, send_from_directory
import queue
import json
from datetime import datetime
import signal

# Importa os nossos módulos
from camera_thread import CameraThread
# (ALTERADO) Importa a base e APENAS o MediaPipeMonitor
from dms_base import BaseMonitor
from dms_mediapipe import MediaPipeMonitor
# (REMOVIDO) DlibMonitor
    
from event_handler import EventHandler

# Tenta importar o Waitress
try:
    from waitress import serve
    HAS_WAITRESS = True
except ImportError:
    HAS_WAITRESS = False

cv2.setUseOptimized(True)

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_level = logging.DEBUG if default_log_level == 'DEBUG' else logging.INFO
logging.basicConfig(level=log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640
FRAME_HEIGHT_DISPLAY = 480
JPEG_QUALITY = 75
TARGET_FPS = 5
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
EVENT_QUEUE_MAX_SIZE = 100
INITIAL_ROTATION = int(os.environ.get('ROTATE_FRAME', '0'))

# (ALTERADO) Backend é fixo
DETECTION_BACKEND = "MEDIAPIPE"

# Configurações Padrão de Deteção (centralizadas)
DEFAULT_EAR_THRESHOLD = 0.25
DEFAULT_EAR_FRAMES = 7
DEFAULT_MAR_THRESHOLD = 0.40
DEFAULT_MAR_FRAMES = 10

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
dms_monitor: BaseMonitor = None # Tipo é a classe Base

app = Flask(__name__)

# --- Funções Auxiliares ---
def create_placeholder_frame(text="Aguardando camera..."):
    """Cria um frame preto com texto."""
    frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        cv2.putText(frame, text, (30, FRAME_HEIGHT_DISPLAY // 2), font, 1, (255, 255, 255), 2)
    except cv2.error as e:
         logging.warning(f"Erro ao desenhar texto no placeholder: {e}")
         cv2.rectangle(frame, (10, FRAME_HEIGHT_DISPLAY//2 - 20), (FRAME_WIDTH_DISPLAY-10, FRAME_HEIGHT_DISPLAY//2 + 20), (50, 50, 50), -1)
    return frame

# --- Threads Principais ---
def detection_loop(cam_thread_ref, dms_monitor_ref: BaseMonitor, event_queue_ref):
    """Loop principal de deteção (Usa a interface BaseMonitor)."""
    global output_frame_display, status_data_global
    logging.info(f">>> Loop de deteção (Backend: {DETECTION_BACKEND}) iniciado (Alvo: {TARGET_FPS} FPS).")
    last_process_time = time.time()
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
            if not stop_event.is_set(): logging.debug("Frame não recebido.")
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

            processed_frame, events, status_data = dms_monitor_ref.process_frame(frame.copy(), gray)
            logging.debug("DetectionLoop: process_frame() retornou.")

            logging.debug("DetectionLoop: A adquirir output_frame_lock...")
            with output_frame_lock:
                logging.debug("DetectionLoop: output_frame_lock adquirido.")
                output_frame_display = processed_frame.copy()
            logging.debug("DetectionLoop: output_frame_lock libertado.")

            frame_count += 1
            if frame_count % 100 == 0: logging.debug(f"Loop deteção: Frame {frame_count}.")

            logging.debug("DetectionLoop: A adquirir status_data_lock...")
            with status_data_lock:
                logging.debug("DetectionLoop: status_data_lock adquirido.")
                status_data_global = status_data.copy()
            logging.debug("DetectionLoop: status_data_lock libertado.")

            if events:
                logging.debug(f"DetectionLoop: A processar {len(events)} eventos...")
                for event in events:
                    try:
                        event_queue_ref.put({"event_data": event, "frame": frame.copy()}, block=False, timeout=0.1)
                    except queue.Full: logging.warning("!!! Fila cheia.")
                    except Exception as q_err: logging.error(f"Erro fila: {q_err}")

        except cv2.error as cv_err:
             logging.error(f"Erro OpenCV (shape: {frame.shape if frame is not None else 'None'}): {cv_err}", exc_info=True)
             stop_event.wait(timeout=1.0)
        except Exception as e:
            logging.error(f"!!! Erro no process_frame: {e}", exc_info=True)
            stop_event.wait(timeout=1.0)

        processing_time = time.time() - start_time
        wait_time = TARGET_FRAME_TIME - processing_time
        logging.debug(f"Tempo: {processing_time:.3f}s, Espera: {max(0, wait_time):.3f}s")

        if wait_time > 0:
            logging.debug(f"DetectionLoop: A esperar {wait_time:.3f}s...")
            stop_event.wait(timeout=wait_time)
        else:
            current_time = time.time()
            if current_time - last_process_time > 5.0:
                 logging.warning(f"!!! LOOP LENTO. Demorou {processing_time:.2f}s (Alvo {TARGET_FRAME_TIME:.2f}s)")
                 last_process_time = current_time
            logging.debug("DetectionLoop: Loop lento, pausa (0.01s).")
            stop_event.wait(timeout=0.01)

    logging.info(">>> Loop de deteção terminado.")


# --- Servidor Web Flask ---
@app.route("/")
def index():
    """Serve a página de calibração."""
    cam_source_desc = cam_thread.source_description if cam_thread else "Indisponível"
    # (ALTERADO) Passa o backend (fixo) para o template
    return render_template("index.html", source_desc=cam_source_desc,
                           width=FRAME_WIDTH_DISPLAY, height=FRAME_HEIGHT_DISPLAY,
                           active_backend=DETECTION_BACKEND)

@app.route("/alerts")
def alerts_page():
    """Serve a página de histórico."""
    return render_template("alerts.html")

def generate_video_stream():
    """Gera frames de vídeo para o stream HTTP."""
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
             frame_to_encode = placeholder.copy(); use_placeholder = True
        try:
            if not isinstance(frame_to_encode, np.ndarray) or frame_to_encode.size == 0:
                 logging.error(f"generate_video_stream: Frame inválido (tipo: {type(frame_to_encode)}). Usando placeholder.")
                 frame_to_encode = placeholder.copy(); use_placeholder = True

            logging.debug("generate_video_stream: A codificar frame...")
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, encode_param)
            logging.debug(f"generate_video_stream: Codificação {'bem-sucedida' if flag else 'falhou'}.")

            if not flag:
                logging.warning(f"generate_video_stream: Falha codificar (ph={use_placeholder}). Tentando placeholder.")
                (flag, encodedImage) = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                if not flag:
                     logging.error("generate_video_stream: Falha codificar placeholder. Saltando frame.")
                     stop_event.wait(timeout=0.1); continue
            frame_bytes = bytearray(encodedImage)
            logging.debug(f"generate_video_stream: A enviar frame {frame_yield_count} ({len(frame_bytes)} bytes).")
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            frame_yield_count += 1
        except GeneratorExit: logging.info("generate_video_stream: Cliente desconectou."); break
        except cv2.error as e: logging.error(f"generate_video_stream: Erro OpenCV codificar (ph={use_placeholder}, shape={frame_to_encode.shape}): {e}", exc_info=True); stop_event.wait(timeout=0.5)
        except Exception as e: logging.error(f"generate_video_stream: Erro inesperado: {e}", exc_info=True); break

        target_stream_time = 1/20
        current_time = time.time()
        sleep_time = target_stream_time - (current_time - last_frame_time)
        if sleep_time > 0: stop_event.wait(timeout=sleep_time)
        last_frame_time = time.time()

    logging.info(f"generate_video_stream: Terminado após {frame_yield_count} frames.")

@app.route("/video_feed")
def video_feed():
    """Serve o stream de vídeo."""
    logging.debug("Rota /video_feed acedida.")
    if not cam_thread or not cam_thread.is_alive():
         logging.error("Rota /video_feed: Thread câmara não ativa.")
         return "Camera thread not running", 503
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Rotas da API ---
@app.route("/api/config", methods=['GET', 'POST'])
def api_config():
    """API para ler/atualizar configurações."""
    global dms_monitor
    logging.debug(f"Rota /api/config (Método: {request.method})")
    if dms_monitor is None or not cam_thread or not event_queue:
         logging.warning("/api/config: Serviço não inicializado.")
         return jsonify({"error": "Service not fully initialized"}), 503

    if request.method == 'GET':
        try:
            current_settings = dms_monitor.get_settings()
            current_settings['brightness'] = cam_thread.get_brightness()
            current_settings['rotation'] = cam_thread.get_rotation()
            
            # Adiciona o backend ativo
            current_settings['active_backend'] = DETECTION_BACKEND

            logging.debug("api_config GET: Lock status...")
            with status_data_lock:
                logging.debug("api_config GET: Lock status OK.")
                current_settings['status'] = status_data_global.copy()
            logging.debug("api_config GET: Lock status libertado.")

            try: # Fila
                current_settings['queue_depth'] = event_queue.qsize()
                current_settings['queue_max_size'] = event_queue.maxsize
            except Exception as e:
                 logging.warning(f"Erro obter tamanho fila: {e}")
                 current_settings['queue_depth'] = -1; current_settings['queue_max_size'] = EVENT_QUEUE_MAX_SIZE

            logging.debug(f"/api/config GET: Retornando {current_settings}")
            return jsonify(current_settings)
        except Exception as e:
             logging.error(f"Erro inesperado /api/config GET: {e}", exc_info=True)
             return jsonify({"error": "Internal server error reading config"}), 500

    elif request.method == 'POST':
        try:
            new_settings = request.json
            logging.debug(f"/api/config POST: Recebido {new_settings}")
            if not new_settings: return jsonify({"success": False, "error": "No data received"}), 400

            dms_success = dms_monitor.update_settings(new_settings)

            cam_success = True # Câmara
            try:
                if 'brightness' in new_settings: cam_thread.update_brightness(new_settings['brightness'])
                if 'rotation' in new_settings: cam_thread.update_rotation(new_settings['rotation'])
            except Exception as e: logging.error(f"Erro atualizar conf câmara: {e}"); cam_success = False

            if dms_success and cam_success:
                logging.info(f"/api/config POST: Configurações atualizadas.")
                return jsonify({"success": True})
            else:
                error_msg = "Failed settings"+(" (DMS)" if not dms_success else "")+(" (Cam)" if not cam_success else "")
                logging.warning(f"/api/config POST: Falha: {error_msg}")
                return jsonify({"success": False, "error": error_msg}), 500
        except Exception as e:
             logging.error(f"Erro inesperado /api/config POST: {e}", exc_info=True)
             return jsonify({"error": "Internal server error updating config"}), 500

@app.route("/api/alerts", methods=['GET'])
def api_alerts():
    """API para obter alertas."""
    logging.debug("Rota /api/alerts acedida.")
    if not event_handler: logging.warning("/api/alerts: Gestor eventos não init."); return jsonify({"error": "Event handler not initialized"}), 503
    try:
        alerts_list = event_handler.get_alerts(limit=50)
        logging.debug(f"/api/alerts: Retornando {len(alerts_list)} alertas.")
        return jsonify(alerts_list)
    except Exception as e: logging.error(f"Erro ler alertas SQLite: {e}", exc_info=True); return jsonify({"error": "Failed to read alerts from database"}), 500

@app.route('/alerts/images/<path:filepath>')
def serve_alert_image(filepath):
    """Serve imagens de alerta."""
    logging.debug(f"Rota /alerts/images: {filepath}")
    if not event_handler: logging.warning(f"/alerts/images: Gestor eventos não init ({filepath})."); return "Event handler not initialized", 503
    image_base_path = os.path.join(event_handler.save_path, "images"); safe_path = os.path.abspath(os.path.join(image_base_path, filepath))
    if not safe_path.startswith(image_base_path): logging.warning(f"Acesso inválido /alerts/images: {filepath}"); return "Invalid path", 400
    if not os.path.isfile(safe_path): logging.warning(f"Imagem não encontrada /alerts/images: {safe_path}"); return "Image not found", 404
    try: logging.debug(f"A servir imagem: {safe_path}"); return send_from_directory(os.path.dirname(safe_path), os.path.basename(safe_path))
    except Exception as e: logging.error(f"Erro servir imagem '{filepath}': {e}", exc_info=True); return "Internal server error", 500

# --- Encerramento Gracioso ---
def shutdown_handler(signum, frame):
    """Lida com SIGINT/SIGTERM."""
    if not stop_event.is_set():
        logging.info(f">>> Sinal {signal.Signals(signum).name} recebido. A encerrar...")
        stop_event.set()


# --- Ponto de Entrada Principal ---
if __name__ == '__main__':
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
        
        # Cria dicionário de padrões para passar ao monitor
        default_dms_settings = {
            "ear_threshold": DEFAULT_EAR_THRESHOLD,
            "ear_frames": DEFAULT_EAR_FRAMES,
            "mar_threshold": DEFAULT_MAR_THRESHOLD,
            "mar_frames": DEFAULT_MAR_FRAMES
        }
        
        # ================== ALTERAÇÃO (Carregamento Fixo) ==================
        # Carrega diretamente o MediaPipeMonitor
        if not HAS_MEDIAPIPE:
             logging.error("!!! MediaPipe não foi importado com sucesso. A parar.")
             raise ImportError("MediaPipeMonitor não encontrado.")
             
        dms_monitor = MediaPipeMonitor(frame_size=frame_size,
                                       default_settings=default_dms_settings)
        # ===================================================================

        cam_thread = CameraThread(
            VIDEO_SOURCE,
            frame_width=FRAME_WIDTH_DISPLAY,
            frame_height=FRAME_HEIGHT_DISPLAY,
            rotation_degrees=INITIAL_ROTATION,
            stop_event=stop_event
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

        detection_thread = threading.Thread(
            target=detection_loop,
            args=(cam_thread, dms_monitor, event_queue),
            name="DetectionThread"
        )
        detection_thread.daemon = True
        detection_thread.start()

        logging.info(">>> A iniciar servidor web porta 5000...")

        if HAS_WAITRESS:
            logging.info("A usar Waitress.")
            serve(app, host='0.0.0.0', port=5000, threads=8)
        else:
            logging.warning("Waitress não encontrado.")
            logging.warning("A usar Flask dev server.")
            try:
                app.run(
                    host='0.0.0.0',
                    port=5000,
                    debug=False,
                    threaded=True,
                    use_reloader=False
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
        # Garante que stop_event é definido antes de tentar juntar threads
        if not stop_event.is_set():
            logging.warning("stop_event não estava definido no finally, definindo agora.")
            stop_event.set()

        logging.info(">>> A iniciar encerramento final...")
        threads_to_join = []

        # Verifica se as variáveis existem antes de aceder
        if 'detection_thread' in locals() and detection_thread and detection_thread.is_alive():
            threads_to_join.append(detection_thread)
        if 'cam_thread' in locals() and cam_thread and cam_thread.is_alive():
            threads_to_join.append(cam_thread)
        if 'event_handler' in locals() and event_handler and event_handler.is_alive():
            threads_to_join.append(event_handler)

        for t in threads_to_join:
            logging.info(f"A aguardar thread '{t.name}'...")
            timeout = 2 if getattr(t, 'daemon', False) else 5  # getattr para segurança
            t.join(timeout=timeout)

            # Verifica se alguma thread ficou presa
            if t.is_alive():
                logging.warning(f"!!! Timeout ao esperar thread '{t.name}'.")

        logging.info(">>> Serviço DMS terminado.")
        os._exit(0)  # Força a saída