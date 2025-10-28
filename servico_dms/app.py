# Documentação: Aplicação Principal Flask para o SistemaDMS
# Orquestra a captura de vídeo, processamento de IA e a interface web.
# (Atualizado com SQLite, Deteção de Bocejo, Waitress, Status da Fila)

import cv2
import time
import os
import numpy as np
import threading
import logging
import sys
from flask import Flask, Response, render_template, jsonify, request, send_from_directory
import queue # Renomeado para evitar conflito com a variável 'queue'
import json
from datetime import datetime
import signal # Para lidar com o encerramento gracioso

# Importa os nossos módulos
from camera_thread import CameraThread
from dms_core import DriverMonitor
from event_handler import EventHandler # A nossa "Central" SQLite

# Tenta importar o Waitress para produção
try:
    from waitress import serve
    HAS_WAITRESS = True
except ImportError:
    HAS_WAITRESS = False

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
# (NOVO) Define o nível DEBUG se LOG_LEVEL=DEBUG
log_level = logging.DEBUG if default_log_level == 'DEBUG' else logging.INFO
logging.basicConfig(level=log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('werkzeug') # Silencia logs HTTP do Flask (ou Waitress)
log.setLevel(logging.WARNING)
# (Removido log duplicado) logging.info(f"Nível de log definido para: {default_log_level}")

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640
FRAME_HEIGHT_DISPLAY = 480
JPEG_QUALITY = 75 # Qualidade para o stream da UI
TARGET_FPS = 5 # Alvo de FPS para o RPi (5 FPS = 0.2s por frame)
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
EVENT_QUEUE_MAX_SIZE = 100 # Define o tamanho máximo da fila de eventos

# Rotação inicial (lida do ambiente, 0 por padrão)
INITIAL_ROTATION = int(os.environ.get('ROTATE_FRAME', '0'))

# --- Variáveis Globais ---
output_frame_display = None # O último frame processado para o stream
output_frame_lock = threading.Lock() # Protege o acesso ao output_frame_display
status_data_global = {} # Últimos dados de status (EAR, Yaw, MAR) para a API
status_data_lock = threading.Lock()
stop_event = threading.Event() # Sinal para parar as threads

# Declara as variáveis globais para as threads
cam_thread = None
detection_thread = None
event_handler = None
event_queue = None # Será inicializada mais tarde

app = Flask(__name__)

# --- Funções Auxiliares ---

def create_placeholder_frame(text="Aguardando camera..."):
    """Cria um frame preto com texto para usar quando a câmara não está disponível."""
    frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
    # (NOVO) Garante que a fonte existe antes de usar
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        cv2.putText(frame, text, (30, FRAME_HEIGHT_DISPLAY // 2),
                    font, 1, (255, 255, 255), 2)
    except cv2.error as e:
         logging.warning(f"Erro ao desenhar texto no placeholder (fonte pode faltar): {e}")
         # Tenta desenhar um retângulo simples como fallback
         cv2.rectangle(frame, (10, FRAME_HEIGHT_DISPLAY//2 - 20), (FRAME_WIDTH_DISPLAY-10, FRAME_HEIGHT_DISPLAY//2 + 20), (50, 50, 50), -1)

    return frame

# --- Threads Principais ---

# 1. Thread de Deteção (Processamento de IA)
def detection_loop(cam_thread, dms_monitor, event_queue_ref):
    """
    Loop principal executado em background para obter frames da câmara,
    processá-los com o dms_monitor e atualizar o output_frame.
    """
    global output_frame_display, status_data_global

    logging.info(f">>> Loop de deteção iniciado (Alvo: {TARGET_FPS} FPS).")
    last_process_time = time.time()
    frame_count = 0 # (NOVO) Contador de frames processados

    while not stop_event.is_set():
        start_time = time.time()

        if not cam_thread.is_alive():
             logging.error("!!! Thread da câmara não está ativa. A parar loop de deteção.")
             break

        frame = cam_thread.get_frame()

        if frame is None:
            if not stop_event.is_set():
                logging.debug("Frame não recebido da câmara. A aguardar...")
            time.sleep(0.1)
            continue

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame, events, status_data = dms_monitor.process_frame(frame.copy(), gray)

            with output_frame_lock:
                output_frame_display = processed_frame.copy()
            # (NOVO) Log de debug a cada 100 frames
            frame_count += 1
            if frame_count % 100 == 0:
                 logging.debug(f"Loop de deteção: Frame {frame_count} processado e atualizado.")


            with status_data_lock:
                status_data_global = status_data.copy()

            if events:
                for event in events:
                    try:
                        event_queue_ref.put({"event_data": event, "frame": frame.copy()}, block=False, timeout=0.1)
                    except queue.Full:
                        logging.warning("!!! Fila de eventos cheia. Evento descartado.")
                    except Exception as q_err:
                         logging.error(f"Erro ao colocar evento na fila: {q_err}")


        except cv2.error as cv_err:
             # (NOVO) Log mais detalhado para erros OpenCV
             logging.error(f"Erro OpenCV no process_frame (frame shape: {frame.shape if frame is not None else 'None'}): {cv_err}", exc_info=True)
             time.sleep(1)
        except Exception as e:
            logging.error(f"!!! Erro no process_frame: {e}", exc_info=True)
            time.sleep(1)

        # --- Controlo de FPS ---
        processing_time = time.time() - start_time
        wait_time = TARGET_FRAME_TIME - processing_time
        logging.debug(f"Tempo de processamento: {processing_time:.3f}s, Espera: {max(0, wait_time):.3f}s") # (NOVO) Debug FPS


        if wait_time > 0:
            stop_event.wait(timeout=wait_time)
        else:
            current_time = time.time()
            if current_time - last_process_time > 5.0:
                 logging.warning(f"!!! LOOP LENTO. Processamento demorou {processing_time:.2f}s (Alvo era {TARGET_FRAME_TIME:.2f}s)")
                 last_process_time = current_time
            stop_event.wait(timeout=0.01)


    logging.info(">>> Loop de deteção terminado.")


# --- Servidor Web Flask ---

@app.route("/")
def index():
    """Rota principal que serve a página de calibração."""
    cam_source_desc = cam_thread.source_description if cam_thread else "Indisponível"
    return render_template("index.html",
                           source_desc=cam_source_desc,
                           width=FRAME_WIDTH_DISPLAY,
                           height=FRAME_HEIGHT_DISPLAY)

@app.route("/alerts")
def alerts_page():
    """Rota que serve a página de histórico de alertas."""
    return render_template("alerts.html")


def generate_video_stream():
    """Gera frames de vídeo para o stream HTTP (usado por /video_feed)."""
    global output_frame_display
    placeholder = create_placeholder_frame()
    last_frame_time = time.time()
    frame_yield_count = 0 # (NOVO) Contador de frames enviados

    logging.debug("generate_video_stream: Iniciando gerador de stream.") # (NOVO)

    while not stop_event.is_set():
        frame_to_encode = None
        # (NOVO) Verifica se há um frame novo ou usa placeholder
        use_placeholder = False
        with output_frame_lock:
            if output_frame_display is not None:
                frame_to_encode = output_frame_display.copy()
                logging.debug("generate_video_stream: Usando frame processado.") # (NOVO)
            else:
                frame_to_encode = placeholder.copy()
                use_placeholder = True
                logging.debug("generate_video_stream: Usando placeholder.") # (NOVO)

        # Garante que frame_to_encode não é None (segurança extra)
        if frame_to_encode is None:
             logging.warning("generate_video_stream: frame_to_encode é None, usando placeholder.")
             frame_to_encode = placeholder.copy()
             use_placeholder = True

        try:
            # (NOVO) Verifica se o frame é válido antes de codificar
            if not isinstance(frame_to_encode, np.ndarray) or frame_to_encode.size == 0:
                 logging.error(f"generate_video_stream: Frame inválido recebido (tipo: {type(frame_to_encode)}). Usando placeholder.")
                 frame_to_encode = placeholder.copy()
                 use_placeholder = True

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, encode_param)

            if not flag:
                logging.warning(f"generate_video_stream: Falha ao codificar frame (placeholder={use_placeholder}). Tentando placeholder.")
                (flag, encodedImage) = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                if not flag:
                     logging.error("generate_video_stream: Falha ao codificar até o placeholder. A saltar frame.")
                     stop_event.wait(timeout=0.1) # Pausa curta antes de tentar de novo
                     continue # Pula para a próxima iteração

            frame_bytes = bytearray(encodedImage)
            # (NOVO) Log antes de enviar
            logging.debug(f"generate_video_stream: A enviar frame {frame_yield_count} ({len(frame_bytes)} bytes).")
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')
            frame_yield_count += 1

        except GeneratorExit:
             logging.info("generate_video_stream: Cliente desconectou.")
             break
        except cv2.error as cv_err:
             # (NOVO) Log específico para erros OpenCV na codificação
             logging.error(f"generate_video_stream: Erro OpenCV ao codificar frame (placeholder={use_placeholder}, shape={frame_to_encode.shape}): {cv_err}", exc_info=True)
             stop_event.wait(timeout=0.5) # Pausa antes de tentar de novo
        except Exception as e:
             logging.error(f"generate_video_stream: Erro inesperado: {e}", exc_info=True)
             break

        # Controla o FPS do *stream* (20 FPS)
        target_stream_time = 1/20
        current_time = time.time()
        sleep_time = target_stream_time - (current_time - last_frame_time)
        if sleep_time > 0:
             stop_event.wait(timeout=sleep_time)
        last_frame_time = time.time()

    logging.info(f"generate_video_stream: Gerador de stream terminado após enviar {frame_yield_count} frames.") # (NOVO)

@app.route("/video_feed")
def video_feed():
    """Rota que serve o stream de vídeo."""
    logging.debug("Rota /video_feed acedida.") # (NOVO)
    if not cam_thread or not cam_thread.is_alive():
         logging.error("Rota /video_feed: Thread da câmara não está ativa.") # (NOVO)
         return "Camera thread not running", 503
    return Response(generate_video_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Rotas da API ---

@app.route("/api/config", methods=['GET', 'POST'])
def api_config():
    """API para ler e atualizar as configurações de calibração."""
    logging.debug(f"Rota /api/config acedida (Método: {request.method})") # (NOVO)
    if not dms_monitor or not cam_thread or not event_queue:
         logging.warning("/api/config: Serviço não totalmente inicializado.") # (NOVO)
         return jsonify({"error": "Service not fully initialized"}), 503

    if request.method == 'GET':
        try: # (NOVO) Bloco try/except para GET
            current_settings = dms_monitor.get_settings()
            current_settings['brightness'] = cam_thread.get_brightness()
            current_settings['rotation'] = cam_thread.get_rotation()
            with status_data_lock:
                current_settings['status'] = status_data_global.copy()
            try:
                current_settings['queue_depth'] = event_queue.qsize()
                current_settings['queue_max_size'] = event_queue.maxsize
            except Exception as e:
                 logging.warning(f"Erro ao obter tamanho da fila: {e}")
                 current_settings['queue_depth'] = -1
                 current_settings['queue_max_size'] = EVENT_QUEUE_MAX_SIZE

            logging.debug(f"/api/config GET: Retornando {current_settings}") # (NOVO)
            return jsonify(current_settings)
        except Exception as e:
             logging.error(f"Erro inesperado em /api/config GET: {e}", exc_info=True)
             return jsonify({"error": "Internal server error reading config"}), 500


    elif request.method == 'POST':
        try: # (NOVO) Bloco try/except para POST
            new_settings = request.json
            logging.debug(f"/api/config POST: Recebido {new_settings}") # (NOVO)
            if not new_settings:
                return jsonify({"success": False, "error": "No data received"}), 400

            dms_success = dms_monitor.update_settings(new_settings)

            cam_success = True
            try:
                if 'brightness' in new_settings:
                    cam_thread.update_brightness(new_settings['brightness'])
                if 'rotation' in new_settings:
                    cam_thread.update_rotation(new_settings['rotation'])
            except Exception as e:
                 logging.error(f"Erro ao atualizar configurações da câmara: {e}")
                 cam_success = False

            if dms_success and cam_success:
                logging.info(f"/api/config POST: Configurações atualizadas com sucesso.") # (NOVO)
                return jsonify({"success": True})
            else:
                error_msg = "Failed to apply settings"
                if not dms_success: error_msg += " (DMS Core)"
                if not cam_success: error_msg += " (Camera)"
                logging.warning(f"/api/config POST: Falha ao aplicar configurações: {error_msg}") # (NOVO)
                return jsonify({"success": False, "error": error_msg}), 500
        except Exception as e:
             logging.error(f"Erro inesperado em /api/config POST: {e}", exc_info=True)
             return jsonify({"error": "Internal server error updating config"}), 500


@app.route("/api/alerts", methods=['GET'])
def api_alerts():
    """API para obter a lista de alertas da base de dados SQLite."""
    logging.debug("Rota /api/alerts acedida.") # (NOVO)
    if not event_handler:
         logging.warning("/api/alerts: Gestor de eventos não inicializado.") # (NOVO)
         return jsonify({"error": "Event handler not initialized"}), 503

    alerts_list = []
    try:
        alerts_list = event_handler.get_alerts(limit=50)
        logging.debug(f"/api/alerts: Retornando {len(alerts_list)} alertas.") # (NOVO)
        return jsonify(alerts_list)
    except Exception as e:
        logging.error(f"Erro ao ler alertas do SQLite: {e}", exc_info=True)
        return jsonify({"error": "Failed to read alerts from database"}), 500

@app.route('/alerts/images/<path:filepath>')
def serve_alert_image(filepath):
    """Rota para servir as imagens JPG dos alertas a partir das subpastas."""
    logging.debug(f"Rota /alerts/images acedida para: {filepath}") # (NOVO)
    if not event_handler:
         logging.warning(f"/alerts/images: Gestor de eventos não inicializado ao tentar servir {filepath}.") # (NOVO)
         return "Event handler not initialized", 503

    image_base_path = os.path.join(event_handler.save_path, "images")
    safe_path = os.path.abspath(os.path.join(image_base_path, filepath))
    if not safe_path.startswith(image_base_path):
        logging.warning(f"Tentativa de acesso inválido via /alerts/images: {filepath}")
        return "Invalid path", 400

    if not os.path.isfile(safe_path):
         logging.warning(f"Imagem não encontrada via /alerts/images: {safe_path}")
         return "Image not found", 404

    try:
        logging.debug(f"A servir imagem: {safe_path}") # (NOVO)
        return send_from_directory(os.path.dirname(safe_path), os.path.basename(safe_path))
    except Exception as e:
        logging.error(f"Erro ao servir imagem de alerta '{filepath}': {e}", exc_info=True) # (NOVO) Usar exc_info
        return "Internal server error", 500


# --- Encerramento Gracioso ---
def shutdown_handler(signum, frame):
    """Lida com sinais SIGINT/SIGTERM para parar as threads."""
    if not stop_event.is_set(): # Evita logs duplicados se o sinal for recebido várias vezes
        logging.info(f">>> Sinal {signal.Signals(signum).name} recebido. A iniciar encerramento...")
        stop_event.set()

# --- Ponto de Entrada Principal ---
if __name__ == '__main__':

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        # (NOVO) Log mais claro do nível de log
        logging.info(f">>> Serviço DMS (com Bocejo + SQLite) a iniciar... (Nível de Log: {logging.getLevelName(logging.getLogger().level)})")

        event_queue = queue.Queue(maxsize=EVENT_QUEUE_MAX_SIZE)

        event_handler = EventHandler(queue=event_queue, stop_event=stop_event)
        event_handler.start()

        frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)
        dms_monitor = DriverMonitor(frame_size=frame_size)

        cam_thread = CameraThread(VIDEO_SOURCE,
                                  frame_width=FRAME_WIDTH_DISPLAY,
                                  frame_height=FRAME_HEIGHT_DISPLAY,
                                  rotation_degrees=INITIAL_ROTATION)
        cam_thread.start()

        logging.info("A aguardar o primeiro frame da câmara...")
        # (NOVO) Timeout mais longo e verificação de stop_event
        start_wait_cam = time.time()
        while cam_thread.get_frame() is None and cam_thread.is_alive():
             if stop_event.wait(timeout=0.2):
                 raise SystemExit("Encerramento solicitado durante inicialização da câmara.")
             if time.time() - start_wait_cam > 15: # Timeout de 15 segundos
                  raise RuntimeError("!!! Timeout à espera do primeiro frame da câmara.")
        if not cam_thread.is_alive():
            raise RuntimeError("!!! Thread da câmara terminou inesperadamente durante a espera pelo primeiro frame.")
        logging.info(">>> Primeiro frame recebido!")

        detection_thread = threading.Thread(target=detection_loop,
                                            args=(cam_thread, dms_monitor, event_queue),
                                            name="DetectionThread") # (NOVO) Nome da thread
        detection_thread.daemon = True
        detection_thread.start()

        logging.info(f">>> A iniciar servidor web na porta 5000...")
        if HAS_WAITRESS:
            logging.info("A usar servidor Waitress.")
            serve(app, host='0.0.0.0', port=5000, threads=8) # Waitress bloqueia aqui
        else:
            logging.warning("Pacote 'waitress' não encontrado. Adicione 'waitress' ao requirements.txt para um servidor de produção.")
            logging.warning("A usar o servidor de desenvolvimento Flask (NÃO RECOMENDADO PARA PRODUÇÃO).")
            # Adiciona try/except para lidar com erros ao iniciar o servidor Flask
            try:
                app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False) # Flask bloqueia aqui
            except OSError as e:
                 if "Address already in use" in str(e):
                      logging.error("!!! ERRO FATAL: A porta 5000 já está em uso. Outro processo pode estar a usá-la.")
                 else:
                      logging.error(f"!!! ERRO FATAL ao iniciar o servidor Flask: {e}", exc_info=True)
                 stop_event.set() # Sinaliza às outras threads para pararem
            except Exception as e:
                 logging.error(f"!!! ERRO FATAL no servidor Flask: {e}", exc_info=True)
                 stop_event.set()

    except (KeyboardInterrupt, SystemExit) as e:
         logging.info(f">>> {type(e).__name__} recebido ou SystemExit chamado. A encerrar...")
         if not stop_event.is_set():
              stop_event.set() # Garante que o evento de paragem é acionado
    except RuntimeError as e: # Captura o timeout da câmara
         logging.error(f"!!! ERRO FATAL durante a inicialização: {e}")
         if not stop_event.is_set():
              stop_event.set()
    except Exception as e:
        logging.error(f"!!! ERRO FATAL não capturado no bloco principal: {e}", exc_info=True)
        if not stop_event.is_set():
             stop_event.set()
    finally:
        logging.info(">>> A iniciar encerramento final do serviço...")
        # A ordem de join pode ser importante
        threads_to_join = []
        if detection_thread and detection_thread.is_alive(): threads_to_join.append(detection_thread)
        if cam_thread and cam_thread.is_alive(): threads_to_join.append(cam_thread)
        if event_handler and event_handler.is_alive(): threads_to_join.append(event_handler)

        for t in threads_to_join:
             logging.info(f"A aguardar thread '{t.name}'...")
             # Timeout mais curto para deteção (daemon)
             timeout = 2 if t == detection_thread else 5
             t.join(timeout=timeout)
             if t.is_alive():
                  logging.warning(f"!!! Timeout ao esperar pela thread '{t.name}'.")

        logging.info(">>> Serviço DMS terminado.")
        # Usa os._exit para forçar a saída se alguma thread não terminou
        os._exit(0)

