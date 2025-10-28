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
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('werkzeug') # Silencia logs HTTP do Flask (ou Waitress)
log.setLevel(logging.WARNING)
logging.info(f"Nível de log definido para: {default_log_level}")

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640
FRAME_HEIGHT_DISPLAY = 480
JPEG_QUALITY = 75 # Qualidade para o stream da UI
TARGET_FPS = 5 # Alvo de FPS para o RPi (5 FPS = 0.2s por frame)
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
EVENT_QUEUE_MAX_SIZE = 100 # (NOVO) Define o tamanho máximo da fila de eventos

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
    cv2.putText(frame, text, (30, FRAME_HEIGHT_DISPLAY // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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

    while not stop_event.is_set():
        start_time = time.time()

        # Verifica se a thread da câmara ainda está viva
        if not cam_thread.is_alive():
             logging.error("!!! Thread da câmara não está ativa. A parar loop de deteção.")
             break

        frame = cam_thread.get_frame()

        if frame is None:
            # Não loga como aviso se a thread estiver a parar
            if not stop_event.is_set():
                logging.debug("Frame não recebido da câmara. A aguardar...")
            time.sleep(0.1) # Pausa curta
            continue

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame, events, status_data = dms_monitor.process_frame(frame.copy(), gray)

            with output_frame_lock:
                output_frame_display = processed_frame.copy()

            with status_data_lock:
                status_data_global = status_data.copy()

            if events:
                for event in events:
                    try:
                        # Envia o evento E o frame ORIGINAL para ser guardado
                        event_queue_ref.put({"event_data": event, "frame": frame.copy()}, block=False, timeout=0.1)
                    except queue.Full:
                        logging.warning("!!! Fila de eventos cheia. Evento descartado.")
                    except Exception as q_err:
                         logging.error(f"Erro ao colocar evento na fila: {q_err}")


        except cv2.error as cv_err:
             logging.error(f"Erro OpenCV no process_frame: {cv_err}", exc_info=False) # Menos verboso
             time.sleep(1) # Pausa para evitar spam
        except Exception as e:
            logging.error(f"!!! Erro no process_frame: {e}", exc_info=True)
            time.sleep(1)

        # --- Controlo de FPS ---
        processing_time = time.time() - start_time
        wait_time = TARGET_FRAME_TIME - processing_time

        if wait_time > 0:
            # Usa sleep_event para permitir paragem rápida
            stop_event.wait(timeout=wait_time)
        else:
            current_time = time.time()
            if current_time - last_process_time > 5.0:
                 logging.warning(f"!!! LOOP LENTO. Processamento demorou {processing_time:.2f}s (Alvo era {TARGET_FRAME_TIME:.2f}s)")
                 last_process_time = current_time
            # Pequena pausa mesmo se lento para ceder CPU
            stop_event.wait(timeout=0.01)


    logging.info(">>> Loop de deteção terminado.")


# --- Servidor Web Flask ---

@app.route("/")
def index():
    """Rota principal que serve a página de calibração."""
    # Garante que cam_thread existe antes de aceder
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

    while not stop_event.is_set():
        frame_to_encode = None
        with output_frame_lock:
            if output_frame_display is not None:
                frame_to_encode = output_frame_display.copy()
            else:
                frame_to_encode = placeholder.copy()

        try:
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode,
                                                 [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not flag:
                logging.warning("Falha ao codificar frame para JPEG.")
                (flag, encodedImage) = cv2.imencode(".jpg", placeholder,
                                                     [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                if not flag: continue

            frame_bytes = bytearray(encodedImage)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')

        except GeneratorExit:
             logging.info("Client disconnected while serving /video_feed")
             break # Sai do loop quando o cliente desconecta
        except Exception as e:
             logging.error(f"Erro em generate_video_stream: {e}")
             break # Sai do loop em caso de erro

        # Controla o FPS do *stream* (20 FPS)
        target_stream_time = 1/20
        current_time = time.time()
        sleep_time = target_stream_time - (current_time - last_frame_time)
        if sleep_time > 0:
             stop_event.wait(timeout=sleep_time) # Permite paragem rápida
        last_frame_time = time.time()

@app.route("/video_feed")
def video_feed():
    """Rota que serve o stream de vídeo."""
    # Garante que o cam_thread está ativo
    if not cam_thread or not cam_thread.is_alive():
         return "Camera thread not running", 503
    return Response(generate_video_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Rotas da API ---

@app.route("/api/config", methods=['GET', 'POST'])
def api_config():
    """API para ler e atualizar as configurações de calibração."""
    # Garante que os monitores estão inicializados
    if not dms_monitor or not cam_thread or not event_queue:
         return jsonify({"error": "Service not fully initialized"}), 503

    if request.method == 'GET':
        current_settings = dms_monitor.get_settings()
        current_settings['brightness'] = cam_thread.get_brightness()
        current_settings['rotation'] = cam_thread.get_rotation()
        with status_data_lock:
            current_settings['status'] = status_data_global.copy()
        # (NOVO) Adiciona informações da fila
        try:
            current_settings['queue_depth'] = event_queue.qsize()
            current_settings['queue_max_size'] = event_queue.maxsize
        except Exception as e:
             logging.warning(f"Erro ao obter tamanho da fila: {e}")
             current_settings['queue_depth'] = -1 # Indica erro
             current_settings['queue_max_size'] = EVENT_QUEUE_MAX_SIZE

        return jsonify(current_settings)

    elif request.method == 'POST':
        new_settings = request.json
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
            return jsonify({"success": True})
        else:
            error_msg = "Failed to apply settings"
            if not dms_success: error_msg += " (DMS Core)"
            if not cam_success: error_msg += " (Camera)"
            return jsonify({"success": False, "error": error_msg}), 500

@app.route("/api/alerts", methods=['GET'])
def api_alerts():
    """API para obter a lista de alertas da base de dados SQLite."""
    # Garante que o event_handler está inicializado
    if not event_handler:
         return jsonify({"error": "Event handler not initialized"}), 503

    alerts_list = []
    try:
        # Usa o método get_alerts do EventHandler
        alerts_list = event_handler.get_alerts(limit=50) # Pega os últimos 50
        return jsonify(alerts_list)
    except Exception as e:
        logging.error(f"Erro ao ler alertas do SQLite: {e}", exc_info=True)
        return jsonify({"error": "Failed to read alerts from database"}), 500

@app.route('/alerts/images/<path:filepath>')
def serve_alert_image(filepath):
    """Rota para servir as imagens JPG dos alertas a partir das subpastas."""
    # Garante que o event_handler está inicializado
    if not event_handler:
         return "Event handler not initialized", 503

    # Segurança básica: Evita que filepath saia da pasta base de imagens
    image_base_path = os.path.join(event_handler.save_path, "images")
    safe_path = os.path.abspath(os.path.join(image_base_path, filepath))
    if not safe_path.startswith(image_base_path):
        logging.warning(f"Tentativa de acesso inválido: {filepath}")
        return "Invalid path", 400

    # Verifica se o ficheiro existe antes de tentar servir
    if not os.path.isfile(safe_path):
         logging.warning(f"Imagem não encontrada: {safe_path}")
         # Tenta criar um placeholder se a imagem não existir?
         # Por agora, retorna 404
         return "Image not found", 404

    try:
        # send_from_directory lida com os cabeçalhos corretos (MIME type)
        return send_from_directory(os.path.dirname(safe_path), os.path.basename(safe_path))
    except Exception as e:
        logging.error(f"Erro ao servir imagem de alerta '{filepath}': {e}")
        return "Internal server error", 500


# --- Encerramento Gracioso ---
def shutdown_handler(signum, frame):
    """Lida com sinais SIGINT/SIGTERM para parar as threads."""
    logging.info(f">>> Sinal {signal.Signals(signum).name} recebido. A iniciar encerramento...")
    stop_event.set() # Sinaliza a todas as threads para pararem

# --- Ponto de Entrada Principal ---
if __name__ == '__main__':

    # Regista os handlers para SIGINT (Ctrl+C) e SIGTERM (docker stop)
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        logging.info(f">>> Serviço DMS (com Bocejo + SQLite) a iniciar...")

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
        while cam_thread.get_frame() is None and cam_thread.is_alive():
            if stop_event.wait(timeout=0.5): # Verifica se deve parar
                 raise SystemExit("Encerramento solicitado durante inicialização da câmara.")
        if not cam_thread.is_alive():
            raise RuntimeError("!!! Thread da câmara terminou inesperadamente.")
        logging.info(">>> Primeiro frame recebido!")

        detection_thread = threading.Thread(target=detection_loop,
                                            args=(cam_thread, dms_monitor, event_queue))
        detection_thread.daemon = True # Permite sair mesmo se esta thread bloquear
        detection_thread.start()

        logging.info(f">>> A iniciar servidor Flask na porta 5000...")
        if HAS_WAITRESS:
            logging.info("A usar servidor Waitress.")
            # Nota: Waitress bloqueia aqui até ser interrompido
            serve(app, host='0.0.0.0', port=5000, threads=8)
        else:
            logging.warning("Pacote 'waitress' não encontrado. Adicione 'waitress' ao requirements.txt para um servidor de produção.")
            logging.warning("A usar o servidor de desenvolvimento Flask (NÃO RECOMENDADO PARA PRODUÇÃO).")
            # O servidor Flask respeita o SIGINT/SIGTERM por padrão
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

    except (KeyboardInterrupt, SystemExit) as e:
         logging.info(f">>> {type(e).__name__} recebido. A encerrar...")
    except Exception as e:
        logging.error(f"!!! ERRO FATAL ao iniciar/executar o serviço: {e}", exc_info=True)
    finally:
        logging.info(">>> A iniciar encerramento final do serviço...")
        stop_event.set() # Garante que está definido

        # Espera pelas threads terminarem (com timeouts)
        if cam_thread and cam_thread.is_alive():
            logging.info("A aguardar thread da câmara...")
            cam_thread.join(timeout=3)
            if cam_thread.is_alive():
                 logging.warning("!!! Timeout ao esperar pela thread da câmara.")

        # A thread de deteção é daemon, mas esperamos um pouco na mesma
        if detection_thread and detection_thread.is_alive():
             logging.info("A aguardar thread de deteção...")
             detection_thread.join(timeout=2)
             if detection_thread.is_alive():
                  logging.warning("!!! Timeout ao esperar pela thread de deteção (pode bloquear se a câmara falhar).")

        if event_handler and event_handler.is_alive():
            logging.info("A aguardar thread do gestor de eventos...")
            event_handler.join(timeout=5) # Dá mais tempo para guardar eventos
            if event_handler.is_alive():
                 logging.warning("!!! Timeout ao esperar pelo gestor de eventos.")

        logging.info(">>> Serviço DMS terminado.")
        sys.exit(0) # Garante que o container termina

    

