import cv2
import time
import os
import numpy as np
import threading
import logging
import sys
from flask import Flask, Response, render_template, jsonify, request, send_from_directory, abort, send_file
from queue import Queue
# import json # Não precisamos mais disto para os alertas
from datetime import datetime
import sqlite3 # (NOVO) Importa a biblioteca SQLite

# Importa os nossos módulos
from camera_thread import CameraThread
from dms_core import DriverMonitor
from event_handler import EventHandler

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('werkzeug') # Silencia logs HTTP do Flask
log.setLevel(logging.WARNING)
logging.info(f"Nível de log definido para: {default_log_level}")

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640
FRAME_HEIGHT_DISPLAY = 480
JPEG_QUALITY = 75
TARGET_FPS = 5
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
INITIAL_ROTATION = int(os.environ.get('ROTATE_FRAME', '0'))

# --- Variáveis Globais ---
output_frame_display = None
output_frame_lock = threading.Lock()
status_data_global = {}
status_data_lock = threading.Lock()

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
def detection_loop(cam_thread, dms_monitor, event_queue):
    """
    Loop principal executado em background para obter frames da câmara,
    processá-los com o dms_monitor e atualizar o output_frame.
    """
    global output_frame_display, status_data_global

    logging.info(f">>> Loop de deteção iniciado (Alvo: {TARGET_FPS} FPS).")

    last_process_time = time.time()

    while cam_thread.is_alive():
        start_time = time.time()

        frame = cam_thread.get_frame()

        if frame is None:
            logging.warning("Frame não recebido da câmara. A aguardar...")
            # (NOVO) Cria um placeholder se a câmara falhar temporariamente
            with output_frame_lock:
                 if output_frame_display is None: # Só na primeira vez ou se falhou antes
                     output_frame_display = create_placeholder_frame("Camera desconectada?")
            time.sleep(0.5)
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
                        event_queue.put({"event_data": event, "frame": frame.copy()}, block=False)
                    except queue.Full:
                        logging.warning("Fila de eventos cheia. Evento descartado.")

        except Exception as e:
            logging.error(f"!!! Erro fatal no process_frame: {e}", exc_info=True)
            # (NOVO) Mostra um frame de erro se a IA falhar
            with output_frame_lock:
                output_frame_display = create_placeholder_frame("Erro no processamento!")
            time.sleep(1)

        # --- Controlo de FPS ---
        processing_time = time.time() - start_time
        wait_time = TARGET_FRAME_TIME - processing_time

        if wait_time > 0:
            time.sleep(wait_time)
        else:
            current_time = time.time()
            if current_time - last_process_time > 5.0:
                 logging.warning(f"!!! LOOP LENTO. Processamento demorou {processing_time:.2f}s (Alvo era {TARGET_FRAME_TIME:.2f}s)")
                 last_process_time = current_time

    logging.info(">>> Loop de deteção terminado.")


# --- Servidor Web Flask ---

@app.route("/")
def index():
    """Rota principal que serve a página de calibração."""
    cam_source_desc = cam_thread.source_description if 'cam_thread' in globals() else "Indisponível"
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

    while True:
        frame_to_encode = None

        with output_frame_lock:
            if output_frame_display is not None:
                frame_to_encode = output_frame_display.copy()
            else:
                frame_to_encode = placeholder.copy()

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

        time.sleep(1/20)

@app.route("/video_feed")
def video_feed():
    """Rota que serve o stream de vídeo."""
    return Response(generate_video_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Rotas da API ---

@app.route("/api/config", methods=['GET', 'POST'])
def api_config():
    """API para ler e atualizar as configurações de calibração."""
    if request.method == 'GET':
        current_settings = dms_monitor.get_settings()
        current_settings['brightness'] = cam_thread.get_brightness()
        current_settings['rotation'] = cam_thread.get_rotation()
        with status_data_lock:
            current_settings['status'] = status_data_global.copy()
        return jsonify(current_settings)

    elif request.method == 'POST':
        new_settings = request.json
        if not new_settings:
            return jsonify({"success": False, "error": "No data received"}), 400

        dms_success = dms_monitor.update_settings(new_settings)

        cam_success = True
        if 'brightness' in new_settings:
            cam_thread.update_brightness(new_settings['brightness'])
        if 'rotation' in new_settings:
            cam_thread.update_rotation(new_settings['rotation'])

        if dms_success and cam_success:
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Failed to apply settings"}), 500

@app.route("/api/alerts", methods=['GET'])
def api_alerts():
    """(NOVO) API para obter a lista de alertas da base de dados SQLite."""
    alerts_list = []
    # TODO: Implementar filtros de data (year, month, day) dos query parameters
    limit = int(request.args.get('limit', 50)) # Limite padrão de 50

    try:
        # Usa o path da BD definido no event_handler
        db_path = event_handler.db_path
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=5) # Abre em modo read-only
        conn.row_factory = sqlite3.Row # Retorna resultados como dicionários
        cursor = conn.cursor()

        # Por agora, busca os últimos 'limit' alertas
        cursor.execute("SELECT id, timestamp, event_type, details, image_file FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()

        # Converte as linhas da BD para uma lista de dicionários
        for row in rows:
            alerts_list.append(dict(row))

        return jsonify(alerts_list)

    except sqlite3.Error as e:
        logging.error(f"Erro ao ler alertas da base de dados SQLite: {e}", exc_info=True)
        return jsonify({"error": "Failed to read alerts database"}), 500
    except Exception as e:
        logging.error(f"Erro inesperado ao buscar alertas: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# (NOVO) Rota ajustada para servir imagens das subpastas Ano/Mês/Dia
@app.route('/alerts/images/<path:filepath>')
def serve_alert_image(filepath):
    """Serve as imagens JPG dos alertas a partir das subpastas."""
    try:
        # Constrói o caminho completo e seguro para a imagem
        base_dir = os.path.abspath(event_handler.image_save_path)
        # Limpa o filepath para evitar ataques (ex: ../../etc/passwd)
        # os.path.normpath remove /./ e /../ mas precisamos de mais segurança
        safe_path = os.path.normpath(filepath).lstrip('./').lstrip('/')
        full_path = os.path.join(base_dir, safe_path)

        # Verifica se o caminho final ainda está dentro da pasta de imagens
        if not full_path.startswith(base_dir):
            logging.warning(f"Tentativa de acesso inválido: {filepath}")
            abort(404) # Not Found

        # Verifica se o ficheiro existe antes de tentar enviar
        if not os.path.isfile(full_path):
             logging.warning(f"Imagem não encontrada: {full_path}")
             abort(404)

        logging.debug(f"A servir imagem: {full_path}")
        return send_file(full_path, mimetype='image/jpeg')

    except Exception as e:
        logging.error(f"Erro ao servir imagem de alerta '{filepath}': {e}", exc_info=True)
        abort(500) # Internal Server Error

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':

    # (NOVO) Define as instâncias globais para que as rotas Flask as possam aceder
    cam_thread = None
    dms_monitor = None
    event_handler = None
    detection_thread = None

    try:
        logging.info(">>> Serviço DMS (Refatorado com SQLite) a iniciar...")

        event_queue = Queue(maxsize=100) # Aumenta um pouco a fila

        # (NOVO) Passa o caminho base para o EventHandler
        event_handler = EventHandler(queue=event_queue, base_save_path="/app/alerts")
        event_handler.start()

        frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)
        dms_monitor = DriverMonitor(frame_size=frame_size)

        cam_thread = CameraThread(VIDEO_SOURCE,
                                  frame_width=FRAME_WIDTH_DISPLAY,
                                  frame_height=FRAME_HEIGHT_DISPLAY,
                                  rotation_degrees=INITIAL_ROTATION)
        cam_thread.start()

        logging.info("A aguardar o primeiro frame da câmara...")
        while cam_thread.get_frame() is None:
            if not cam_thread.is_alive():
                 logging.error("!!! Thread da câmara terminou inesperadamente durante a inicialização.")
                 # (NOVO) Tenta parar o event_handler antes de sair
                 if event_handler and event_handler.is_alive():
                     event_handler.stop()
                     event_handler.join(timeout=2)
                 sys.exit(1)
            time.sleep(0.5)
        logging.info(">>> Primeiro frame recebido!")

        detection_thread = threading.Thread(target=detection_loop, args=(cam_thread, dms_monitor, event_queue))
        detection_thread.daemon = True
        detection_thread.start()

        logging.info(f">>> A iniciar servidor Flask na porta 5000...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

    except KeyboardInterrupt:
        logging.info(">>> Interrupção de teclado recebida. A encerrar...")
    except Exception as e:
        logging.error(f"!!! ERRO FATAL ao iniciar o serviço: {e}", exc_info=True)
    finally:
        logging.info(">>> A iniciar encerramento do serviço...")

        if cam_thread and cam_thread.is_alive():
            cam_thread.stop()
            cam_thread.join(timeout=2)

        if event_handler and event_handler.is_alive():
            event_handler.stop()
            event_handler.join(timeout=5) # Espera para guardar eventos pendentes

        # A thread de deteção é daemon e termina automaticamente

        logging.info(">>> Serviço DMS terminado.")
        # (NOVO) Usa os._exit para forçar a terminação se algo bloquear
        os._exit(0)
