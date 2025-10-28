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
TARGET_FPS = 5 # Mantém 5 FPS por agora, otimizaremos depois
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
    # Ajusta tamanho da fonte para caber melhor
    font_scale = 0.8
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (FRAME_WIDTH_DISPLAY - text_width) // 2
    text_y = (FRAME_HEIGHT_DISPLAY + text_height) // 2
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
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
            with output_frame_lock:
                 # Cria placeholder APENAS se não houver frame anterior válido
                 if output_frame_display is None or output_frame_display.shape[0] != FRAME_HEIGHT_DISPLAY:
                     output_frame_display = create_placeholder_frame("Camera desconectada?")
            time.sleep(0.5)
            continue

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Retorna 3 valores: frame processado, lista de eventos, dados de status
            processed_frame, events, status_data = dms_monitor.process_frame(frame, gray) # Passa frame original, dms_core faz cópia

            with output_frame_lock:
                output_frame_display = processed_frame # Já é uma cópia feita no dms_core

            with status_data_lock:
                # Garante que status_data é sempre um dicionário
                status_data_global = status_data if isinstance(status_data, dict) else {}

            if events:
                for event in events:
                    try:
                        # Envia o evento E o frame ORIGINAL para ser guardado
                        # A cópia é feita aqui para garantir que o frame não muda antes de ir para a fila
                        event_queue.put({"event_data": event, "frame": frame.copy()}, block=False)
                    except queue.Full:
                        logging.warning("Fila de eventos cheia. Evento descartado.")

        except Exception as e:
            logging.error(f"!!! Erro fatal no process_frame: {e}", exc_info=True)
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
    cam_source_desc = cam_thread.source_description if 'cam_thread' in globals() and cam_thread else "Indisponível"
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
            # Verifica se output_frame_display é válido antes de copiar
            if output_frame_display is not None and isinstance(output_frame_display, np.ndarray) and output_frame_display.size > 0:
                frame_to_encode = output_frame_display.copy()
            else:
                frame_to_encode = placeholder.copy()

        try:
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode,
                                                 [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

            if not flag:
                logging.warning("Falha ao codificar frame para JPEG.")
                # Tenta codificar o placeholder se o frame normal falhar
                (flag, encodedImage) = cv2.imencode(".jpg", placeholder,
                                                     [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                if not flag: continue # Se até o placeholder falhar, ignora este ciclo

            frame_bytes = bytearray(encodedImage)

            # Envia o frame no formato multipart
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')
        except cv2.error as cv_err:
             # Erro específico do OpenCV (ex: frame inválido)
             logging.error(f"Erro OpenCV ao codificar frame: {cv_err}")
             # Envia placeholder
             (flag, encodedImage) = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
             if flag:
                 frame_bytes = bytearray(encodedImage)
                 yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as encode_err:
             logging.error(f"Erro ao codificar ou enviar frame: {encode_err}")
             # Envia placeholder em caso de erro genérico
             (flag, encodedImage) = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
             if flag:
                 frame_bytes = bytearray(encodedImage)
                 yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Controla o FPS do *stream* (não precisa ser igual ao da deteção)
        # Reduzir um pouco se a rede for lenta
        time.sleep(1/15) # Stream a 15 FPS para a UI

@app.route("/video_feed")
def video_feed():
    """Rota que serve o stream de vídeo."""
    return Response(generate_video_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Rotas da API ---

@app.route("/api/config", methods=['GET', 'POST'])
def api_config():
    """API para ler e atualizar as configurações de calibração."""
    global dms_monitor # Garante acesso à instância global
    global cam_thread

    # Verifica se os objetos foram inicializados
    if not dms_monitor or not cam_thread:
         return jsonify({"error": "Serviço ainda não inicializado."}), 503

    if request.method == 'GET':
        try:
            current_settings = dms_monitor.get_settings()
            # Adiciona configurações da câmara
            current_settings['brightness'] = cam_thread.get_brightness()
            current_settings['rotation'] = cam_thread.get_rotation()
            # Adiciona dados de status
            with status_data_lock:
                # Garante que status_data_global é um dicionário antes de copiar
                current_settings['status'] = status_data_global.copy() if isinstance(status_data_global, dict) else {}
            return jsonify(current_settings)
        except Exception as e:
             logging.error(f"Erro no GET /api/config: {e}", exc_info=True)
             return jsonify({"error": "Erro interno ao obter configurações"}), 500


    elif request.method == 'POST':
        try:
            new_settings = request.json
            if not new_settings:
                return jsonify({"success": False, "error": "No data received"}), 400

            # Atualiza configurações do DMS Core (agora inclui MAR)
            dms_success = dms_monitor.update_settings(new_settings)

            # Atualiza configurações da Câmara (Brilho e Rotação)
            cam_success = True # Assume sucesso
            if 'brightness' in new_settings:
                # Validação básica
                try:
                    brightness_val = float(new_settings['brightness'])
                    cam_thread.update_brightness(brightness_val)
                except (ValueError, TypeError):
                    logging.warning(f"Valor de brilho inválido recebido: {new_settings['brightness']}")
                    cam_success = False

            if 'rotation' in new_settings:
                try:
                    rotation_val = int(new_settings['rotation'])
                    if rotation_val in [0, 90, 180, 270]:
                        cam_thread.update_rotation(rotation_val)
                    else:
                        logging.warning(f"Valor de rotação inválido recebido: {new_settings['rotation']}")
                        cam_success = False
                except (ValueError, TypeError):
                     logging.warning(f"Valor de rotação inválido recebido: {new_settings['rotation']}")
                     cam_success = False

            if dms_success and cam_success:
                return jsonify({"success": True})
            else:
                # Monta uma mensagem de erro mais específica
                errors = []
                if not dms_success: errors.append("Falha ao aplicar conf. DMS.")
                if not cam_success: errors.append("Falha ao aplicar conf. da câmara.")
                error_msg = " ".join(errors) if errors else "Erro desconhecido ao aplicar configurações."
                return jsonify({"success": False, "error": error_msg}), 500
        except Exception as e:
            logging.error(f"Erro no POST /api/config: {e}", exc_info=True)
            return jsonify({"success": False, "error": "Erro interno ao guardar configurações"}), 500


@app.route("/api/alerts", methods=['GET'])
def api_alerts():
    """API para obter a lista de alertas da base de dados SQLite."""
    global event_handler # Garante acesso à instância global
    if not event_handler:
        return jsonify({"error": "Serviço ainda não inicializado."}), 503

    alerts_list = []
    # (Futuro) Adicionar filtros: request.args.get('year'), request.args.get('month'), etc.
    limit = int(request.args.get('limit', 50)) # Limite padrão de 50

    try:
        db_path = event_handler.db_path
        # Verifica se o ficheiro da BD existe
        if not os.path.exists(db_path):
             logging.warning("Ficheiro da base de dados de alertas não encontrado.")
             return jsonify([]) # Retorna lista vazia se a BD não existe

        # Timeout aumentado e modo read-only
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # TODO: Adicionar cláusula WHERE com base nos filtros de data
        cursor.execute("SELECT id, timestamp, event_type, details, image_file FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            alerts_list.append(dict(row))

        return jsonify(alerts_list)

    except sqlite3.OperationalError as e:
        # Erro comum se a tabela ainda não foi criada ou BD corrompida
        logging.error(f"Erro operacional SQLite ao ler alertas: {e}", exc_info=True)
        return jsonify({"error": f"Erro na base de dados: {e}"}), 500
    except sqlite3.DatabaseError as e:
        # Outros erros da base de dados
        logging.error(f"Erro SQLite DatabaseError ao ler alertas: {e}", exc_info=True)
        return jsonify({"error": f"Erro na base de dados: {e}"}), 500
    except Exception as e:
        logging.error(f"Erro inesperado ao buscar alertas: {e}", exc_info=True)
        return jsonify({"error": "Erro interno do servidor"}), 500


@app.route('/alerts/images/<path:filepath>')
def serve_alert_image(filepath):
    """Serve as imagens JPG dos alertas a partir das subpastas."""
    global event_handler # Garante acesso à instância global
    if not event_handler:
         abort(503) # Service Unavailable

    try:
        # Validação mais robusta do caminho
        base_dir = os.path.abspath(event_handler.image_save_path)
        # Junta o diretório base com o caminho do ficheiro de forma segura
        full_path = os.path.abspath(os.path.join(base_dir, filepath))

        # Verifica se o caminho final está realmente dentro do diretório base
        # e se o ficheiro existe
        if not full_path.startswith(base_dir) or '..' in filepath or not os.path.isfile(full_path):
            logging.warning(f"Tentativa de acesso inválido ou ficheiro não encontrado: {filepath}")
            abort(404)

        logging.debug(f"A servir imagem: {full_path}")
        # Usa send_file para melhor gestão de mimetypes e caching
        return send_file(full_path, mimetype='image/jpeg')

    except Exception as e:
        logging.error(f"Erro ao servir imagem de alerta '{filepath}': {e}", exc_info=True)
        abort(500)

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':

    cam_thread = None
    dms_monitor = None
    event_handler = None
    detection_thread = None

    try:
        logging.info(">>> Serviço DMS (com Bocejo + SQLite) a iniciar...")

        event_queue = Queue(maxsize=100)

        # O event_handler usa o path padrão /app/alerts
        event_handler = EventHandler(queue=event_queue)
        event_handler.start()

        frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)
        dms_monitor = DriverMonitor(frame_size=frame_size)

        cam_thread = CameraThread(VIDEO_SOURCE,
                                  frame_width=FRAME_WIDTH_DISPLAY,
                                  frame_height=FRAME_HEIGHT_DISPLAY,
                                  rotation_degrees=INITIAL_ROTATION)
        cam_thread.start()

        logging.info("A aguardar o primeiro frame da câmara...")
        # Loop mais robusto para esperar pela câmara
        wait_start_time = time.time()
        while cam_thread.get_frame() is None:
            if not cam_thread.is_alive():
                 logging.error("!!! Thread da câmara terminou inesperadamente durante a inicialização.")
                 if event_handler and event_handler.is_alive(): event_handler.stop()
                 sys.exit(1)
            if time.time() - wait_start_time > 30: # Timeout de 30s
                 logging.error("!!! Timeout à espera do primeiro frame da câmara.")
                 if event_handler and event_handler.is_alive(): event_handler.stop()
                 if cam_thread and cam_thread.is_alive(): cam_thread.stop()
                 sys.exit(1)
            time.sleep(0.5)
        logging.info(">>> Primeiro frame recebido!")

        detection_thread = threading.Thread(target=detection_loop, args=(cam_thread, dms_monitor, event_queue), name="DetectionThread")
        detection_thread.daemon = True
        detection_thread.start()

        logging.info(f">>> A iniciar servidor Flask na porta 5000...")
        # Usa waitress como servidor WSGI de produção (mais robusto que o dev server)
        try:
             # Tenta importar 'serve' de 'waitress'
             from waitress import serve
             # Inicia o servidor waitress com 8 threads (ajustável)
             serve(app, host='0.0.0.0', port=5000, threads=8)
        except ImportError:
             # Se 'waitress' não estiver instalado, avisa e usa o servidor de desenvolvimento
             logging.warning("Pacote 'waitress' não encontrado. Adicione 'waitress' ao requirements.txt para um servidor de produção.")
             logging.warning("A usar o servidor de desenvolvimento Flask (NÃO RECOMENDADO PARA PRODUÇÃO).")
             app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)


    except KeyboardInterrupt:
        logging.info(">>> Interrupção de teclado recebida. A encerrar...")
    except Exception as e:
        # Loga o erro fatal que impediu o arranque
        logging.critical(f"!!! ERRO FATAL ao iniciar o serviço: {e}", exc_info=True)
    finally:
        # --- Encerramento Gracioso ---
        logging.info(">>> A iniciar encerramento do serviço...")

        # Sinaliza paragem de forma mais ordenada
        threads_to_join = []
        if cam_thread and cam_thread.is_alive():
            logging.info("A parar thread da câmara...")
            cam_thread.stop()
            threads_to_join.append(cam_thread)

        # A thread de deteção é daemon, mas esperamos que termine se a cam_thread parar
        if detection_thread and detection_thread.is_alive():
             logging.debug("A aguardar thread de deteção...") # Log de debug
             threads_to_join.append(detection_thread)

        if event_handler and event_handler.is_alive():
            logging.info("A parar thread do gestor de eventos...")
            event_handler.stop() # Envia 'None' para a fila
            threads_to_join.append(event_handler)

        # Espera que as threads terminem
        logging.info(f"A aguardar {len(threads_to_join)} threads...")
        for t in threads_to_join:
            try:
                # Dá um timeout razoável para cada thread terminar
                t.join(timeout=5)
                if t.is_alive():
                     logging.warning(f"Thread {t.name} não terminou após timeout!")
                else:
                     logging.info(f"Thread {t.name} terminada.")
            except Exception as join_e:
                 logging.warning(f"Erro ao esperar pela thread {t.name}: {join_e}")


        logging.info(">>> Serviço DMS terminado.")
        # Usa os._exit(0) para garantir que o processo termina mesmo que alguma thread bloqueie
        os._exit(0)

