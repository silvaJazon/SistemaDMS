# Documentação: Aplicação Principal Flask para o SistemaDMS
# Orquestra a captura de vídeo, processamento de IA e a interface web.
# (Atualizado com Visualizador de Alertas, Refatoração HTML, APIs, Rotação Dinâmica, Correção Unpack)

import cv2
import time
import os
import numpy as np
import threading
import logging
import sys
from flask import Flask, Response, render_template, jsonify, request, send_from_directory
from queue import Queue
import json
from datetime import datetime

# Importa os nossos módulos
from camera_thread import CameraThread
from dms_core import DriverMonitor
from event_handler import EventHandler # A nossa "Central"

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
JPEG_QUALITY = 75 # Aumenta um pouco a qualidade para a UI
TARGET_FPS = 5 # Alvo de FPS mais realista para o RPi (5 FPS = 0.2s por frame)
TARGET_FRAME_TIME = 1.0 / TARGET_FPS

# (NOVO) Rotação inicial (lida do ambiente, 0 por padrão)
INITIAL_ROTATION = int(os.environ.get('ROTATE_FRAME', '0'))

# --- Variáveis Globais ---
output_frame_display = None # O último frame processado para o stream
output_frame_lock = threading.Lock() # Protege o acesso ao output_frame_display
status_data_global = {} # (NOVO) Últimos dados de status (EAR, Yaw) para a API
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

        # Obtém o frame mais recente da thread da câmara
        frame = cam_thread.get_frame()

        if frame is None:
            logging.warning("Frame não recebido da câmara. A aguardar...")
            time.sleep(0.5)
            continue

        try:
            # Converte para tons de cinza (necessário para Dlib)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Processamento Principal ---
            # (NOVO) Recebe 3 valores
            processed_frame, events, status_data = dms_monitor.process_frame(frame.copy(), gray)
            # -------------------------------

            # Atualiza o frame de saída para o servidor web (thread-safe)
            with output_frame_lock:
                output_frame_display = processed_frame.copy()
            
            # (NOVO) Atualiza os dados de status globais (thread-safe)
            with status_data_lock:
                status_data_global = status_data.copy()

            # (NOVO) Envia eventos para a fila do EventHandler (se houver)
            if events:
                for event in events:
                    # Envia o evento E o frame ORIGINAL (sem desenhos) para ser guardado
                    try:
                        event_queue.put({"event_data": event, "frame": frame.copy()}, block=False)
                    except queue.Full:
                        logging.warning("Fila de eventos cheia. Evento descartado.")

        except Exception as e:
            logging.error(f"!!! Erro fatal no process_frame: {e}", exc_info=True)
            # Em caso de erro grave no processamento, para evitar spam,
            # podemos parar a thread ou apenas logar e continuar
            time.sleep(1) # Pausa para evitar spam de logs

        # --- Controlo de FPS ---
        processing_time = time.time() - start_time
        wait_time = TARGET_FRAME_TIME - processing_time

        if wait_time > 0:
            time.sleep(wait_time)
        else:
            # Loga se o processamento estiver consistentemente lento
            # Evita spam logando apenas se a última vez foi há mais de 5s
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
                           source_desc=cam_source_desc, # (NOVO) Renomeado para evitar conflito
                           width=FRAME_WIDTH_DISPLAY,
                           height=FRAME_HEIGHT_DISPLAY)

@app.route("/alerts")
def alerts_page():
    """(NOVO) Rota que serve a página de histórico de alertas."""
    return render_template("alerts.html")


def generate_video_stream():
    """Gera frames de vídeo para o stream HTTP (usado por /video_feed)."""
    global output_frame_display
    
    placeholder = create_placeholder_frame()
    
    while True:
        frame_to_encode = None
        
        # Obtém o último frame processado de forma segura
        with output_frame_lock:
            if output_frame_display is not None:
                frame_to_encode = output_frame_display.copy()
            else:
                frame_to_encode = placeholder.copy() # Usa placeholder se nada foi processado ainda

        # Codifica o frame como JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode,
                                             [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        if not flag:
            logging.warning("Falha ao codificar frame para JPEG.")
            # Se a codificação falhar, envia o placeholder
            (flag, encodedImage) = cv2.imencode(".jpg", placeholder,
                                                 [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if not flag: continue # Se até o placeholder falhar, ignora

        frame_bytes = bytearray(encodedImage)

        # Envia o frame no formato multipart
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

        # Controla o FPS do *stream* (não precisa ser igual ao da deteção)
        time.sleep(1/20) # Stream a 20 FPS para a UI

@app.route("/video_feed")
def video_feed():
    """Rota que serve o stream de vídeo."""
    return Response(generate_video_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# --- (NOVO) Rotas da API ---

@app.route("/api/config", methods=['GET', 'POST'])
def api_config():
    """API para ler e atualizar as configurações de calibração."""
    if request.method == 'GET':
        # Combina configurações do DMS e da Câmara
        current_settings = dms_monitor.get_settings()
        current_settings['brightness'] = cam_thread.get_brightness()
        current_settings['rotation'] = cam_thread.get_rotation() # Adiciona rotação
        # (NOVO) Adiciona os dados de status
        with status_data_lock:
            current_settings['status'] = status_data_global.copy()
        return jsonify(current_settings)
        
    elif request.method == 'POST':
        new_settings = request.json
        if not new_settings:
            return jsonify({"success": False, "error": "No data received"}), 400
            
        # Atualiza configurações do DMS Core
        dms_success = dms_monitor.update_settings(new_settings)
        
        # Atualiza configurações da Câmara (Brilho e Rotação)
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
    """(NOVO) API para obter a lista de alertas do ficheiro JSONL."""
    alerts_list = []
    log_path = event_handler.log_file_path # Obtém o caminho do ficheiro do handler
    try:
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    try:
                        alerts_list.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        logging.warning(f"Linha mal formada no log de alertas: {line.strip()}")
        # Retorna os alertas ordenados do mais recente para o mais antigo
        return jsonify(sorted(alerts_list, key=lambda x: x.get('timestamp', ''), reverse=True))
    except Exception as e:
        logging.error(f"Erro ao ler log de alertas: {e}")
        return jsonify({"error": "Failed to read alerts log"}), 500

@app.route('/alerts/images/<path:filename>')
def serve_alert_image(filename):
    """(NOVO) Rota para servir as imagens JPG dos alertas."""
    try:
        return send_from_directory(event_handler.save_path, filename)
    except FileNotFoundError:
        return "Image not found", 404
    except Exception as e:
        logging.error(f"Erro ao servir imagem de alerta '{filename}': {e}")
        return "Internal server error", 500

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    
    detection_thread = None
    
    try:
        # --- Inicialização ---
        logging.info(">>> Serviço DMS (Refatorado) a iniciar...")

        # (NOVO) Cria a fila para comunicação entre threads
        event_queue = Queue(maxsize=50) # Limita a fila para evitar consumo excessivo de memória

        # (NOVO) Inicializa o Gestor de Eventos (Central)
        event_handler = EventHandler(queue=event_queue) # Usa argumento nomeado
        event_handler.start() # Inicia a thread do gestor

        # Inicializa o Núcleo DMS (IA)
        frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)
        dms_monitor = DriverMonitor(frame_size=frame_size)

        # Inicializa e inicia a Thread da Câmara
        cam_thread = CameraThread(VIDEO_SOURCE,
                                  frame_width=FRAME_WIDTH_DISPLAY,
                                  frame_height=FRAME_HEIGHT_DISPLAY,
                                  rotation_degrees=INITIAL_ROTATION) # Passa rotação inicial
        cam_thread.start()
        
        # Aguarda a câmara conectar e obter o primeiro frame
        logging.info("A aguardar o primeiro frame da câmara...")
        while cam_thread.get_frame() is None:
            if not cam_thread.is_alive():
                 logging.error("!!! Thread da câmara terminou inesperadamente durante a inicialização.")
                 sys.exit(1)
            time.sleep(0.5)
        logging.info(">>> Primeiro frame recebido!")

        # Inicializa e inicia a Thread de Deteção
        detection_thread = threading.Thread(target=detection_loop, args=(cam_thread, dms_monitor, event_queue))
        detection_thread.daemon = True
        detection_thread.start()

        # --- Inicia o Servidor Flask ---
        logging.info(f">>> A iniciar servidor Flask na porta 5000...")
        # Nota: O debug=False e use_reloader=False são importantes para produção
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

    except KeyboardInterrupt:
        logging.info(">>> Interrupção de teclado recebida. A encerrar...")
    except Exception as e:
        logging.error(f"!!! ERRO FATAL ao iniciar o serviço: {e}", exc_info=True)
    finally:
        logging.info(">>> A iniciar encerramento do serviço...")
        
        # Sinaliza às threads para pararem
        if 'cam_thread' in locals() and cam_thread.is_alive():
            cam_thread.stop()
            cam_thread.join(timeout=2) # Espera um pouco pela thread
            
        # A thread de deteção é daemon, termina automaticamente, mas podemos sinalizar
        # (Se tivéssemos um loop com `while running:`, chamaríamos stop() aqui)

        if 'event_handler' in locals() and event_handler.is_alive():
            event_handler.stop()
            event_handler.join(timeout=5) # Espera mais tempo para guardar eventos pendentes

        logging.info(">>> Serviço DMS terminado.")
        sys.exit(0) # Garante que o container termina

