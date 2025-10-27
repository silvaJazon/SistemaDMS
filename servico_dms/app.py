# Documentação: Aplicação Principal (Servidor Flask)
# Responsável por:
# 1. Orquestrar as threads (Câmara, Deteção, Eventos).
# 2. Servir a interface web (HTML/JS) para calibração.
# 3. Fornecer uma API REST (/api/config) para ler/atualizar settings.
# 4. Servir o stream de vídeo MJPEG.

import cv2
import time
import os
import numpy as np
import threading
import logging
import sys
from flask import Flask, Response, render_template, request, jsonify # NOVO: render_template, request, jsonify
import json # NOVO

# --- Importar Módulos Locais ---
from camera_thread import CameraThread
from dms_core import DriverMonitor
from event_handler import EventHandler # NOVO: Para guardar alertas

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

# --- Configuração do Logging ---
# Define o nível de log a partir da variável de ambiente (padrão: INFO)
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Reduz o log "chato" do servidor web (werkzeug)
log_werkzeug = logging.getLogger('werkzeug')
if default_log_level != 'DEBUG':
    log_werkzeug.setLevel(logging.WARNING)

logging.info(f"Nível de log definido para: {default_log_level}")

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
ROTATE_FRAME_DEGREES = int(os.environ.get('ROTATE_FRAME', "0")) # Lê a rotação do ambiente

FRAME_WIDTH_DISPLAY = 640
FRAME_HEIGHT_DISPLAY = 480
TARGET_FPS = 5 # NOVO: Alvo de 5 FPS (1 / 5 = 0.20s), mais realista para o Pi
TARGET_FRAME_TIME = 1.0 / TARGET_FPS

JPEG_QUALITY = 60 # Qualidade do stream (50-70 é bom)

# --- Variáveis Globais ---
output_frame_display = None # O frame final a ser enviado para o stream
lock = threading.Lock() # Protege o output_frame_display

# --- Inicialização ---
app = Flask(__name__) # NOVO: O Flask agora procura a pasta 'templates'
cam_thread = None
dms_monitor = None
event_manager = None # NOVO: Gestor de Eventos

# --- Funções do Servidor Web (Flask) ---

def generate_frames():
    """Gera frames de vídeo para o stream HTTP."""
    global output_frame_display, lock
    
    # Cria um frame "placeholder" caso a câmara falhe
    placeholder_frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, "Aguardando camera...", (30, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    (flag, encodedImage) = cv2.imencode(".jpg", placeholder_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    placeholder_bytes = bytearray(encodedImage) if flag else None
    
    while True:
        frame_to_encode = None
        
        with lock:
            if output_frame_display is not None:
                # Copia o frame mais recente para evitar 'race conditions'
                frame_to_encode = output_frame_display.copy()

        if frame_to_encode is None:
            frame_bytes = placeholder_bytes
            time.sleep(0.5) # Evita spam se a câmara falhar
        else:
            # Codifica o frame para JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if flag:
                frame_bytes = bytearray(encodedImage)
            else:
                logging.warning("Falha ao codificar frame para JPEG.")
                frame_bytes = placeholder_bytes 

        # Envia o frame no formato MJPEG
        if frame_bytes is not None:
             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Controla o FPS do *stream* (não da deteção)
        time.sleep(1 / 20) # Stream a 20 FPS (pode ser mais rápido que a deteção)

@app.route("/")
def index():
    """Rota principal que serve a página HTML a partir do template."""
    global cam_thread
    # Passa as variáveis para o template HTML
    cam_source_desc = cam_thread.source_description if cam_thread else "N/A"
    
    # NOVO: Renomeado 'source' para 'cam_source_desc' para corrigir bug
    return render_template(
        "index.html", 
        width=FRAME_WIDTH_DISPLAY, 
        height=FRAME_HEIGHT_DISPLAY,
        cam_source_desc=cam_source_desc
    )

@app.route("/video_feed")
def video_feed():
    """Rota que serve o stream de vídeo."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- NOVO: Rotas de API para Calibração ---

@app.route("/api/config", methods=["GET"])
def get_config():
    """API (GET): Retorna as configurações atuais (DMS + Câmara)."""
    global dms_monitor, cam_thread
    if not dms_monitor or not cam_thread:
        return jsonify({"error": "Sistema não inicializado"}), 500
        
    try:
        # Pede as configurações aos módulos
        dms_settings = dms_monitor.get_settings()
        cam_settings = cam_thread.get_settings()
        
        # Combina os dois dicionários
        full_settings = {**dms_settings, **cam_settings}
        
        return jsonify(full_settings)
    except Exception as e:
        logging.error(f"Erro ao obter configurações: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/config", methods=["POST"])
def set_config():
    """API (POST): Atualiza as configurações (DMS + Câmara)."""
    global dms_monitor, cam_thread
    if not dms_monitor or not cam_thread:
        return jsonify({"error": "Sistema não inicializado"}), 500
        
    try:
        data = request.json
        logging.info(f"Recebida atualização de configuração via API: {data}")
        
        # 1. Envia as configurações para o DMS Core
        dms_monitor.update_settings(
            ear_thresh=data.get('ear_threshold'),
            ear_frames=data.get('ear_consec_frames'),
            distraction_angle=data.get('distraction_threshold_angle'),
            distraction_frames=data.get('distraction_consec_frames')
        )
        
        # 2. Envia as configurações para a Câmara
        if 'brightness' in data:
            cam_thread.update_brightness(data.get('brightness'))
            
        return jsonify({"success": True, "message": "Configurações atualizadas."})
        
    except Exception as e:
        logging.error(f"Erro ao atualizar configurações: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- Thread de Deteção ---

def detection_loop():
    """
    Função principal executada em background para processar vídeo.
    """
    global output_frame_display, lock, dms_monitor, cam_thread, event_manager
    
    logging.info(f">>> Loop de deteção iniciado (Alvo: {TARGET_FPS} FPS).")
    
    while True:
        start_time = time.time() # Para controlo de FPS
        
        # 1. Obter o frame da câmara
        frame = cam_thread.get_frame() # Método corrigido
        
        if frame is None:
            logging.warning("Frame não recebido da câmara. A aguardar...")
            time.sleep(0.5)
            continue
            
        # 2. Preparar Frames
        # Converte para preto e branco (para Dlib)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Correção do bug (BGR2GRAY)
        
        # 3. Processar (IA)
        # Passa ambos os frames (cor para desenhar, p&b para analisar)
        processed_frame, status_data, events = dms_monitor.process_frame(frame, gray)
        
        # 4. NOVO: Gerir Eventos
        if events and event_manager:
            for event in events:
                # Envia o evento (com a imagem original) para o 'worker'
                event_manager.log_event(event['type'], event['value'], frame)
        
        # 5. Atualizar o Stream (com 'lock')
        with lock:
            output_frame_display = processed_frame.copy()

        # 6. Controlo de FPS
        elapsed_time = time.time() - start_time
        sleep_time = TARGET_FRAME_TIME - elapsed_time
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # Avisa se o processamento for mais lento que o alvo
            logging.warning(f"!!! LOOP LENTO. Processamento demorou {elapsed_time:.2f}s (Alvo era {TARGET_FRAME_TIME:.2f}s)")

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    try:
        logging.info(">>> Serviço DMS (Refatorado) a iniciar...")
        
        # 1. Inicializar o Gestor de Eventos (Central)
        event_manager = EventHandler()
        event_manager.start()

        # 2. Inicializar o "Cérebro" (DMS Core)
        frame_size_tuple = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)
        dms_monitor = DriverMonitor(frame_size=frame_size_tuple)
        
        # 3. Inicializar os "Olhos" (Câmara)
        logging.info(f">>> A iniciar thread da câmara (Fonte: {VIDEO_SOURCE})...")
        cam_thread = CameraThread(
            VIDEO_SOURCE, 
            FRAME_WIDTH_DISPLAY, 
            FRAME_HEIGHT_DISPLAY,
            ROTATE_FRAME_DEGREES # Passa o valor da rotação
        )
        cam_thread.start()
        
        # 4. Aguardar pelo primeiro frame
        logging.info("A aguardar o primeiro frame da câmara...")
        while cam_thread.get_frame() is None:
            time.sleep(0.5)
            if not cam_thread.is_alive():
                raise RuntimeError("Falha ao iniciar a thread da câmara.")
        logging.info(">>> Primeiro frame recebido!")

        # 5. Iniciar a thread de Deteção
        logging.info(">>> A iniciar thread de deteção...")
        detect_thread = threading.Thread(target=detection_loop, daemon=True)
        detect_thread.start()

        # 6. Iniciar o Servidor Web (Flask)
        logging.info(f">>> A iniciar servidor Flask na porta 5000...")
        # 'use_reloader=False' é crucial para evitar que a app corra duas vezes
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

    except (KeyboardInterrupt, SystemExit):
        logging.info(">>> Recebido sinal de paragem. A desligar...")
    except Exception as e:
        logging.error(f"!!! ERRO FATAL no arranque: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Limpa os recursos
        if cam_thread:
            cam_thread.stop()
        if event_manager:
            event_manager.stop()
        logging.info(">>> Serviço DMS terminado.")

