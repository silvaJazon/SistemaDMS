# Documentação: Servidor Web Principal (Flask) - O Orquestrador
# Responsável por:
# 1. Servir a interface web (index.html).
# 2. Servir o stream de vídeo (video_feed).
# 3. Fornecer uma API (/api/config) para calibração.
# 4. Gerir as threads de câmara, deteção e eventos.

import cv2
import time
import os
import numpy as np
import threading 
import logging 
import sys
from flask import Flask, Response, render_template, jsonify, request
from queue import Queue

# Módulos do nosso sistema
from camera_thread import CameraThread
from dms_core import DriverMonitor
from event_handler import EventHandler

# --- Configuração do Logging ---
# Define o nível de log padrão (pode ser sobrescrito pela variável de ambiente)
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING) # Reduz o "barulho" do Flask nos logs

logging.info(f"Nível de log definido para: {default_log_level}")

# --- Configurações da Aplicação ---
# Lê as variáveis de ambiente (com valores padrão)
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640   
FRAME_HEIGHT_DISPLAY = 480 
JPEG_QUALITY = 50 
ROTATE_FRAME = int(os.environ.get('ROTATE_FRAME', 0)) # Lê a rotação

# Otimização de Desempenho (FPS Alvo)
# O Pi 4B, com Dlib, consegue ~5-7 FPS.
TARGET_FPS = 5 
TARGET_FRAME_TIME = 1.0 / TARGET_FPS # Tempo ideal por frame (ex: 0.2s para 5 FPS)

# --- Variáveis Globais ---
output_frame_display = None # O frame que será enviado para o stream
lock = threading.Lock() # Protege o output_frame_display
app = Flask(__name__)

# Módulos centrais
cam_thread = None       # A thread da câmara
dms_monitor = None      # O "cérebro" (Dlib)
event_handler = None    # A "central" (guarda os alertas)
event_queue = Queue()   # A fila para comunicação assíncrona

# --- Funções do Servidor Web (Flask) ---

def generate_frames():
    """Gera frames de vídeo para o stream HTTP."""
    global output_frame_display, lock
    
    # Cria um frame "placeholder" caso a câmara esteja a demorar
    placeholder_frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, "Aguardando camera...", (30, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    (flag, encodedImage) = cv2.imencode(".jpg", placeholder_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    placeholder_bytes = bytearray(encodedImage) if flag else None
    
    while True:
        frame_to_encode = None
        
        with lock:
            if output_frame_display is not None:
                # Copia o último frame processado
                frame_to_encode = output_frame_display.copy()
        
        if frame_to_encode is None:
            frame_bytes = placeholder_bytes
            time.sleep(0.5) # Não envia frames nulos muito rápido
        else:
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if flag:
                frame_bytes = bytearray(encodedImage)
            else:
                logging.warning("Falha ao codificar frame para JPEG.")
                frame_bytes = placeholder_bytes 

        if frame_bytes is not None:
            # Envia o frame no formato multipart
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Controla o FPS do stream
        time.sleep(TARGET_FRAME_TIME)

# --- Rotas HTTP (Interface Web e API) ---

@app.route("/")
def index():
    """Rota principal que serve a página HTML (templates/index.html)."""
    # Passa as variáveis para o template HTML
    return render_template(
        "index.html", 
        width=FRAME_WIDTH_DISPLAY, 
        height=FRAME_HEIGHT_DISPLAY,
        # (CORREÇÃO) Renomeia a variável de 'source' para 'cam_source_desc'
        cam_source_desc=cam_thread.source_description if cam_thread else "N/A"
    )

@app.route("/video_feed")
def video_feed():
    """Rota que serve o stream de vídeo."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- API de Calibração ---

@app.route("/api/config", methods=["GET"])
def get_config():
    """API (GET): Retorna as configurações atuais para os sliders."""
    global dms_monitor, cam_thread
    try:
        if dms_monitor and cam_thread:
            # 1. Obtém as configurações do Dlib (EAR, Ângulo, etc.)
            dms_settings = dms_monitor.get_settings()
            
            # 2. (CORREÇÃO) Obtém a configuração de Brilho
            #    O nome da função era 'get_brightness', não 'get_settings'
            dms_settings['brightness'] = cam_thread.get_brightness()
            
            return jsonify(dms_settings)
        else:
            return jsonify({"error": "Módulos não inicializados"}), 500
    except Exception as e:
        # (CORREÇÃO) Log mais detalhado do erro
        logging.error(f"Erro ao obter configurações: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/config", methods=["POST"])
def set_config():
    """API (POST): Recebe e aplica as novas configurações dos sliders."""
    global dms_monitor, cam_thread
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "Nenhum dado JSON recebido"}), 400

        logging.info(f"Recebidas novas configurações via API: {data}")

        # 1. Atualiza as configurações do Dlib (EAR, Ângulo, etc.)
        if dms_monitor:
            dms_monitor.update_settings(data)

        # 2. Atualiza a configuração de Brilho (na thread da câmara)
        if cam_thread and 'brightness' in data:
            cam_thread.update_brightness(data.get('brightness'))

        return jsonify({"status": "success", "message": "Configurações aplicadas"})
    except Exception as e:
        logging.error(f"Erro ao aplicar configurações: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Thread de Deteção (O "Coração" do Sistema) ---

def detection_loop():
    """
    Thread principal que busca frames da câmara, envia-os para o "cérebro" (DMS Core)
    e atualiza o stream de vídeo.
    """
    global output_frame_display, lock, cam_thread, dms_monitor, event_queue
    
    logging.info(f">>> Loop de deteção iniciado (Alvo: {TARGET_FPS} FPS).")
    
    last_frame_time = time.time()

    while True:
        try:
            start_time = time.time() # Mede o tempo de início do processamento

            # 1. Obtém o último frame da thread da câmara
            frame = cam_thread.get_frame()
            if frame is None:
                # logging.warning("Frame não recebido da câmara. A aguardar...")
                time.sleep(0.1)
                continue
                
            # 2. (CORREÇÃO) Converte para escala de cinza
            #    O nome correto é COLOR_BGR2GRAY (com '2')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 3. Envia para o "Cérebro" (Dlib) para processamento
            # O dms_core processa E desenha no 'frame'
            processed_frame, events = dms_monitor.process_frame(frame, gray)
            
            # 4. Verifica se houveram alertas
            if events:
                for event in events:
                    logging.warning(f"*** ALERTA DETETADO *** Tipo: {event['type']}")
                    # Envia o evento (com o frame original) para a fila do Event Handler
                    event_queue.put({"event": event, "frame": frame})

            # 5. Atualiza o frame de saída para o servidor web
            with lock:
                output_frame_display = processed_frame

            # 6. Controlo de FPS
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verifica se estamos a demorar mais do que o nosso alvo
            if processing_time > TARGET_FRAME_TIME:
                logging.warning(f"!!! LOOP LENTO. Processamento demorou {processing_time:.2f}s (Alvo era {TARGET_FRAME_TIME:.2f}s)")
            else:
                # Se fomos rápidos, esperamos o tempo restante
                sleep_time = TARGET_FRAME_TIME - processing_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            logging.error(f"!!! Erro fatal no loop de deteção: {e}", exc_info=True)
            time.sleep(5.0) # Espera 5 segundos antes de tentar novamente

# --- Ponto de Entrada Principal ---
if __name__ == '__main__':
    try:
        logging.info(">>> Serviço DMS (Refatorado) a iniciar...")

        # 1. Inicia o Gestor de Eventos (A "Central")
        # (CORREÇÃO) Usa um argumento nomeado (keyword argument) para evitar confusão
        event_handler = EventHandler(queue=event_queue)
        event_handler.start()
        
        # 2. Inicia o "Cérebro"
        frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)
        dms_monitor = DriverMonitor(frame_size=frame_size)
        
        # 3. Inicia os "Olhos" (Thread da Câmara)
        cam_thread = CameraThread(VIDEO_SOURCE, FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY, ROTATE_FRAME)
        cam_thread.start()
        
        # 4. Aguarda o primeiro frame
        logging.info("A aguardar o primeiro frame da câmara...")
        while cam_thread.get_frame() is None:
            time.sleep(0.5)
        logging.info(">>> Primeiro frame recebido!")

        # 5. Inicia a Thread de Deteção (O "Coração")
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        # 6. Inicia o Servidor Web (Flask)
        logging.info(f">>> A iniciar servidor Flask na porta 5000...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

    except Exception as e:
        logging.error(f"!!! ERRO FATAL ao iniciar o serviço: {e}", exc_info=True)
        if event_handler:
            event_handler.stop() # Para a thread do gestor de eventos
        sys.exit(1)
    finally:
        if event_handler:
            event_handler.stop() # Garante que a thread para
        logging.info(">>> Servidor Flask terminado.")

