# Documentação: Aplicação Principal do SistemaDMS (Refatorada)
# Este ficheiro é o ponto de entrada.
# Responsabilidades:
# 1. Iniciar o servidor Flask (Interface Web).
# 2. Iniciar o CameraThread (para ler a câmara em background).
# 3. Iniciar o DriverMonitor (o núcleo de IA).
# 4. Orquestrar o loop principal de processamento.

import cv2
import time
import os
import numpy as np
import threading 
import logging 
import sys
from flask import Flask, Response, render_template_string

# Importa as nossas classes refatoradas
from camera_thread import CameraThread
from dms_core import DriverMonitor

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('werkzeug') # Silencia os logs HTTP do Flask
log.setLevel(logging.WARNING)

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640  
FRAME_HEIGHT_DISPLAY = 480 
JPEG_QUALITY = 50 
FPS_STREAM = 30 # Frames por segundo para o stream de vídeo

# --- Variáveis Globais ---
output_frame_display = None # O frame final (com anotações) para o stream
lock = threading.Lock() # Lock para proteger o output_frame_display
app = Flask(__name__) # A nossa aplicação web

# Variáveis para os nossos objetos
cam_thread = None
dms_monitor = None

# --- Loop Principal de Deteção (executado numa thread) ---

def detection_loop():
    """
    Loop principal que pega frames da câmara, processa-os
    e atualiza o frame de saída para o servidor web.
    """
    global output_frame_display, lock, cam_thread, dms_monitor
    
    logging.info(">>> Loop de deteção a iniciar...")
    
    # Cria um placeholder caso a câmara demore a ligar
    placeholder_frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, "Aguardando camera...", (30, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    while True:
        # Pega o último frame da thread da câmara
        frame = cam_thread.get_frame()
        
        if frame is None:
            # Se a câmara ainda não iniciou, mostra o placeholder
            with lock:
                output_frame_display = placeholder_frame.copy()
            time.sleep(0.5)
            continue
            
        # Converte para escala de cinza (necessário para o Dlib)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Processa o frame com o nosso núcleo DMS
        # A função process_frame desenha os alertas e landmarks no 'frame'
        frame_processado, status_data = dms_monitor.process_frame(frame, gray)
        
        # O status_data contém {'ear': 0.3, 'yaw': 5.1, 'alarm_drowsy': False, ...}
        # TODO: No futuro, podemos enviar 'status_data' para a base de dados ou central.
        if status_data.get('alarm_drowsy') or status_data.get('alarm_distraction'):
            # A ação (logging) já é feita dentro do dms_core.py
            pass

        # Atualiza o frame de saída para o servidor web
        with lock:
            output_frame_display = frame_processado.copy()
            
        # Controla o FPS do loop de processamento
        # Não precisamos de processar a 1000 FPS, poupamos CPU.
        time.sleep(1 / FPS_STREAM) 

    logging.info(">>> Loop de deteção terminado.")

# --- Servidor Web Flask ---

def generate_frames():
    """Gera frames de vídeo para o stream HTTP."""
    global output_frame_display, lock
    
    while True:
        frame_to_encode = None
        
        with lock:
            if output_frame_display is not None:
                frame_to_encode = output_frame_display.copy()

        if frame_to_encode is None:
            # Isto só deve acontecer mesmo no início
            time.sleep(0.1) 
            continue
        
        # Codifica o frame para JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        
        if not flag:
            logging.warning("Falha ao codificar frame para JPEG.")
            continue

        # Envia o frame como parte de uma resposta multipart
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')
        
        # Controla o FPS do stream
        time.sleep(1 / FPS_STREAM)

@app.route("/")
def index():
    """Rota principal que serve a página HTML."""
    # O HTML está agora num ficheiro separado 'index.html'
    # Mas para manter simples por agora, usamos render_template_string
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SistemaDMS - Monitoramento</title>
            <style>
                body { font-family: sans-serif; background-color: #222; color: #eee; margin: 0; padding: 20px; text-align: center;}
                h1 { color: #eee; }
                img { border: 1px solid #555; background-color: #000; max-width: 95%; height: auto; margin-top: 20px;}
            </style>
        </head>
        <body>
            <h1>SistemaDMS - Monitoramento ao Vivo</h1>
            <img id="stream" src="{{ url_for('video_feed') }}" width="{{ width }}" height="{{ height }}">
            <script>
                var stream = document.getElementById("stream");
                stream.onerror = function() {
                    console.log("Erro no stream, a tentar recarregar em 5s...");
                    setTimeout(function() {
                        stream.src = "{{ url_for('video_feed') }}?" + new Date().getTime();
                    }, 5000);
                };
            </script>
        </body>
        </html>
    """, width=FRAME_WIDTH_DISPLAY, height=FRAME_HEIGHT_DISPLAY)

@app.route("/video_feed")
def video_feed():
    """Rota que serve o stream de vídeo."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    logging.info(">>> Serviço DMS (Refatorado) a iniciar...")

    # 1. Inicia o Núcleo de Deteção (carrega os modelos Dlib)
    try:
        dms_monitor = DriverMonitor()
    except Exception as e:
        logging.error(f"!!! ERRO FATAL ao inicializar o DriverMonitor: {e}", exc_info=True)
        sys.exit(1)

    # 2. Inicia a Thread da Câmara
    try:
        cam_thread = CameraThread(VIDEO_SOURCE, FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY)
        cam_thread.start()
        logging.info("Thread da câmara iniciada.")
    except Exception as e:
        logging.error(f"!!! ERRO FATAL ao iniciar a CameraThread: {e}", exc_info=True)
        sys.exit(1)

    # 3. Inicia a Thread de Deteção
    try:
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        logging.info("Thread de deteção iniciada.")
    except Exception as e:
        logging.error(f"!!! ERRO FATAL ao iniciar a detection_loop: {e}", exc_info=True)
        sys.exit(1)

    # 4. Inicia o Servidor Web (na thread principal)
    logging.info(f">>> A iniciar servidor Flask na porta 5000...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        logging.error(f"Erro ao iniciar servidor Flask: {e}", exc_info=True)
        cam_thread.stop() # Tenta parar a thread da câmara
        sys.exit(1)

    logging.info(">>> Servidor Flask terminado.")
    cam_thread.stop() # Garante que a thread da câmara para
