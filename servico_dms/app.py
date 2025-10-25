# Documentação: Aplicação Principal Flask (Orquestrador)
# Responsável por:
# 1. Iniciar o servidor web Flask.
# 2. Iniciar o CameraThread (para ler a câmara).
# 3. Iniciar o DriverMonitor (para a lógica de IA).
# 4. Orquestrar o loop principal de deteção.

import cv2
import time
import os
import numpy as np
import threading 
import logging 
import sys
from flask import Flask, Response, render_template_string

# --- Importações Locais ---
from camera_thread import CameraThread
from dms_core import DriverMonitor

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING) # Reduz o spam de logs do Flask

# --- Configurações da Aplicação ---
# Lidas das variáveis de ambiente (definidas no docker-compose.yml)
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640
FRAME_HEIGHT_DISPLAY = 480
JPEG_QUALITY = 50
TARGET_FPS = 15 # Quantas deteções por segundo tentamos executar

# Lê a variável de rotação
# Converte a string (ex: "180") para um inteiro
ROTATE_FRAME_DEGREES = int(os.environ.get('ROTATE_FRAME', "0"))

# --- Variáveis Globais ---
output_frame_display = None
lock = threading.Lock()
app = Flask(__name__)

# Instâncias
cam_thread = None
dms_monitor = None

# --- Servidor Web Flask ---

def generate_frames():
    """Gera frames de vídeo para o stream HTTP."""
    global output_frame_display, lock
    
    # Cria um frame 'placeholder' caso a câmara ainda não tenha iniciado
    placeholder_frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, "Aguardando camera...", (30, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    (flag, encodedImage) = cv2.imencode(".jpg", placeholder_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    placeholder_bytes = bytearray(encodedImage) if flag else None
    
    while True:
        frame_to_encode = None
        
        with lock:
            if output_frame_display is not None:
                # Pega uma cópia do último frame processado
                frame_to_encode = output_frame_display.copy()

        if frame_to_encode is None:
            frame_bytes = placeholder_bytes
            time.sleep(0.5) # Evita spam se a câmara estiver desligada
        else:
            # Codifica o frame para JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if flag:
                frame_bytes = bytearray(encodedImage)
            else:
                logging.warning("Falha ao codificar frame para JPEG.")
                frame_bytes = placeholder_bytes 

        if frame_bytes is not None:
            # Envia o frame para o cliente (navegador)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Controla o FPS do stream
        time.sleep(1/30) # Tenta enviar a 30 FPS, mesmo que a deteção seja a 15 FPS

@app.route("/")
def index():
    """Rota principal que serve a página HTML."""
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
                // Script para tentar recarregar o stream em caso de falha
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

# --- Loop Principal de Deteção ---

def detection_loop():
    """Função principal executada em background para processar o vídeo."""
    global output_frame_display, lock, cam_thread, dms_monitor
    
    logging.info(f">>> Loop de deteção iniciado (Alvo: {TARGET_FPS} FPS).")
    
    # Tempo de espera entre frames para atingir o TARGET_FPS
    frame_time_seconds = 1.0 / TARGET_FPS 
    
    while True:
        try:
            start_time = time.time() # Marca o início do processamento

            # Pega o frame mais recente da thread da câmara
            frame = cam_thread.get_frame()

            if frame is None:
                logging.warning("Frame não recebido da câmara. A aguardar...")
                time.sleep(1.0)
                continue
                
            # --- NOVO: Converte para Grayscale ---
            # O Dlib (no dms_core) precisa da imagem a preto e branco para detetar faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ------------------------------------

            # Processa o frame (aqui acontece a magia da IA)
            # Passa ambos os frames
            processed_frame, status_data = dms_monitor.process_frame(frame, gray)
            
            # --- Enviar Alertas (Lógica Futura) ---
            # if status_data["alarm_drowsy"] or status_data["alarm_distraction"]:
            #    logging.warning(f"ALERTA: Sonolência={status_data['alarm_drowsy']}, Distração={status_data['alarm_distraction']}")
            #    # TODO: Enviar 'status_data' para a central via API/MQTT
            
            # Atualiza o frame de saída para o servidor web
            with lock:
                output_frame_display = processed_frame.copy()

            # Calcula o tempo gasto e espera o restante para atingir o TARGET_FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, frame_time_seconds - elapsed_time)
            
            if sleep_time == 0:
                 logging.warning(f"!!! LOOP LENTO. Processamento demorou {elapsed_time:.2f}s (Alvo era {frame_time_seconds:.2f}s)")
            
            time.sleep(sleep_time)

        except Exception as e:
            logging.error(f"!!! Erro no loop de deteção: {e}", exc_info=True)
            time.sleep(5.0) # Evita spam de logs em caso de erro

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    try:
        logging.info(">>> Serviço DMS (Refatorado) a iniciar...")
        
        # 1. Inicializa o "Cérebro" (Dlib)
        logging.info("A inicializar o DriverMonitor Core...")
        frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)
        dms_monitor = DriverMonitor(frame_size=frame_size)
        
        # 2. Inicializa os "Olhos" (Câmara)
        logging.info(f">>> A iniciar thread da câmara (Fonte: {VIDEO_SOURCE})...")
        cam_thread = CameraThread(
            VIDEO_SOURCE, 
            FRAME_WIDTH_DISPLAY, 
            FRAME_HEIGHT_DISPLAY,
            ROTATE_FRAME_DEGREES # Passa a rotação
        )
        cam_thread.start()
        
        # 3. Inicializa o "Processamento" (Loop de Deteção)
        logging.info(">>> A iniciar thread de deteção...")
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        # 4. Inicia o Servidor Web (na thread principal)
        logging.info(f">>> A iniciar servidor Flask na porta 5000...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

    except Exception as e:
         logging.error(f"!!! ERRO FATAL ao inicializar o DriverMonitor: {e}", exc_info=True)
         sys.exit(1)
    
    logging.info(">>> Servidor Flask terminado.")

