# Documentação: Script principal da Aplicação Flask (Refatorado)
# Responsável por:
# 1. Orquestrar os threads (Câmara, Deteção)
# 2. Servir a interface web (Flask)
# 3. Lidar com a lógica de negócio (ex: salvar alertas)

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

# Habilita otimizações do OpenCV (pode ser útil no Pi)
cv2.setUseOptimized(True)

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING) # Silencia os logs 'GET /' do Flask

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640  
FRAME_HEIGHT_DISPLAY = 480 
JPEG_QUALITY = 50 
FPS_STREAMING = 30 # FPS para o stream da web
FPS_PROCESSAMENTO = 15 # Quantos frames processar por segundo (limita o uso de CPU)

# --- Variáveis Globais ---
output_frame_display = None # O frame final (com desenhos) para o stream
lock = threading.Lock() # Protege o output_frame_display
app = Flask(__name__)

# Instâncias dos nossos módulos
cam_thread = None # Thread da câmara
dms_monitor = None # Instância do nosso "cérebro" de IA

# --- Funções do Servidor Web (Flask) ---

def generate_frames():
    """Gera frames de vídeo para o stream HTTP."""
    global output_frame_display, lock
    
    # Prepara um frame 'placeholder' caso a câmara ainda não tenha iniciado
    placeholder_frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, "Aguardando camera...", (30, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    (flag, encodedImage) = cv2.imencode(".jpg", placeholder_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    placeholder_bytes = bytearray(encodedImage) if flag else None
    
    while True:
        frame_to_encode = None
        
        with lock:
            if output_frame_display is not None:
                frame_to_encode = output_frame_display.copy()

        if frame_to_encode is None:
            frame_bytes = placeholder_bytes
            time.sleep(0.5) # Se não houver frame, espera um pouco
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

        # Controla o FPS do streaming
        time.sleep(1/FPS_STREAMING) 

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

# --- Função Principal de Processamento ---

def detection_loop():
    """
    Loop principal que corre em background.
    Obtém frames da câmara e envia-os para o dms_monitor para processamento.
    """
    global dms_monitor, cam_thread, output_frame_display, lock
    
    # Define o tempo de espera entre frames para controlar o FPS
    frame_time_target = 1.0 / FPS_PROCESSAMENTO 
    
    logging.info(f">>> Loop de deteção iniciado (Alvo: {FPS_PROCESSAMENTO} FPS).")
    
    while True:
        try:
            time_start = time.time()
            
            # 1. Obter o frame mais recente da thread da câmara
            frame = cam_thread.read()
            if frame is None:
                logging.warning("Frame não recebido da câmara. A aguardar...")
                time.sleep(0.5)
                continue
                
            # 2. Redimensionar o frame para o tamanho de processamento
            frame_display = cv2.resize(frame, (FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY))
            
            # 3. Converter para escala de cinza (necessário para Dlib)
            gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
            
            # 4. Processar o frame
            processed_frame, alarms = dms_monitor.process_frame(frame_display, gray)
            
            # (Opcional: Lidar com os alarmes)
            (alarm_drowsy, alarm_distraction, alarm_cellphone) = alarms
            if alarm_drowsy or alarm_distraction or alarm_cellphone:
                # TODO: Enviar alerta para a 'Central' ou guardar na Base de Dados
                pass 
                
            # 5. Atualizar o frame de saída para o servidor web
            with lock:
                output_frame_display = processed_frame.copy()

            # 6. Controlar o FPS de processamento
            time_elapsed = time.time() - time_start
            sleep_time = max(0, frame_time_target - time_elapsed)
            time.sleep(sleep_time)

        except Exception as e:
            logging.error(f"!!! Erro no loop de deteção: {e}", exc_info=True)
            time.sleep(5) # Espera 5 segundos antes de tentar novamente

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    logging.info(">>> Serviço DMS (Refatorado) a iniciar...")
    
    try:
        # --- CORREÇÃO AQUI ---
        # 1. Define o frame_size que o DriverMonitor espera
        frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)
        
        # 2. Inicializa o "cérebro" de IA (passando o frame_size)
        dms_monitor = DriverMonitor(frame_size=frame_size)
        
        # 3. Inicializa e inicia a thread da câmara
        logging.info(f">>> A iniciar thread da câmara (Fonte: {VIDEO_SOURCE})...")
        cam_thread = CameraThread(VIDEO_SOURCE)
        cam_thread.start()
        
        # 4. Inicia o loop de deteção (em background)
        logging.info(">>> A iniciar thread de deteção...")
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()

        # 5. Inicia o servidor web (na thread principal)
        logging.info(f">>> A iniciar servidor Flask na porta 5000...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

    except FileNotFoundError as e:
         logging.error(f"!!! ERRO FATAL: Ficheiro de modelo não encontrado: {e}")
         logging.error("Verifique se os modelos Dlib e MobileNet-SSD foram descarregados corretamente no Dockerfile.")
         sys.exit(1)
    except Exception as e:
         logging.error(f"!!! ERRO FATAL ao inicializar o DriverMonitor: {e}", exc_info=True)
         sys.exit(1)

    logging.info(">>> Servidor Flask terminado.")

