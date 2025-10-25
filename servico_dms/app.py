# Documentação: Aplicação Principal (Servidor Flask)
# Responsável por:
# 1. Orquestrar as threads (Câmara, Deteção).
# 2. Servir a aplicação web Flask (streaming de vídeo e UI).

import cv2
import time
import os
import numpy as np
import threading 
import logging 
import sys
from flask import Flask, Response, render_template_string

# --- Importações Locais ---
# Importa as classes que refatorámos
from camera_thread import CameraThread
from dms_core import DriverMonitor

# --- Configuração do Logging ---
# Define o nível de log a partir da variável de ambiente (padrão: INFO)
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Reduz o log do servidor web (werkzeug) para apenas WARNINGS
log_werkzeug = logging.getLogger('werkzeug')
log_werkzeug.setLevel(logging.WARNING)
logging.info(f"Nível de log definido para: {default_log_level}")

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640   
FRAME_HEIGHT_DISPLAY = 480  
JPEG_QUALITY = 50 

# --- NOVO: Configuração de Rotação ---
# Lê a variável de ambiente. Se não existir, define como 0 (sem rotação).
# O Makefile passa o valor (ex: 180) para aqui.
ROTATION_DEGREES = int(os.environ.get('ROTATE_FRAME', "0"))
# ------------------------------------

# --- NOVO: Alvo de Desempenho ---
# O Raspberry Pi 4 consegue processar (Dlib) a cerca de 5-6 FPS (0.17s).
# Definir um alvo realista evita os avisos de "LOOP LENTO".
TARGET_FPS = 5 # Alvo realista (anteriormente era 7)
TARGET_FRAME_TIME = 1.0 / TARGET_FPS # (ex: 1/5 = 0.2 segundos)
# ---------------------------------

# --- Variáveis Globais ---
output_frame_display = None # O frame (já processado) para o stream de vídeo
lock = threading.Lock()     # Para proteger o output_frame_display
app = Flask(__name__)       # A nossa aplicação web

# Instâncias das classes
cam_thread = None           # A thread que lê a câmara
dms_monitor = None          # O "cérebro" que processa o frame (Dlib)
detection_thread = None     # A thread que corre o loop de deteção


# --- Rota Principal da UI ---

@app.route("/")
def index():
    """Rota principal que serve a página HTML (template string)."""
    
    # Este HTML é servido quando você acede a http://[IP]:5000
    html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SistemaDMS - Monitoramento</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    background-color: #121212; /* Fundo escuro */
                    color: #e0e0e0; /* Texto claro */
                    margin: 0; 
                    padding: 0; 
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                }
                h1 { 
                    color: #ffffff; /* Título a branco */
                    font-weight: 300;
                    margin-bottom: 10px;
                }
                p {
                    font-size: 0.9rem;
                    color: #888; /* Subtítulo cinzento */
                    margin-top: 0;
                }
                .stream-container {
                    border: 1px solid #333; /* Borda subtil */
                    background-color: #000;
                    border-radius: 8px; /* Cantos arredondados */
                    overflow: hidden; /* Garante que a imagem fica dentro dos cantos */
                    box-shadow: 0 4px 15px rgba(0,0,0,0.4); /* Sombra */
                    min-width: {{ width }}px;
                    min-height: {{ height }}px;
                }
                img { 
                    display: block;
                    width: 100%;
                    height: auto;
                }
            </style>
        </head>
        <body>
            <h1>SistemaDMS - Monitoramento ao Vivo</h1>
            <!-- CORREÇÃO: Alterado de {{ source }} para {{ source_text }} -->
            <p>Fonte: Câmara {{ source_text }} | Resolução: {{ width }}x{{ height }}</p>
            
            <!-- CORREÇÃO: Corrigido 'class.' para 'class=' -->
            <div class="stream-container">
                <img id="stream" src="{{ url_for('video_feed') }}" width="{{ width }}" height="{{ height }}">
            </div>
            
            <script>
                // Script simples para tentar recarregar o stream se ele falhar
                var stream = document.getElementById("stream");
                stream.onerror = function() {
                    console.log("Erro no stream. A tentar recarregar em 5 segundos...");
                    setTimeout(function() {
                        // Adiciona um timestamp para evitar o cache do navegador
                        stream.src = "{{ url_for('video_feed') }}?" + new Date().getTime();
                    }, 5000);
                };
            </script>
        </body>
        </html>
    """
    
    # CORREÇÃO: Alterado de 'source_desc' para 'source_text'
    source_text = "RTSP" if VIDEO_SOURCE.startswith("rtsp://") else f"USB ({VIDEO_SOURCE})"
    
    return render_template_string(
        html_template, 
        width=FRAME_WIDTH_DISPLAY, 
        height=FRAME_HEIGHT_DISPLAY,
        source_text=source_text # CORREÇÃO: Alterado de 'source=' para 'source_text='
    )

# --- Rota do Stream de Vídeo ---

@app.route("/video_feed")
def video_feed():
    """Rota que serve o stream de vídeo (multipart/x-mixed-replace)."""
    return Response(generate_frames_for_web(), mimetype="multipart/x-mixed-replace; boundary=frame")

def generate_frames_for_web():
    """Gera frames de vídeo para o stream HTTP."""
    global output_frame_display, lock
    
    while True:
        frame_to_encode = None
        
        # Espera até que um frame processado esteja disponível
        time.sleep(TARGET_FRAME_TIME) 
        
        with lock:
            if output_frame_display is not None:
                frame_to_encode = output_frame_display.copy()

        if frame_to_encode is None:
            # Se ainda não houver frames, continua a tentar
            continue
            
        # Codifica o frame para JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        
        if not flag:
            logging.warning("Falha ao codificar frame para JPEG.")
            continue

        # Envia o frame como bytes
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')

# --- Thread de Deteção (O Loop Principal) ---

def detection_loop():
    """
    Função principal executada em background (numa thread).
    Busca o frame mais recente da câmara e envia-o para o dms_monitor.
    """
    global output_frame_display, lock, cam_thread, dms_monitor
    
    logging.info(f">>> Loop de deteção iniciado (Alvo: {TARGET_FPS} FPS).")
    
    while True:
        start_time = time.time()
        
        # 1. Obter o frame mais recente da câmara
        frame = cam_thread.get_frame() # Usa o método get_frame() corrigido
        
        if frame is None:
            # Se a câmara ainda não estiver pronta
            logging.warning("Frame não recebido da câmara. A aguardar...")
            time.sleep(0.5)
            continue
            
        # --- CORREÇÃO DO 'TypeError' ---
        # O dms_core.py (process_frame) espera dois argumentos: o frame
        # a cores (para desenhar) e o frame a preto e branco (para o Dlib).
        
        # 2. Converter para Preto e Branco (Gray)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR_GRAY)
        except cv2.error as e:
            logging.warning(f"Erro ao converter frame para gray: {e}")
            continue
            
        # 3. Processar o frame (Dlib: EAR + Pose)
        # Passa os dois argumentos
        processed_frame, status_data = dms_monitor.process_frame(frame, gray)
        # ---------------------------------

        # 4. Atualizar o frame de saída para o servidor web
        with lock:
            output_frame_display = processed_frame.copy()
            
        # 5. Controlo de FPS
        end_time = time.time()
        processing_time = end_time - start_time
        
        if processing_time > TARGET_FRAME_TIME:
            # O processamento está a demorar mais do que o nosso alvo
            # (Isto é esperado no Raspberry Pi)
            logging.warning(f"!!! LOOP LENTO. Processamento demorou {processing_time:.2f}s (Alvo era {TARGET_FRAME_TIME:.2f}s)")
        else:
            # Se o processamento for rápido, esperamos o tempo restante
            sleep_time = TARGET_FRAME_TIME - processing_time
            logging.debug(f"Processamento: {processing_time:.3f}s, Espera: {sleep_time:.3f}s")
            time.sleep(sleep_time)


# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    try:
        logging.info(">>> Serviço DMS (Refatorado) a iniciar...")
        
        # Define o tamanho do frame que queremos
        frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)
        
        # 1. Inicializa o "Cérebro" (DMS Core)
        # (Tem de ser inicializado antes da thread, pois demora a carregar os modelos)
        dms_monitor = DriverMonitor(frame_size=frame_size)

        # 2. Inicializa os "Olhos" (Camera Thread)
        logging.info(f">>> A iniciar thread da câmara (Fonte: {VIDEO_SOURCE})...")
        cam_thread = CameraThread(
            VIDEO_SOURCE, 
            FRAME_WIDTH_DISPLAY, 
            FRAME_HEIGHT_DISPLAY,
            ROTATION_DEGREES # Passa o valor da rotação
        )
        cam_thread.start()
        
        # Espera que a câmara se ligue e capture o primeiro frame
        logging.info("A aguardar o primeiro frame da câmara...")
        while cam_thread.get_frame() is None:
            time.sleep(0.5)
            if not cam_thread.is_alive():
                logging.error("!!! ERRO FATAL: Thread da câmara falhou ao iniciar.")
                sys.exit(1)
        logging.info(">>> Primeiro frame recebido!")

        # 3. Inicializa o "Processador" (Detection Thread)
        logging.info(">>> A iniciar thread de deteção...")
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()

        # 4. Inicia o Servidor Web (Flask)
        logging.info(f">>> A iniciar servidor Flask na porta 5000...")
        # (use_reloader=False é crucial para não correr tudo duas vezes)
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

    except (KeyboardInterrupt, SystemExit):
        logging.info(">>> A terminar o serviço DMS...")
    except Exception as e:
        logging.error(f"!!! ERRO FATAL no arranque: {e}", exc_info=True)
    finally:
        if cam_thread:
            cam_thread.stop() # Sinaliza à thread da câmara para parar
        logging.info(">>> Serviço DMS terminado.")
        sys.exit(0)

