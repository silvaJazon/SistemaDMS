# Documentação: Aplicação Principal Flask (Servidor Web)
# Responsável por:
# 1. Orquestrar as threads (Câmara, Deteção).
# 2. Servir o stream de vídeo (MJPEG).
# 3. Servir a interface web de calibração.
# 4. Fornecer uma API REST (/api/config) para atualizar os parâmetros.

import cv2
import time
import os
import numpy as np
import threading 
import logging 
import sys
from flask import Flask, Response, render_template_string, jsonify, request
import dlib 
from scipy.spatial import distance as dist 
import math 

# --- Importações Locais ---
from camera_thread import CameraThread
from dms_core import DriverMonitor # NOVO: Classe principal do DMS Core

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Reduz o ruído do servidor web nos logs
log = logging.getLogger('werkzeug')
if default_log_level != 'DEBUG':
    log.setLevel(logging.WARNING)

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
# NOVO: Lê o ângulo de rotação da variável de ambiente
ROTATE_FRAME = int(os.environ.get('ROTATE_FRAME', 0))

FRAME_WIDTH_DISPLAY = 640   
FRAME_HEIGHT_DISPLAY = 480 
JPEG_QUALITY = 50 

# --- NOVO: Configuração de Desempenho (FPS) ---
# Ajustado para 5 FPS, um alvo realista para o RPi
TARGET_FPS = 5
TARGET_FRAME_TIME = 1.0 / TARGET_FPS # Tempo-alvo por frame em segundos

# --- Variáveis Globais ---
output_frame_display = None # O frame final a ser enviado para o stream
lock = threading.Lock() # Protege o acesso ao output_frame_display
app = Flask(__name__)

# --- NOVO: Variáveis Globais de Deteção ---
dms_monitor = None # Objeto que contém a lógica do Dlib
cam_thread = None # Objeto que contém a thread da câmara
detection_thread = None # Objeto que contém a thread de deteção
app_running = True # Flag para parar as threads

# --- NOVO: Página HTML com UI de Calibração ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SistemaDMS - Monitoramento e Calibração</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
               background-color: #1a1a1a; color: #eee; margin: 0; padding: 20px; 
               display: flex; flex-direction: column; align-items: center;
        }
        h1 { color: #eee; border-bottom: 2px solid #444; padding-bottom: 10px; }
        #container { display: flex; flex-wrap: wrap; justify-content: center; gap: 30px; width: 100%; max-width: 1200px; }
        #stream-container { flex-basis: 640px; flex-grow: 1; min-width: 320px; }
        #controls-container { flex-basis: 300px; flex-grow: 1; background-color: #2a2a2a; border-radius: 8px; padding: 20px; }
        img { border: 1px solid #555; background-color: #000; width: 100%; height: auto; border-radius: 8px; }
        h2 { margin-top: 0; }
        .control-group { margin-bottom: 20px; }
        .control-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .control-group input[type="range"] { width: 100%; }
        .control-group span { float: right; font-weight: bold; color: #3498db; }
        button { background-color: #3498db; color: white; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; width: 100%; }
        button:hover { background-color: #2980b9; }
        #status-box { background-color: #2a2a2a; border-radius: 8px; padding: 15px; margin-top: 20px; width: 100%; max-width: 640px; text-align: left; }
        #status-box div { font-size: 1.1em; }
        #status-box span { float: right; font-weight: bold; }
        #face-status.detected { color: #2ecc71; }
        #face-status.not-detected { color: #e74c3c; }
    </style>
</head>
<body>
    <h1>SistemaDMS - Monitoramento e Calibração</h1>
    <div id="status-box">
        <div>Fonte: <span id="source-name">{{ source_name }}</span></div>
        <div>Resolução: <span>{{ width }}x{{ height }}</span></div>
        <div>Status: <span id="face-status">Aguardando...</span></div>
    </div>

    <div id="container">
        <div id="stream-container">
            <img id="stream" src="{{ url_for('video_feed') }}" width="{{ width }}" height="{{ height }}">
        </div>

        <div id="controls-container">
            <h2>Calibração em Tempo Real</h2>
            
            <div class="control-group">
                <label for="ear_threshold">Limite EAR (Sonolência)</label>
                <input type="range" id="ear_threshold" min="0.10" max="0.40" step="0.01">
                <span id="ear_threshold_val">0.25</span>
            </div>
            
            <div class="control-group">
                <label for="ear_consec_frames">Frames (Sonolência)</label>
                <input type="range" id="ear_consec_frames" min="5" max="50" step="1">
                <span id="ear_consec_frames_val">15</span>
            </div>
            
            <div class="control-group">
                <label for="distraction_threshold_angle">Ângulo (Distração)</label>
                <input type="range" id="distraction_threshold_angle" min="15" max="60" step="1">
                <span id="distraction_threshold_angle_val">30</span>
            </div>
            
            <div class="control-group">
                <label for="distraction_consec_frames">Frames (Distração)</label>
                <input type="range" id="distraction_consec_frames" min="10" max="60" step="1">
                <span id="distraction_consec_frames_val">25</span>
            </div>
            
            <button id="saveButton">Guardar Configuração</button>
        </div>
    </div>

    <script>
        // --- Atualiza os valores dos sliders ---
        const sliders = document.querySelectorAll('input[type="range"]');
        sliders.forEach(slider => {
            const valEl = document.getElementById(slider.id + '_val');
            slider.addEventListener('input', () => {
                valEl.textContent = slider.value;
            });
        });

        // --- Carrega os valores atuais da API ---
        async function loadConfig() {
            try {
                const response = await fetch('/api/config');
                const config = await response.json();
                
                Object.keys(config).forEach(key => {
                    const slider = document.getElementById(key);
                    const valEl = document.getElementById(key + '_val');
                    if (slider) {
                        slider.value = config[key];
                        valEl.textContent = config[key];
                    }
                });
                console.log('Configuração carregada:', config);
            } catch (error) {
                console.error('Erro ao carregar configuração:', error);
            }
        }

        // --- Guarda os novos valores (POST para a API) ---
        async function saveConfig() {
            const config = {
                ear_threshold: document.getElementById('ear_threshold').value,
                ear_consec_frames: document.getElementById('ear_consec_frames').value,
                distraction_threshold_angle: document.getElementById('distraction_threshold_angle').value,
                distraction_consec_frames: document.getElementById('distraction_consec_frames').value
            };
            
            try {
                const response = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                if (response.ok) {
                    console.log('Configuração guardada:', config);
                    alert('Configuração atualizada com sucesso!');
                } else {
                    alert('Erro ao guardar configuração.');
                }
            } catch (error) {
                console.error('Erro ao guardar configuração:', error);
            }
        }
        
        document.getElementById('saveButton').addEventListener('click', saveConfig);
        window.addEventListener('load', loadConfig);

        // --- Lógica de Reconexão do Stream ---
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
"""

# --- Rota Principal ---
@app.route("/")
def index():
    """Rota principal que serve a página HTML."""
    source_desc = "Câmara USB (Índice 0)" if VIDEO_SOURCE == "0" else f"Fonte: {VIDEO_SOURCE}"
    
    # NOVO: Corrigido o bug do 'source'
    return render_template_string(
        HTML_TEMPLATE, 
        width=FRAME_WIDTH_DISPLAY, 
        height=FRAME_HEIGHT_DISPLAY,
        source_name=source_desc # Nome da variável alterado
    )

# --- Rota do Stream de Vídeo ---
@app.route("/video_feed")
def video_feed():
    """Rota que serve o stream de vídeo (MJPEG)."""
    
    def generate_frames():
        """Gera frames de vídeo para o stream HTTP."""
        global output_frame_display, lock
        
        # Cria um placeholder se a câmara estiver a demorar
        placeholder_frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
        cv2.putText(placeholder_frame, "Aguardando camera...", (30, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        (flag, encodedImage) = cv2.imencode(".jpg", placeholder_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        placeholder_bytes = bytearray(encodedImage)
        
        while app_running:
            frame_to_encode = None
            
            with lock:
                if output_frame_display is not None:
                    frame_to_encode = output_frame_display.copy()

            if frame_to_encode is None:
                # Envia o placeholder se o frame não estiver pronto
                frame_bytes = placeholder_bytes
                time.sleep(0.5) 
            else:
                (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if flag:
                    frame_bytes = bytearray(encodedImage)
                else:
                    logging.warning("Falha ao codificar frame para JPEG.")
                    frame_bytes = placeholder_bytes # Envia placeholder em caso de falha

            # Envia o frame no formato MJPEG
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                   frame_bytes + b'\r\n')
            
            # Controla o FPS do stream
            time.sleep(TARGET_FRAME_TIME) 

    # Retorna a resposta do stream
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- NOVO: Rotas da API de Configuração ---

@app.route("/api/config", methods=['GET'])
def get_config():
    """API (GET): Retorna os valores de configuração atuais do dms_monitor."""
    if dms_monitor:
        settings = dms_monitor.get_settings()
        return jsonify(settings)
    return jsonify({"error": "Monitor não inicializado"}), 500

@app.route("/api/config", methods=['POST'])
def set_config():
    """API (POST): Atualiza os valores de configuração no dms_monitor."""
    data = request.json
    if not data:
        return jsonify({"error": "Dados inválidos"}), 400
        
    if dms_monitor:
        try:
            dms_monitor.update_settings(
                ear_thresh=data.get('ear_threshold'),
                ear_frames=data.get('ear_consec_frames'),
                distraction_angle=data.get('distraction_threshold_angle'),
                distraction_frames=data.get('distraction_consec_frames')
            )
            return jsonify({"success": True, "message": "Configuração atualizada."})
        except Exception as e:
            logging.error(f"Erro ao atualizar configuração: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
            
    return jsonify({"error": "Monitor não inicializado"}), 500

# --- NOVO: Thread de Deteção ---

def detection_loop():
    """
    Função principal executada em background para processar vídeo.
    Lê da CameraThread e processa no DriverMonitor.
    """
    global output_frame_display, dms_monitor, cam_thread
    
    logging.info(f">>> Loop de deteção iniciado (Alvo: {TARGET_FPS} FPS).")
    
    frame_count = 0
    start_time = time.time()
    
    while app_running:
        frame_time_start = time.time()
        
        if not cam_thread or not cam_thread.connected:
            logging.warning("A aguardar ligação da câmara...")
            time.sleep(1.0)
            continue
            
        # 1. Obtém o frame da thread da câmara
        frame = cam_thread.get_frame() 
        
        if frame is None:
            logging.warning("Frame não recebido da câmara. A aguardar...")
            time.sleep(0.5)
            continue
            
        # 2. Converte para Preto e Branco (Gray)
        # NOVO: Corrigido o bug de digitação
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        # 3. Processa o frame no DMS Core
        if dms_monitor:
            processed_frame, status_data = dms_monitor.process_frame(frame, gray)
            
            # 4. Atualiza o frame de saída para o servidor web
            with lock:
                output_frame_display = processed_frame.copy()
        else:
            # Se o monitor não estiver pronto, apenas mostra o frame original
            with lock:
                output_frame_display = frame.copy()
        
        # 5. Lógica de controlo de FPS
        frame_time_end = time.time()
        processing_time = frame_time_end - frame_time_start
        
        sleep_time = TARGET_FRAME_TIME - processing_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # Loga apenas se estivermos significativamente lentos
            if abs(sleep_time) > (TARGET_FRAME_TIME * 0.1): 
                logging.warning(f"!!! LOOP LENTO. Processamento demorou {processing_time:.2f}s (Alvo era {TARGET_FRAME_TIME:.2f}s)")
        
        frame_count += 1
        if (frame_count % (TARGET_FPS * 10)) == 0: # Loga o FPS a cada 10 segundos
            avg_fps = frame_count / (time.time() - start_time)
            logging.info(f"FPS Médio (Deteção): {avg_fps:.2f} FPS")

    logging.info(">>> Loop de deteção terminado.")

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    logging.info(f">>> Serviço DMS (Refatorado) a iniciar...")
    
    frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)

    try:
        # 1. Inicializa o "Cérebro" (DMS Core)
        dms_monitor = DriverMonitor(frame_size=frame_size)
        
        # 2. Inicializa os "Olhos" (Thread da Câmara)
        logging.info(f">>> A iniciar thread da câmara (Fonte: {VIDEO_SOURCE})...")
        cam_thread = CameraThread(VIDEO_SOURCE, FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY, ROTATE_FRAME)
        cam_thread.start()

        # Espera pelo primeiro frame antes de continuar
        logging.info("A aguardar o primeiro frame da câmara...")
        time.sleep(2.0) # Dá tempo à câmara para arrancar
        first_frame = cam_thread.get_frame()
        
        if first_frame is None:
            logging.error("!!! ERRO FATAL: Não foi possível obter o primeiro frame da câmara.")
            logging.error("Verifique se a câmara está ligada em /dev/video0 e se o contentor tem permissão.")
            cam_thread.stop()
            sys.exit(1)
            
        logging.info(">>> Primeiro frame recebido!")
        
        # 3. Inicializa o "Processamento" (Thread de Deteção)
        logging.info(">>> A iniciar thread de deteção...")
        detection_thread = threading.Thread(target=detection_loop, daemon=True)
        detection_thread.start()

        # 4. Inicia o Servidor Web
        logging.info(f">>> A iniciar servidor Flask na porta 5000...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

    except Exception as e:
         logging.error(f"Erro fatal no arranque: {e}", exc_info=True)
         
    finally:
        # Limpa os recursos ao fechar
        app_running = False
        if cam_thread:
            cam_thread.stop()
        logging.info(">>> Servidor Flask terminado. A limpar recursos...")
        time.sleep(1.0)
        logging.info(">>> Aplicação terminada.")

