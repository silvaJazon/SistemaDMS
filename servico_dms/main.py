# Documentação: Script principal para o SistemaDMS (Monolítico)
# Fase 1: Deteção de Sonolência (Eye Aspect Ratio - EAR)

import cv2
import time
import os
import numpy as np
import threading 
import logging 
import sys
from flask import Flask, Response, render_template_string
import dlib # Biblioteca para deteção de face e landmarks
from scipy.spatial import distance as dist # Para calcular a distância euclidiana

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640  
FRAME_HEIGHT_DISPLAY = 480 
MODEL_PATH = 'shape_predictor_68_face_landmarks.dat' # Modelo Dlib
JPEG_QUALITY = 50 

# --- Configurações de Sonolência ---
EAR_THRESHOLD = 0.25 # Limite do Eye Aspect Ratio para considerar "fechado"
EAR_CONSEC_FRAMES = 15 # Número de frames consecutivos com olhos fechados para disparar o alarme
EYE_AR_LEFT_START = 42
EYE_AR_LEFT_END = 48
EYE_AR_RIGHT_START = 36
EYE_AR_RIGHT_END = 42

# --- Variáveis Globais ---
output_frame_display = None
lock = threading.Lock()
app = Flask(__name__)
detector = None
predictor = None
(lStart, lEnd) = (EYE_AR_LEFT_START, EYE_AR_LEFT_END)
(rStart, rEnd) = (EYE_AR_RIGHT_START, EYE_AR_RIGHT_END)
alarm_counter = 0 # Contador de frames com olhos fechados
alarm_on = False # Se o alarme está ativo

# --- Funções Auxiliares ---

def eye_aspect_ratio(eye):
    """Calcula a distância euclidiana entre os pontos verticais e horizontais do olho."""
    # Pontos verticais
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Ponto horizontal
    C = dist.euclidean(eye[0], eye[3])
    # Cálculo do EAR
    ear = (A + B) / (2.0 * C)
    return ear

def initialize_model():
    """Carrega o detetor de face e o preditor de landmarks do Dlib."""
    global detector, predictor
    try:
        logging.info(">>> Carregando detetor de faces do Dlib...")
        detector = dlib.get_frontal_face_detector()
        logging.info(">>> Carregando preditor de landmarks faciais (modelo)...")
        predictor = dlib.shape_predictor(MODEL_PATH)
        logging.info(">>> Modelos Dlib carregados com sucesso.")
    except Exception as e:
        logging.error(f"!!! ERRO FATAL ao carregar modelos Dlib ({MODEL_PATH}): {e}", exc_info=True)
        sys.exit(1)

def shape_to_np(shape, dtype="int"):
    """Converte o objeto de landmarks do Dlib para um array NumPy."""
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# --- Thread de Captura e Deteção ---

def capture_and_detect_loop():
    """Função principal executada em background para capturar e processar vídeo."""
    global output_frame_display, lock, alarm_counter, alarm_on

    logging.info(f">>> Serviço Monolítico DMS a iniciar...")
    
    initialize_model()

    is_rtsp = VIDEO_SOURCE.startswith("rtsp://")
    source_description = f"stream de rede: {VIDEO_SOURCE}" if is_rtsp else f"câmara local no índice: {VIDEO_SOURCE}"
    video_source_arg = VIDEO_SOURCE if is_rtsp else int(VIDEO_SOURCE)

    logging.info(f">>> A tentar conectar a: {source_description}...")
    cap = cv2.VideoCapture(video_source_arg)
    time.sleep(2.0)

    if not cap.isOpened():
        logging.error(f"!!! ERRO FATAL: Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")
        sys.exit(1)

    logging.info(">>> Fonte de vídeo conectada com sucesso! Iniciando loop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("!!! Frame não recebido. A verificar ligação...")
            if is_rtsp:
                # Lógica de reconexão
                cap.release(); time.sleep(5); cap = cv2.VideoCapture(video_source_arg)
                if not cap.isOpened(): logging.error("Falha ao reconectar. Terminando."); break
                else: logging.info("Reconectado com sucesso!"); continue
            else:
                logging.error("Falha ao ler frame da câmara local. Terminando."); break
        
        # Redimensiona o frame para exibição
        frame_display = cv2.resize(frame, (FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY))
        gray = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY) # Dlib usa grayscale
        
        # Deteção de Faces
        rects = detector(gray, 0)
        
        alarm_on = False # Reseta o alarme por frame

        # Loop sobre as faces detetadas (deve ser apenas 1, o motorista)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            # Extrai coordenadas dos olhos
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            # Calcula o EAR para ambos os olhos
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Média do EAR
            ear = (leftEAR + rightEAR) / 2.0

            # Desenha os contornos dos olhos
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame_display, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame_display, [rightEyeHull], -1, (0, 255, 0), 1)

            # Verifica se o EAR está abaixo do limiar (sonolência)
            if ear < EAR_THRESHOLD:
                alarm_counter += 1
                logging.debug(f"Contador de sonolência: {alarm_counter}")
                
                # Se os olhos estiverem fechados por tempo suficiente, soa o alarme
                if alarm_counter >= EAR_CONSEC_FRAMES:
                    alarm_on = True
                    cv2.putText(frame_display, "ALERTA: SONOLENCIA!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    logging.warning(f"DETEÇÃO DE SONOLÊNCIA (EAR: {ear:.2f})")
            
            else:
                # Se os olhos abriram, reinicia o contador
                alarm_counter = 0

            # Desenha o EAR no frame para debug
            cv2.putText(frame_display, f"EAR: {ear:.2f}", (FRAME_WIDTH_DISPLAY - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Se nenhuma face for detetada, reinicia o contador
        if not rects:
             alarm_counter = 0

        # Atualiza o frame de saída para o servidor web
        with lock:
            output_frame_display = frame_display.copy()

    cap.release()
    logging.info(">>> Loop de captura terminado.")

# --- Servidor Web Flask ---

def generate_frames():
    """Gera frames de vídeo para o stream HTTP."""
    global output_frame_display, lock
    
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
            time.sleep(0.5) 
        else:
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if flag:
                frame_bytes = bytearray(encodedImage)
            else:
                logging.warning("Falha ao codificar frame para JPEG.")
                frame_bytes = placeholder_bytes 

        if frame_bytes is not None:
             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(1/30) # Stream a 30 FPS

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

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    capture_thread = threading.Thread(target=capture_and_detect_loop)
    capture_thread.daemon = True
    capture_thread.start()

    logging.info(f">>> A iniciar servidor Flask na porta 5000...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
         logging.error(f"Erro ao iniciar servidor Flask: {e}", exc_info=True)
         sys.exit(1)

logging.info(">>> Servidor Flask terminado.")
