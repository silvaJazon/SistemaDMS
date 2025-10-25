# Documentação: Script principal para o SistemaDMS (Monolítico)
# Fase 2: Deteção de Sonolência (EAR) + Deteção de Distração (Pose da Cabeça)

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
import math # Para a matemática da Pose da Cabeça

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

# --- Configurações de Alerta ---
# Sonolência
EAR_THRESHOLD = 0.25 # Limite do Eye Aspect Ratio para considerar "fechado"
EAR_CONSEC_FRAMES = 15 # Número de frames consecutivos com olhos fechados para disparar o alarme

# Distração (Pose da Cabeça)
DISTRACTION_THRESHOLD_ANGLE = 30.0 # Ângulo (em graus) para considerar "distraído"
DISTRACTION_CONSEC_FRAMES = 25 # Número de frames consecutivos distraído para disparar o alarme

# Índices dos landmarks do Dlib
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

# Contadores de Alerta
drowsiness_counter = 0 # Contador para sonolência
distraction_counter = 0 # Contador para distração

# --- Funções Auxiliares ---

def eye_aspect_ratio(eye):
    """Calcula a distância euclidiana entre os pontos verticais e horizontais do olho."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def estimate_head_pose(shape, frame_size):
    """Estima a pose da cabeça (para onde o motorista está a olhar)."""
    
    # Pontos de referência faciais 3D (modelo genérico)
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Ponta do nariz (30)
                            (0.0, -330.0, -65.0),        # Queixo (8)
                            (-225.0, 170.0, -135.0),     # Canto do olho esquerdo (36)
                            (225.0, 170.0, -135.0),      # Canto do olho direito (45)
                            (-150.0, -150.0, -125.0),    # Canto da boca esquerdo (48)
                            (150.0, -150.0, -125.0)     # Canto da boca direito (54)
                        ])
    
    # Parâmetros da câmara (assumidos)
    focal_length = frame_size[1]
    center = (frame_size[1]/2, frame_size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
    
    dist_coeffs = np.zeros((4,1)) # Assumindo sem distorção de lente

    # Pontos 2D correspondentes do Dlib
    image_points = np.array([
                            shape[30],     # Ponta do nariz
                            shape[8],      # Queixo
                            shape[36],     # Canto do olho esquerdo
                            shape[45],     # Canto do olho direito
                            shape[48],     # Canto da boca esquerdo
                            shape[54]      # Canto da boca direito
                        ], dtype="double")
    
    try:
        # Resolve a pose da cabeça
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Converte o vetor de rotação em ângulos de Euler (pitch, yaw, roll)
        (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        # Ângulos em graus (Yaw é o mais importante para distração lateral)
        yaw = angles[1]
        pitch = angles[0]
        roll = angles[2]
        
        return yaw, pitch, roll
    except Exception as e:
        logging.debug(f"Erro ao calcular pose da cabeça: {e}")
        return 0, 0, 0 # Retorna 0 se falhar

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
    global output_frame_display, lock, drowsiness_counter, distraction_counter

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
    
    frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY)

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
        
        alarm_drowsy = False
        alarm_distraction = False

        # Loop sobre as faces detetadas (deve ser apenas 1, o motorista)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            # --- 1. Verificação de Sonolência (EAR) ---
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Desenha os contornos dos olhos
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame_display, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame_display, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EAR_THRESHOLD:
                drowsiness_counter += 1
                if drowsiness_counter >= EAR_CONSEC_FRAMES:
                    alarm_drowsy = True
                    logging.warning(f"DETEÇÃO DE SONOLÊNCIA (EAR: {ear:.2f})")
            else:
                drowsiness_counter = 0

            # --- 2. Verificação de Distração (Pose da Cabeça) ---
            yaw, pitch, roll = estimate_head_pose(shape, frame_size)
            
            # Verifica se está a olhar para os lados (Yaw) ou muito para baixo (Pitch)
            if abs(yaw) > DISTRACTION_THRESHOLD_ANGLE or pitch > DISTRACTION_THRESHOLD_ANGLE:
                distraction_counter += 1
                if distraction_counter >= DISTRACTION_CONSEC_FRAMES:
                    alarm_distraction = True
                    logging.warning(f"DETEÇÃO DE DISTRAÇÃO (Yaw: {yaw:.1f}, Pitch: {pitch:.1f})")
            else:
                distraction_counter = 0
                
            # Desenha informações de debug no frame
            cv2.putText(frame_display, f"EAR: {ear:.2f}", (FRAME_WIDTH_DISPLAY - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Yaw: {yaw:.1f}", (FRAME_WIDTH_DISPLAY - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Se nenhuma face for detetada, reinicia os contadores
        if not rects:
             drowsiness_counter = 0
             distraction_counter = 0

        # Desenha os Alertas Visuais Finais
        if alarm_drowsy:
             cv2.putText(frame_display, "ALERTA: SONOLENCIA!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if alarm_distraction:
             cv2.putText(frame_display, "ALERTA: DISTRACAO!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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

