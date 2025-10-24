# Documentação: Script principal para o Serviço de IA do Projeto FluxoAI (Monolítico)
# Responsabilidade: Captura, Deteção (IA), Tracking (Vadiagem) e Streaming Web.

import cv2
import time
import os
import numpy as np
import threading 
import logging 
import sys
from flask import Flask, Response, render_template_string
import tflite_runtime.interpreter as tflite
import json # Para simular a estrutura JSON de output
import math # Para cálculo de distância

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - Monolito - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640  # Largura para exibir no stream
FRAME_HEIGHT_DISPLAY = 480 # Altura para exibir no stream
MODEL_PATH = 'model.tflite'
LABELS_PATH = 'labels.txt'
DETECTION_THRESHOLD = 0.55 
TARGET_LABEL = 'person'
PROCESS_EVERY_N_FRAMES = 5 # Processa IA em 1 de cada 5 frames (para otimização de CPU)
JPEG_QUALITY = 50 # Qualidade do JPEG para o stream (0-100)

# Tracking e Vadiagem
LOITERING_THRESHOLD_SECONDS = 10 
LOITERING_MAX_DISTANCE = 30 # Distância máxima (em pixels) que o centro pode mover-se para ainda ser considerado parado
MIN_BOX_SIZE = 30 # Tamanho mínimo (em pixels) para a caixa de deteção (para filtrar falsos positivos)


# --- Variáveis Globais ---
output_frame_display = None
lock = threading.Lock()
app = Flask(__name__)
labels = None
person_class_id = -1 
# Tracking Data
tracked_persons = {} # {track_id: {'box': [], 'center': (x,y), 'start_time': t, 'is_loitering': False, 'last_seen': t}}
next_track_id = 0


# --- Funções do Modelo e IA ---

def load_labels_and_find_person(path):
    """Carrega as etiquetas e encontra o ID da classe 'person'."""
    global person_class_id
    try:
        loaded_labels = {}
        with open(path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                label_name = line.strip()
                loaded_labels[i] = label_name
                if label_name == TARGET_LABEL:
                    person_class_id = i
        logging.info(f">>> Etiquetas carregadas ({len(loaded_labels)}): {list(loaded_labels.values())[:5]}...")
        if person_class_id == -1:
            logging.error(f"!!! ERRO FATAL: A etiqueta alvo '{TARGET_LABEL}' não foi encontrada em {path}")
            sys.exit(1)
        return loaded_labels
    except FileNotFoundError:
        logging.error(f"!!! ERRO FATAL: Ficheiro de etiquetas não encontrado em {path}")
        sys.exit(1)

def initialize_model():
    """Carrega o modelo TFLite e aloca tensores."""
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        floating_model = (input_details[0]['dtype'] == np.float32)
        logging.info(">>> Modelo TFLite carregado com sucesso.")
        return interpreter, input_details, output_details, height, width, floating_model
    except Exception as e:
        logging.error(f"!!! ERRO FATAL ao carregar o modelo TFLite ({MODEL_PATH}): {e}", exc_info=True)
        sys.exit(1)

def detect_objects(frame, interpreter, input_details, output_details, model_height, model_width, floating_model):
    """Executa a deteção de objetos num frame."""
    image_resized = cv2.resize(frame, (model_width, model_height))
    input_data = np.expand_dims(image_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = np.uint8(input_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
    except IndexError as e:
        logging.error(f"!!! ERRO ao obter outputs do modelo. Erro: {e}", exc_info=True)
        return [], [], [], 0

    return boxes, classes, scores, inference_time


# --- Funções de Tracking e Desenho ---

def calculate_center(xmin, ymin, xmax, ymax):
    return int((xmin + xmax) / 2), int((ymin + ymax) / 2)

def update_tracking(boxes, scores, current_frame_h, current_frame_w):
    """Processa detecções, faz tracking e atualiza o estado de vadiagem."""
    global tracked_persons, next_track_id
    
    current_time = time.time()
    detections_for_draw = []
    matched_track_ids = set()

    for i in range(len(scores)):
        if scores[i] < DETECTION_THRESHOLD:
            continue

        # 1. Desnormalização (0.0 - 1.0) para Pixeis
        ymin, xmin, ymax, xmax = boxes[i]
        
        # Multiplica pelas dimensões do FRAME CRU (640x480)
        xmin_abs = int(xmin * current_frame_w)
        xmax_abs = int(xmax * current_frame_w)
        ymin_abs = int(ymin * current_frame_h)
        ymax_abs = int(ymax * current_frame_h)

        # Filtro: Ignora deteções minúsculas ou fora do frame
        if (xmax_abs - xmin_abs) < MIN_BOX_SIZE or (ymax_abs - ymin_abs) < MIN_BOX_SIZE:
            continue

        center_x, center_y = calculate_center(xmin_abs, ymin_abs, xmax_abs, ymax_abs)
        
        best_match_id = -1
        min_distance = float('inf')

        # Tenta associar a um track existente
        for track_id, data in tracked_persons.items():
            distance = math.dist(data['center'], (center_x, center_y))
            if distance < LOITERING_MAX_DISTANCE * 2 and track_id not in matched_track_ids:
                if distance < min_distance:
                    min_distance = distance
                    best_match_id = track_id

        if best_match_id != -1:
            track_id = best_match_id
            tracked_persons[track_id]['last_seen'] = current_time

            # Verifica vadiagem
            distance_moved = math.dist(tracked_persons[track_id]['center'], (center_x, center_y))
            
            if distance_moved > LOITERING_MAX_DISTANCE:
                tracked_persons[track_id]['center'] = (center_x, center_y)
                tracked_persons[track_id]['start_time'] = current_time
                if tracked_persons[track_id]['is_loitering']:
                    tracked_persons[track_id]['is_loitering'] = False
                    logging.info(f"Pessoa ID {track_id} deixou de ter Atitude Suspeita (Moveu).")
            else:
                time_stopped = current_time - tracked_persons[track_id]['start_time']
                if time_stopped > LOITERING_THRESHOLD_SECONDS and not tracked_persons[track_id]['is_loitering']:
                    tracked_persons[track_id]['is_loitering'] = True
                    logging.warning(f"Pessoa ID {track_id} DETETADA com ATITUDE SUSPEITA (tempo: {time_stopped:.1f}s)")
            
            detections_for_draw.append({'box': [ymin_abs, xmin_abs, ymax_abs, xmax_abs], 'score': scores[i], 'is_loitering': tracked_persons[track_id]['is_loitering'], 'track_id': track_id})
            matched_track_ids.add(best_match_id)

        else:
            # Cria novo track
            track_id = next_track_id
            tracked_persons[track_id] = {
                'box': [ymin_abs, xmin_abs, ymax_abs, xmax_abs],
                'center': (center_x, center_y),
                'start_time': current_time,
                'is_loitering': False,
                'last_seen': current_time
            }
            logging.info(f"Novo Track ID {track_id} criado.")
            next_track_id += 1
            detections_for_draw.append({'box': [ymin_abs, xmin_abs, ymax_abs, xmax_abs], 'score': scores[i], 'is_loitering': False, 'track_id': track_id})

    # 2. Remove tracks antigos
    ids_to_remove = [track_id for track_id, data in tracked_persons.items() if current_time - data['last_seen'] > 3]
    for track_id in ids_to_remove:
        if tracked_persons[track_id]['is_loitering']:
             logging.info(f"Pessoa ID {track_id} (que tinha Atitude Suspeita) desapareceu.")
        del tracked_persons[track_id]
        
    return detections_for_draw

def draw_final_frame(frame, detections_to_draw):
    """Desenha as caixas no frame final."""
    
    loitering_count = 0

    for det in detections_to_draw:
        ymin, xmin, ymax, xmax = det['box']
        score = det['score']
        is_loitering = det['is_loitering']
        track_id = det['track_id']

        color = (0, 0, 255) if is_loitering else (0, 255, 0) # Vermelho para Suspeita, Verde normal
        loitering_count += 1 if is_loitering else 0

        # Desenha o retângulo
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Prepara o texto da etiqueta
        label_text = f'ID:{track_id} {int(score*100)}%'
        if is_loitering:
             label_text += " (SUSPEITA)"

        # Desenha fundo e texto da etiqueta
        label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(ymin, label_size[1] + 10)

        cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                      (xmin + label_size[0], label_ymin - base_line - 10), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label_text, (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
    return frame, loitering_count


# --- Thread Principal de Captura e Deteção ---

def capture_and_detect_loop():
    """Função principal executada em background para capturar e processar vídeo."""
    global output_frame_display, lock, labels
    
    logging.info("--- Iniciando Serviço Monolítico FluxoAI ---")

    # 1. Inicializa o Modelo
    interpreter, input_details, output_details, model_height, model_width, floating_model = initialize_model()
    labels = load_labels_and_find_person(LABELS_PATH)
    
    # 2. Inicializa a Captura
    is_rtsp = VIDEO_SOURCE.startswith("rtsp://")
    source_description = f"stream de rede: {VIDEO_SOURCE}" if is_rtsp else f"câmara local no índice: {VIDEO_SOURCE}"
    video_source_arg = VIDEO_SOURCE if is_rtsp else int(VIDEO_SOURCE)

    logging.info(f">>> A tentar conectar a: {source_description}...")
    cap = cv2.VideoCapture(video_source_arg)
    time.sleep(2) # Espera a câmara inicializar

    if not cap.isOpened():
        logging.error(f"!!! ERRO FATAL: Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")
        sys.exit(1)

    logging.info(">>> Fonte de vídeo conectada com sucesso! Iniciando loop.")

    frame_count = 0
    last_process_time = time.time()
    
    # Variável para armazenar as últimas detecções válidas para desenho
    detections_for_draw = []

    while True:
        ret, frame = cap.read()
        if not ret:
            # Lógica de reconexão
            if is_rtsp:
                cap.release(); time.sleep(5); cap = cv2.VideoCapture(video_source_arg)
                if not cap.isOpened(): logging.error("Falha ao reconectar. Terminando."); break
            else:
                logging.error("Falha ao ler frame da câmara local. Terminando."); break
            continue

        frame_count += 1
        
        # Redimensiona o frame para exibição (Este é o tamanho final da imagem)
        frame_display = cv2.resize(frame, (FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY))
        frame_h, frame_w, _ = frame_display.shape # 480x640

        # --- FASE IA & TRACKING (Apenas 1 em cada N frames) ---
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            boxes, classes, scores, inference_time = detect_objects(frame_display, interpreter, input_details, output_details, model_height, model_width, floating_model)
            
            # O SSD MobileNet deteta a pessoa (classe 0)
            detections_ia = []
            for i in range(len(scores)):
                 if scores[i] >= DETECTION_THRESHOLD and int(classes[i]) == person_class_id:
                    # Desnormaliza as coordenadas diretamente para o tamanho de FRAME_DISPLAY (640x480)
                    ymin, xmin, ymax, xmax = boxes[i]
                    detections_ia.append({
                        'box_pixels': [int(ymin * frame_h), int(xmin * frame_w), int(ymax * frame_h), int(xmax * frame_w)],
                        'score': scores[i],
                        'label': TARGET_LABEL 
                    })

            # Aplica o tracking e a lógica de vadiagem
            detections_for_draw = update_tracking(detections_ia)
            
            logging.info(f"Frame Processado. Pessoas: {len(detections_ia)}. Inferência: {inference_time:.3f}s. Suspeita: {len([d for d in detections_for_draw if d['is_loitering']])}")
            last_process_time = time.time()

        # --- FASE DE STREAMING ---
        
        # Desenha as últimas detecções válidas (mesmo que o frame não tenha sido processado agora)
        frame_final, loitering_count = draw_final_frame(frame_display, detections_for_draw)

        # Atualiza o frame de saída para o servidor web (de forma segura)
        with lock:
            output_frame_display = frame_final.copy()
            
        # Pequena pausa para controlar o FPS
        time.sleep(1/120) # Tenta ter o menor delay possível (para deixar o Flask gerir o stream)
            

    cap.release()
    logging.info(">>> Loop de captura terminado.")


# --- Servidor Web Flask ---

def generate_frames():
    """Gera frames de vídeo para o stream HTTP."""
    global output_frame_display, lock
    # Placeholder Frame (para exibição enquanto espera)
    placeholder_frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, "Aguardando video...", (10, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    (flag, encodedImage) = cv2.imencode(".jpg", placeholder_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    placeholder_bytes = bytearray(encodedImage) if flag else None
    
    while True:
        frame_bytes = None
        
        with lock:
            if output_frame_display is not None:
                # Codifica o frame para JPEG
                (flag, encodedImage) = cv2.imencode(".jpg", output_frame_display, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if flag:
                    frame_bytes = bytearray(encodedImage)
        
        if frame_bytes is None:
             frame_bytes = placeholder_bytes
             time.sleep(0.5) # Espera antes de reenviar o placeholder

        if frame_bytes is not None:
             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Tenta enviar a 30 FPS
        time.sleep(1/30) 


@app.route("/")
def index():
    """Rota principal que serve a página HTML."""
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FluxoAI - Detecção Monolítica</title>
            <style>
                body { font-family: sans-serif; background-color: #222; color: #eee; margin: 0; padding: 20px; text-align: center;}
                h1 { color: #eee; }
                img { border: 1px solid #555; background-color: #000; max-width: 95%; height: auto; margin-top: 20px;}
            </style>
        </head>
        <body>
            <h1>FluxoAI - Detecção Monolítica</h1>
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
    # Inicia a thread de captura e deteção em background
    capture_thread = threading.Thread(target=capture_and_detect_loop)
    capture_thread.daemon = True
    capture_thread.start()

    # Inicia o servidor Flask
    logging.info(">>> A iniciar servidor Flask na porta 5000...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
         logging.error(f"Erro ao iniciar servidor Flask: {e}", exc_info=True)
         sys.exit(1)
