# Documentação: Script principal para o Serviço de IA do Projeto FluxoAI (Monolítico)
# Responsabilidade: Captura, Deteção (IA com YOLOv8), Tracking (Atitude Suspeita) e Streaming Web.

import cv2
import time
import os
import numpy as np
import threading 
import logging 
import sys
from flask import Flask, Response, render_template_string
import tflite_runtime.interpreter as tflite
import math # Para cálculo de distância

# OTIMIZAÇÃO: Habilita otimizações do OpenCV (SIMD/SSE, etc.)
cv2.setUseOptimized(True)

# --- Configuração do Logging ---
# Define o nível padrão (pode ser substituído pela variável de ambiente)
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - FluxoAI - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Reduzir logs do Werkzeug (servidor Flask)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# --- Configurações da Aplicação (AJUSTADAS CONFORME RECOMENDAÇÃO) ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640  # Largura para exibir no stream
FRAME_HEIGHT_DISPLAY = 480 # Altura para exibir no stream
MODEL_PATH = 'model.tflite' # IMPORTANTE: Este deve ser um modelo YOLOv8s-TFLite
LABELS_PATH = 'labels.txt'
DETECTION_THRESHOLD = 0.70 # OTIMIZAÇÃO: Aumentado para maior precisão
TARGET_LABEL = 'person'
PROCESS_EVERY_N_FRAMES = 3 # OTIMIZAÇÃO: Reduzido para melhor precisão temporal
JPEG_QUALITY = 35 # OTIMIZAÇÃO: Qualidade do JPEG mais baixa (35) para reduzir o trabalho de encode no RPi
LOITERING_THRESHOLD_SECONDS = 10 # Tempo (em segundos) parado para considerar atitude suspeita
LOITERING_MAX_DISTANCE = 50 # OTIMIZAÇÃO: Aumentado para tolerar pequenos movimentos
MIN_BOX_SIZE = 40 # OTIMIZAÇÃO: Aumentado para filtrar falsos positivos

# --- Variáveis Globais ---
output_frame_display = None
lock = threading.Lock() # Para acesso seguro ao output_frame por múltiplas threads
app = Flask(__name__)
labels = None
person_class_id = -1 # ID da classe 'person', será descoberto ao carregar as etiquetas
tracked_persons = {} # Dicionário para guardar pessoas seguidas {track_id: {'box': [], 'center': (x,y), 'start_time': t, 'is_loitering': False, 'last_seen': t}}
next_track_id = 0
last_valid_detections = [] # Guarda as últimas deteções válidas para desenhar nos frames saltados

# --- Funções Auxiliares ---

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
        if person_class_id != -1:
            logging.info(f">>> ID da classe '{TARGET_LABEL}': {person_class_id}")
        else:
            logging.error(f"!!! ERRO FATAL: A etiqueta alvo '{TARGET_LABEL}' não foi encontrada em {path}")
            sys.exit(1) # Termina se não encontrar a etiqueta 'person'
        return loaded_labels
    except FileNotFoundError:
        logging.error(f"!!! ERRO FATAL: Ficheiro de etiquetas não encontrado em {path}")
        sys.exit(1) # Termina se não encontrar o ficheiro

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

        logging.info(f">>> Modelo TFLite carregado: {MODEL_PATH}")
        logging.debug(f">>> Input Shape: {input_details[0]['shape']}, Input Type: {input_details[0]['dtype']}")
        logging.debug(f">>> Modelo espera input flutuante: {floating_model}")
        logging.debug(f">>> Detalhes dos Outputs: {output_details}")

        return interpreter, input_details, output_details, height, width, floating_model
    except Exception as e:
        logging.error(f"!!! ERRO FATAL ao carregar o modelo TFLite ({MODEL_PATH}): {e}", exc_info=True)
        sys.exit(1) # Termina se não conseguir carregar o modelo

# ---
# CORREÇÃO: Função detect_objects SUBSTITUÍDA pela recomendação do guia (YOLOv8)
# ---
def detect_objects(frame, interpreter, input_details, output_details, model_height, model_width, floating_model):
    """Executa a deteção de objetos num frame (Otimizado para YOLOv8 TFLite)."""
    
    # 1. Preparação da Imagem (Mesmo que antes)
    image_resized = cv2.resize(frame, (model_width, model_height))
    input_data = np.expand_dims(image_resized, axis=0)
 
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        input_data = np.uint8(input_data)
 
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # 2. Execução da Inferência
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
 
    # 3. Interpretação da Saída (YOLOv8 TFLite - 1 Tensor de Saída)
    # Saída esperada: [1, N, 6] -> [xmin, ymin, xmax, ymax, score, class_id]
    try:
        output_data = interpreter.get_tensor(output_details[0]['index'])[0] # [N, 6]
        
        # Cria listas vazias para armazenar os resultados no formato esperado pelo código original
        boxes = []
        classes = []
        scores = []
        
        # Itera sobre todas as deteções
        for det in output_data:
            score = det[4] # Score está na 5ª posição (índice 4)
            class_id = int(det[5]) # Class ID está na 6ª posição (índice 5)
            
            # Aplica os filtros de precisão e classe (person_class_id é global)
            if score >= DETECTION_THRESHOLD and class_id == person_class_id:
                # YOLOv8 retorna [xmin, ymin, xmax, ymax]
                # O código original (update_tracking) espera [ymin, xmin, ymax, xmax]
                
                # Coordenadas normalizadas
                xmin, ymin, xmax, ymax = det[0:4]
                
                # Adiciona ao formato esperado
                boxes.append([ymin, xmin, ymax, xmax])
                classes.append(class_id)
                scores.append(score)
        
        # Converte para arrays numpy, como o código original esperava
        boxes = np.array(boxes, dtype=np.float32)
        classes = np.array(classes, dtype=np.float32) 
        scores = np.array(scores, dtype=np.float32)
        
        logging.debug(f"YOLOv8 TFLite - Deteções válidas: {len(scores)}. Score Máximo: {scores[0]:.2f}" if len(scores) > 0 else "Nenhuma deteção válida.")
 
    except Exception as e:
        logging.error(f"!!! ERRO ao interpretar outputs do modelo YOLOv8. Erro: {e}", exc_info=True)
        return [], [], [], 0 # Retorna vazio se não conseguir interpretar
 
    return boxes, classes, scores, inference_time
# ---
# FIM DA SUBSTITUIÇÃO
# ---

def calculate_center(xmin, ymin, xmax, ymax):
    """Calcula o ponto central de uma caixa."""
    return int((xmin + xmax) / 2), int((ymin + ymax) / 2)

def update_tracking(boxes, classes, scores, frame_h, frame_w):
    """Associa deteções atuais a tracks existentes e atualiza o estado de vadiagem."""
    global tracked_persons, next_track_id, last_valid_detections
    
    current_detections = []
    current_time = time.time()
    matched_track_ids = set()
    new_last_valid_detections = []

    # 1. Tenta associar deteções atuais a tracks existentes
    for i in range(len(scores)):
        # FILTRO: Ignora deteções que não são 'person' ou estão abaixo do limiar
        # (Este filtro já foi aplicado na nova função detect_objects, mas mantemos por segurança)
        if scores[i] < DETECTION_THRESHOLD or int(classes[i]) != person_class_id:
            continue
            
        ymin, xmin, ymax, xmax = boxes[i] # Coordenadas normalizadas (0.0 a 1.0)
        
        # Converte coordenadas normalizadas para pixels no frame de DISPLAY
        xmin_disp = int(xmin * frame_w)
        xmax_disp = int(xmax * frame_w)
        ymin_disp = int(ymin * frame_h)
        ymax_disp = int(ymax * frame_h)

        # Garante coordenadas válidas
        xmin_disp = max(0, xmin_disp)
        ymin_disp = max(0, ymin_disp)
        xmax_disp = min(frame_w - 1, xmax_disp)
        ymax_disp = min(frame_h - 1, ymax_disp)

        # FILTRO: Ignora deteções minúsculas (falsos positivos)
        if (xmax_disp - xmin_disp) < MIN_BOX_SIZE or (ymax_disp - ymin_disp) < MIN_BOX_SIZE:
            continue # Ignora caixas inválidas

        center_x, center_y = calculate_center(xmin_disp, ymin_disp, xmax_disp, ymax_disp)
        current_detection = {'box': [ymin_disp, xmin_disp, ymax_disp, xmax_disp], 'center': (center_x, center_y), 'score': scores[i]}
        current_detections.append(current_detection)
        
        best_match_id = -1
        min_distance = float('inf')

        # Encontra o track mais próximo (se houver)
        for track_id, data in tracked_persons.items():
            distance = math.dist(data['center'], current_detection['center'])
            # Verifica se a distância é razoável e se o track ainda não foi associado
            if distance < LOITERING_MAX_DISTANCE * 2 and track_id not in matched_track_ids: # Um pouco mais flexível para associação
                if distance < min_distance:
                    min_distance = distance
                    best_match_id = track_id

        if best_match_id != -1:
            # Atualiza track existente
            track_id = best_match_id
            tracked_persons[track_id]['box'] = current_detection['box']
            tracked_persons[track_id]['last_seen'] = current_time
            matched_track_ids.add(best_match_id)

            # Verifica vadiagem (atitude suspeita)
            distance_moved = math.dist(tracked_persons[track_id]['center'], current_detection['center'])
            if distance_moved > LOITERING_MAX_DISTANCE:
                # Pessoa moveu-se, reinicia contador
                tracked_persons[track_id]['center'] = current_detection['center']
                tracked_persons[track_id]['start_time'] = current_time
                if tracked_persons[track_id]['is_loitering']:
                    tracked_persons[track_id]['is_loitering'] = False
                    logging.info(f"Pessoa ID {track_id} deixou de ter Atitude Suspeita (Moveu).")
            else:
                # Pessoa está parada, verifica tempo
                time_stopped = current_time - tracked_persons[track_id]['start_time']
                if time_stopped > LOITERING_THRESHOLD_SECONDS and not tracked_persons[track_id]['is_loitering']:
                    tracked_persons[track_id]['is_loitering'] = True
                    logging.warning(f"Pessoa ID {best_match_id} DETETADA com ATITUDE SUSPEITA (tempo: {time_stopped:.1f}s)")
            
            # Guarda para desenhar
            new_last_valid_detections.append({
                     'box': tracked_persons[best_match_id]['box'],
                     'score': current_detection['score'],
                     'is_loitering': tracked_persons[best_match_id]['is_loitering'],
                     'track_id': best_match_id
                 })

        else:
            # Cria novo track
            track_id = next_track_id
            tracked_persons[track_id] = {
                'box': current_detection['box'],
                'center': current_detection['center'],
                'start_time': current_time,
                'is_loitering': False,
                'last_seen': current_time
            }
             # Guarda para desenhar
            new_last_valid_detections.append({
                     'box': tracked_persons[track_id]['box'],
                     'score': current_detection['score'],
                     'is_loitering': False,
                     'track_id': track_id
                 })
            logging.info(f"Novo Track ID {track_id} criado.")
            next_track_id += 1

    # 2. Remove tracks antigos (que não foram vistos recentemente)
    ids_to_remove = [track_id for track_id, data in tracked_persons.items() if current_time - data['last_seen'] > 5] # Remove se não visto por 5 segundos
    for track_id in ids_to_remove:
        if tracked_persons[track_id]['is_loitering']:
             logging.info(f"Pessoa ID {track_id} (que tinha Atitude Suspeita) desapareceu.")
        del tracked_persons[track_id]
        logging.debug(f"Track ID {track_id} removido por inatividade.")

    # 3. Atualiza a lista global para desenhar nos frames saltados
    last_valid_detections = new_last_valid_detections


def draw_detections(frame_display, current_detections_drawn=None):
    """Desenha as caixas de deteção no frame de exibição."""
    # current_detections_drawn: lista das deteções feitas neste frame (se processado)
    # last_valid_detections: lista global das últimas deteções válidas
    
    detections_to_draw = current_detections_drawn if current_detections_drawn is not None else last_valid_detections
    loitering_count = 0
    
    for det in detections_to_draw:
        ymin, xmin, ymax, xmax = det['box']
        score = det['score']
        track_id = det['track_id']
        is_loitering = det.get('is_loitering', False) 

        color = (0, 0, 255) if is_loitering else (0, 255, 0) # Vermelho para Suspeita, Verde normal
        if is_loitering:
            loitering_count += 1
            
        label_text = f'ID:{track_id} {int(score*100)}%'
        if is_loitering:
            label_text += " (SUSPEITA)" # CORREÇÃO: Nova Terminologia

        # Desenha o retângulo
        cv2.rectangle(frame_display, (xmin, ymin), (xmax, ymax), color, 2)

        # Prepara o texto da etiqueta
        label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(ymin, label_size[1] + 10)

        # Desenha fundo e texto da etiqueta
        cv2.rectangle(frame_display, (xmin, label_ymin - label_size[1] - 10),
                      (xmin + label_size[0], label_ymin - base_line - 10), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame_display, label_text, (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return frame_display, len(detections_to_draw), loitering_count

# --- Thread de Captura e Deteção ---

def capture_and_detect_loop():
    """Função principal executada em background."""
    global output_frame_display, lock, labels, last_valid_detections

    logging.info(f">>> Serviço Monolítico FluxoAI a iniciar...")
    logging.info(f">>> Versão do OpenCV: {cv2.__version__}")

    interpreter, input_details, output_details, model_height, model_width, floating_model = initialize_model()
    labels = load_labels_and_find_person(LABELS_PATH)

    is_rtsp = VIDEO_SOURCE.startswith("rtsp://")
    source_description = f"stream de rede: {VIDEO_SOURCE}" if is_rtsp else f"câmara local no índice: {VIDEO_SOURCE}"
    video_source_arg = VIDEO_SOURCE if is_rtsp else int(VIDEO_SOURCE)

    logging.info(f">>> A tentar conectar a: {source_description}...")
    cap = cv2.VideoCapture(video_source_arg)
    time.sleep(2)

    if not cap.isOpened():
        logging.error(f"!!! ERRO FATAL: Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")
        sys.exit(1)

    logging.info(">>> Fonte de vídeo conectada com sucesso!")
    logging.info(f">>> A processar 1 em cada {PROCESS_EVERY_N_FRAMES} frames.")
    logging.info(">>> A iniciar loop de captura e deteção...")

    frame_count = 0
    last_process_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("!!! Frame não recebido. A verificar ligação...")
            if is_rtsp:
                logging.info(">>> A tentar reconectar ao stream RTSP...")
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(video_source_arg)
                if not cap.isOpened():
                    logging.error("!!! Falha ao reconectar. A terminar.")
                    break
                else:
                    logging.info(">>> Reconectado com sucesso!")
                    continue
            else:
                logging.error("!!! Falha ao ler frame da câmara local. A terminar.")
                break

        frame_count += 1
        current_time = time.time()
        
        # Redimensiona o frame para exibição ANTES de qualquer processamento pesado
        frame_display = cv2.resize(frame, (FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY))
        frame_h, frame_w, _ = frame_display.shape # 480x640

        detections_count = 0
        loitering_count = 0
        detections_drawn = None # Flag para saber se processou este frame

        # Processa apenas 1 em cada N frames
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            start_inference_time = time.time()
            
            # Executa a deteção (usa o frame_display, que já está em 640x480)
            boxes, classes, scores, inference_time = detect_objects(frame_display, interpreter, input_details, output_details, model_height, model_width, floating_model)
            
            # Atualiza o tracking e o estado de vadiagem
            # Passa as dimensões do frame_display (640x480) para a desnormalização
            update_tracking(boxes, classes, scores, frame_h, frame_w)
            
            # Guarda as deteções deste frame para desenhar
            detections_drawn = last_valid_detections # update_tracking atualiza esta lista global
            
            end_inference_time = time.time()
            total_process_time_this_frame = end_inference_time - start_inference_time
            fps = 1 / total_process_time_this_frame if total_process_time_this_frame > 0 else 0
            
            logging.debug(f">>> Frame {frame_count} PROCESSADO | Inferência: {inference_time:.3f}s | FPS Proc: {fps:.1f}")
            last_process_time = current_time # Atualiza o tempo do último processamento

        # Desenha as deteções (atuais ou as últimas válidas)
        frame_with_detections, detections_count, loitering_count = draw_detections(frame_display, detections_drawn)

        # Adiciona info extra no canto se for frame saltado
        if detections_drawn is None: # Se não processou este frame
             # Desenha as caixas antigas para persistência visual
             frame_with_detections, detections_count, loitering_count = draw_detections(frame_display, last_valid_detections)
             # Adiciona info de FPS
             info_text = f"P:{detections_count} S:{loitering_count} (FPS:{1/(current_time - last_process_time):.1f})" if last_process_time else "P:?"
             cv2.putText(frame_with_detections, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # Ciano
        
        # Log de INFO apenas quando processou
        if detections_drawn is not None:
            logging.info(f"Frame Processado. Pessoas: {detections_count}. Suspeita: {loitering_count}.")

        # Atualiza o frame de saída para o servidor web
        with lock:
            output_frame_display = frame_with_detections.copy()

    cap.release()
    logging.info(">>> Loop de captura terminado.")

# --- Servidor Web Flask ---

def generate_frames():
    """Gera frames de vídeo para o stream HTTP."""
    global output_frame_display, lock
    
    # Placeholder Frame (para exibição enquanto espera)
    placeholder_frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
    cv2.putText(placeholder_frame, "Aguardando video...", (30, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    (flag, encodedImage) = cv2.imencode(".jpg", placeholder_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    placeholder_bytes = bytearray(encodedImage) if flag else None
    
    while True:
        frame_to_encode = None
        
        # OTIMIZAÇÃO: Loop de stream rápido
        with lock:
            if output_frame_display is not None:
                frame_to_encode = output_frame_display.copy()

        if frame_to_encode is None:
            frame_bytes = placeholder_bytes
            time.sleep(0.5) # Espera antes de reenviar o placeholder
        else:
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if flag:
                frame_bytes = bytearray(encodedImage)
            else:
                logging.warning("Falha ao codificar frame para JPEG.")
                frame_bytes = placeholder_bytes # Envia placeholder em caso de falha

        if frame_bytes is not None:
             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Controla o FPS do stream (30 FPS) para garantir fluidez no navegador
        time.sleep(1/30) 

@app.route("/")
def index():
    """Rota principal que serve a página HTML."""
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FluxoAI - Deteção ao Vivo</title>
            <style>
                body { font-family: sans-serif; background-color: #222; color: #eee; margin: 0; padding: 20px; text-align: center;}
                h1 { color: #eee; }
                img { border: 1px solid #555; background-color: #000; max-width: 95%; height: auto; margin-top: 20px;}
            </style>
        </head>
        <body>
            <h1>FluxoAI - Deteção ao Vivo</h1>
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

logging.info(">>> Servidor Flask terminado.")

