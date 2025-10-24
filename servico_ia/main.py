# Documentação: Script principal para o Serviço de IA do Projeto FluxoAI
# Fase 4: Deteção de Vadiagem

import cv2
import time
import os
import numpy as np
import threading # Para executar a captura em paralelo com o servidor web
import logging # Para logging
import sys # Para sys.exit()
from flask import Flask, Response, render_template_string
import tflite_runtime.interpreter as tflite # Importa o motor TFLite

# --- Configuração do Logging ---
# Define o nível padrão (pode ser substituído pela variável de ambiente)
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Reduzir logs do Werkzeug (servidor Flask)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640  # Largura para exibir no stream
FRAME_HEIGHT_DISPLAY = 480 # Altura para exibir no stream
MODEL_PATH = 'model.tflite'
LABELS_PATH = 'labels.txt'
DETECTION_THRESHOLD = 0.55 # Limite de confiança para considerar uma deteção válida
TARGET_LABEL = 'person'
PROCESS_EVERY_N_FRAMES = 5 # Processa 1 em cada N frames
JPEG_QUALITY = 50 # Qualidade do JPEG para o stream (0-100)
LOITERING_THRESHOLD_SECONDS = 10 # Tempo (em segundos) parado para considerar vadiagem (ajuste conforme necessário)
LOITERING_MAX_DISTANCE = 30 # Distância máxima (em pixels) que o centro pode mover-se para ainda ser considerado parado

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

def detect_objects(frame, interpreter, input_details, output_details, model_height, model_width, floating_model):
    """Executa a deteção de objetos num frame."""
    # Redimensiona e prepara a imagem de entrada para o modelo
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

    # Ordem padrão esperada para SSD MobileNet TFLite (pode precisar de ajuste)
    # 0: locations (boxes), 1: classes, 2: scores, 3: num_detections
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        # num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0]) # Opcional
        logging.debug(f"Outputs obtidos na ordem padrão. Scores[0]: {scores[0]:.2f}, Classe[0]: {int(classes[0])}")
    except IndexError:
        # Tenta ordem alternativa ou menos outputs
        try:
             # Ordem comum diferente: 0:scores, 1:boxes, 2:count, 3:classes (ou similar)
             # Ajuste os índices conforme os 'output_details' impressos no log DEBUG
            scores = interpreter.get_tensor(output_details[0]['index'])[0]
            boxes = interpreter.get_tensor(output_details[1]['index'])[0]
            classes = interpreter.get_tensor(output_details[3]['index'])[0] # Exemplo, ajuste se necessário
            logging.warning("Aviso: Ordem alternativa de outputs do modelo TFLite detetada/utilizada.")
        except IndexError as e:
            logging.error(f"!!! ERRO ao obter outputs do modelo. Detalhes: {output_details}. Erro: {e}", exc_info=True)
            return [], [], [], 0 # Retorna vazio se não conseguir interpretar

    return boxes, classes, scores, inference_time

def calculate_center(xmin, ymin, xmax, ymax):
    """Calcula o ponto central de uma caixa."""
    return int((xmin + xmax) / 2), int((ymin + ymax) / 2)

def update_tracking(boxes, classes, scores):
    """Associa deteções atuais a tracks existentes e atualiza o estado de vadiagem."""
    global tracked_persons, next_track_id, last_valid_detections
    
    current_detections = []
    current_time = time.time()
    matched_track_ids = set()
    new_last_valid_detections = []

    # 1. Tenta associar deteções atuais a tracks existentes
    for i in range(len(scores)):
        if scores[i] > DETECTION_THRESHOLD and int(classes[i]) == person_class_id:
            ymin, xmin, ymax, xmax = boxes[i]
            # Converte coordenadas normalizadas para pixels no frame de DISPLAY
            xmin_disp = int(xmin * FRAME_WIDTH_DISPLAY)
            xmax_disp = int(xmax * FRAME_WIDTH_DISPLAY)
            ymin_disp = int(ymin * FRAME_HEIGHT_DISPLAY)
            ymax_disp = int(ymax * FRAME_HEIGHT_DISPLAY)

            # Garante coordenadas válidas
            xmin_disp = max(0, xmin_disp)
            ymin_disp = max(0, ymin_disp)
            xmax_disp = min(FRAME_WIDTH_DISPLAY - 1, xmax_disp)
            ymax_disp = min(FRAME_HEIGHT_DISPLAY - 1, ymax_disp)

            if xmax_disp <= xmin_disp or ymax_disp <= ymin_disp:
                continue # Ignora caixas inválidas

            center_x, center_y = calculate_center(xmin_disp, ymin_disp, xmax_disp, ymax_disp)
            current_detection = {'box': [ymin_disp, xmin_disp, ymax_disp, xmax_disp], 'center': (center_x, center_y), 'score': scores[i]}
            current_detections.append(current_detection)
            
            best_match_id = -1
            min_distance = float('inf')

            # Encontra o track mais próximo (se houver)
            for track_id, data in tracked_persons.items():
                distance = np.linalg.norm(np.array(data['center']) - np.array(current_detection['center']))
                # Verifica se a distância é razoável e se o track ainda não foi associado
                if distance < LOITERING_MAX_DISTANCE * 2 and track_id not in matched_track_ids: # Um pouco mais flexível para associação
                    if distance < min_distance:
                        min_distance = distance
                        best_match_id = track_id

            if best_match_id != -1:
                # Atualiza track existente
                tracked_persons[best_match_id]['box'] = current_detection['box']
                tracked_persons[best_match_id]['last_seen'] = current_time
                matched_track_ids.add(best_match_id)

                # Verifica vadiagem
                distance_moved = np.linalg.norm(np.array(tracked_persons[best_match_id]['center']) - np.array(current_detection['center']))
                if distance_moved > LOITERING_MAX_DISTANCE:
                    # Pessoa moveu-se, reinicia contador de vadiagem
                    tracked_persons[best_match_id]['center'] = current_detection['center']
                    tracked_persons[best_match_id]['start_time'] = current_time
                    tracked_persons[best_match_id]['is_loitering'] = False
                else:
                    # Pessoa está parada, verifica tempo
                    time_stopped = current_time - tracked_persons[best_match_id]['start_time']
                    if time_stopped > LOITERING_THRESHOLD_SECONDS:
                        tracked_persons[best_match_id]['is_loitering'] = True
                        logging.debug(f"Pessoa ID {best_match_id} marcada como vadiando (tempo: {time_stopped:.1f}s)")
                
                # Guarda para desenhar
                new_last_valid_detections.append({
                     'box': tracked_persons[best_match_id]['box'],
                     'score': current_detection['score'],
                     'is_loitering': tracked_persons[best_match_id]['is_loitering']
                 })

            else:
                # Cria novo track
                tracked_persons[next_track_id] = {
                    'box': current_detection['box'],
                    'center': current_detection['center'],
                    'start_time': current_time,
                    'is_loitering': False,
                    'last_seen': current_time
                }
                # Guarda para desenhar
                new_last_valid_detections.append({
                     'box': tracked_persons[next_track_id]['box'],
                     'score': current_detection['score'],
                     'is_loitering': False
                 })
                next_track_id += 1

    # 2. Remove tracks antigos (que não foram vistos recentemente)
    ids_to_remove = [track_id for track_id, data in tracked_persons.items() if current_time - data['last_seen'] > 5] # Remove se não visto por 5 segundos
    for track_id in ids_to_remove:
        del tracked_persons[track_id]
        logging.debug(f"Track ID {track_id} removido por inatividade.")

    # 3. Atualiza a lista global para desenhar nos frames saltados
    last_valid_detections = new_last_valid_detections


def draw_detections(frame_display, current_detections_drawn=None):
    """Desenha as caixas de deteção no frame de exibição."""
    # current_detections_drawn: lista das deteções feitas neste frame (se processado)
    # last_valid_detections: lista global das últimas deteções válidas
    
    detections_to_draw = current_detections_drawn if current_detections_drawn is not None else last_valid_detections
    
    for det in detections_to_draw:
        ymin, xmin, ymax, xmax = det['box']
        score = det['score']
        is_loitering = det.get('is_loitering', False) # Default é False se não for deste frame

        color = (0, 0, 255) if is_loitering else (0, 255, 0) # Vermelho para vadiagem, Verde normal
        label_text = f'{TARGET_LABEL}: {int(score*100)}%'
        if is_loitering:
             label_text += " (Vadiando)"

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

    return frame_display, len(detections_to_draw)

# --- Thread de Captura e Deteção ---

def capture_and_detect():
    """Função principal executada em background."""
    global output_frame_display, lock, labels

    logging.info(">>> Serviço de IA do FluxoAI a iniciar (Fase 4: Deteção de Vadiagem)...")
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

        detections_count = 0
        detections_drawn = None # Para passar para draw_detections

        # Processa apenas 1 em cada N frames
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            start_inference_time = time.time()
            
            # Executa a deteção (usa o frame_display já redimensionado se for compatível, senão redimensiona de novo)
            # Nota: O ideal é que FRAME_WIDTH/HEIGHT_DISPLAY seja igual a model_width/height se possível
            boxes, classes, scores, inference_time = detect_objects(frame_display, interpreter, input_details, output_details, model_height, model_width, floating_model)
            
            # Atualiza o tracking e o estado de vadiagem
            update_tracking(boxes, classes, scores)
            
            # Guarda as deteções deste frame para desenhar
            detections_drawn = last_valid_detections # update_tracking atualiza esta lista global
            detections_count = len([d for d in detections_drawn if d.get('is_loitering', False) == False]) # Conta apenas os não vadiando? Ou todos? Ajustar se necessário
            
            end_inference_time = time.time()
            total_process_time_this_frame = end_inference_time - start_inference_time
            fps = 1 / total_process_time_this_frame if total_process_time_this_frame > 0 else 0
            
            logging.debug(f">>> Frame {frame_count} PROCESSADO | Pessoas: {detections_count} | Inferência: {inference_time:.3f}s | FPS Proc: {fps:.1f}")
            last_process_time = current_time # Atualiza o tempo do último processamento

        # Desenha as deteções (atuais ou as últimas válidas)
        frame_with_detections, _ = draw_detections(frame_display, detections_drawn)

        # Adiciona info extra no canto se for frame saltado
        if detections_drawn is None: # Se não processou este frame
             info_text = f"P:{len(last_valid_detections)} (FPS:{1/(current_time - last_process_time):.1f})" if last_process_time else "P:?"
             cv2.putText(frame_with_detections, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # Ciano


        # Atualiza o frame de saída para o servidor web
        with lock:
            output_frame_display = frame_with_detections.copy()

    cap.release()
    logging.info(">>> Loop de captura terminado.")

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
            black_frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
            cv2.putText(black_frame, "Aguardando video...", (30, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            (flag, encodedImage) = cv2.imencode(".jpg", black_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if flag:
                frame_bytes = bytearray(encodedImage)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)
        else:
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if flag:
                frame_bytes = bytearray(encodedImage)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Controla o FPS do stream independentemente do processamento
        time.sleep(1/30) # Tenta enviar ~30 FPS para o navegador

@app.route("/")
def index():
    """Rota principal que serve a página HTML."""
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FluxoAI - Deteção ao Vivo</title>
            <style>
                body { font-family: sans-serif; background-color: #f0f0f0; margin: 0; padding: 20px; text-align: center;}
                h1 { color: #333; }
                img { border: 1px solid #ccc; background-color: #fff; max-width: 90%; height: auto; margin-top: 20px;}
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
    capture_thread = threading.Thread(target=capture_and_detect)
    capture_thread.daemon = True
    capture_thread.start()

    logging.info(">>> A iniciar servidor Flask na porta 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

logging.info(">>> Servidor Flask terminado.")

