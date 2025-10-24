# Documentação: Script principal para o Serviço de IA do Projeto FluxoAI
# Fase 3: Deteção de Pessoas (Logging + Novos Parâmetros)

import cv2
import time
import os
import numpy as np
import threading
from flask import Flask, Response, render_template_string
import tflite_runtime.interpreter as tflite
import sys
import logging # Importa o módulo de logging

# --- Configuração do Logging ---
# Define o nível de log padrão (INFO). DEBUG mostraria mais detalhes.
log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
# Desativa logs muito verbosos do Werkzeug (servidor Flask de desenvolvimento)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Configurações da Aplicação ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640  # Largura para exibir no stream
FRAME_HEIGHT_DISPLAY = 480 # Altura para exibir no stream
MODEL_PATH = 'model.tflite'
LABELS_PATH = 'labels.txt'
DETECTION_THRESHOLD = 0.55 # Limite de confiança para considerar uma deteção válida
TARGET_LABEL = 'person'
PROCESS_EVERY_N_FRAMES = 5 # Processa 1 em cada 5 frames
JPEG_QUALITY = 50 # Qualidade do JPEG para o stream (0-100)

# --- Variáveis Globais ---
output_frame_display = None
last_inference_time = 0
last_detections_count = 0
last_valid_boxes = []
last_valid_classes = []
last_valid_scores = []
lock = threading.Lock()
app = Flask(__name__)

# --- Funções Auxiliares ---

def load_labels(path):
    """Carrega as etiquetas."""
    labels = {}
    try:
        with open(path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                labels[i] = line.strip()
        logging.info(f"Etiquetas carregadas ({len(labels)}): {list(labels.values())[:5]}...")
        if TARGET_LABEL not in labels.values():
            logging.warning(f"A etiqueta alvo '{TARGET_LABEL}' não foi encontrada nas etiquetas!")
        person_id = -1
        for key, value in labels.items():
            if value == TARGET_LABEL:
                person_id = key
                break
        logging.info(f"ID da classe '{TARGET_LABEL}': {person_id}")
        return labels
    except FileNotFoundError:
        logging.error(f"Ficheiro de etiquetas não encontrado em {path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Erro ao carregar etiquetas: {e}")
        sys.exit(1)


def initialize_model():
    """Carrega o modelo TFLite."""
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        floating_model = (input_details[0]['dtype'] == np.float32)

        logging.info(f"Modelo TFLite carregado: {MODEL_PATH}")
        logging.debug(f"Input Shape: {input_details[0]['shape']}, Input Type: {input_details[0]['dtype']}")
        logging.debug(f"Modelo espera input flutuante: {floating_model}")
        logging.debug(f"Detalhes dos Outputs: {output_details}")

        return interpreter, input_details, output_details, height, width, floating_model
    except Exception as e:
        logging.error(f"Erro fatal ao carregar o modelo TFLite ({MODEL_PATH}): {e}")
        sys.exit(1)


def detect_objects(frame_model_input, interpreter, input_details, output_details, floating_model):
    """Executa a deteção de objetos."""
    input_data = np.expand_dims(frame_model_input, axis=0)
    if floating_model: input_data = (np.float32(input_data) - 127.5) / 127.5
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time(); interpreter.invoke(); inference_time = time.time() - start_time
    try:
        # A ordem dos outputs foi confirmada nos logs anteriores para este modelo específico
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        # num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0]) # Não usado diretamente aqui
    except IndexError as e:
        logging.error(f"Erro crítico ao obter outputs do modelo: {e}"); return [], [], [], 0
    return boxes, classes, scores, inference_time

def draw_single_detection(frame_display, box, class_id, score, labels, color=(0, 255, 0)):
    """Desenha UMA caixa de deteção no frame."""
    display_height, display_width, _ = frame_display.shape
    label = labels.get(class_id, f'ID:{class_id}')

    if label == TARGET_LABEL: # Só processa se for 'person'
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * display_width); xmax = int(xmax * display_width)
        ymin = int(ymin * display_height); ymax = int(ymax * display_height)
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(display_width - 1, xmax); ymax = min(display_height - 1, ymax)

        if xmax > xmin and ymax > ymin:
            cv2.rectangle(frame_display, (xmin, ymin), (xmax, ymax), color, 2)
            label_text = f'{label}: {int(score*100)}%'
            label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(frame_display, (xmin, label_ymin - label_size[1] - 10),
                          (xmin + label_size[0], label_ymin - base_line - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame_display, label_text, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            return True # Indica que uma pessoa foi desenhada
    return False


def draw_new_detections_and_update_last(frame_display, boxes, classes, scores, labels):
    """Processa novas deteções: desenha-as e atualiza a lista de últimas válidas."""
    global last_valid_boxes, last_valid_classes, last_valid_scores
    detections_count = 0
    current_valid_boxes = []
    current_valid_classes = []
    current_valid_scores = []

    for i in range(len(scores)):
        if scores[i] > DETECTION_THRESHOLD:
            class_id = int(classes[i])
            if labels.get(class_id) == TARGET_LABEL: # Verifica se é 'person' ANTES de desenhar
                # Desenha e verifica se foi bem-sucedido (caixa válida)
                if draw_single_detection(frame_display, boxes[i], class_id, scores[i], labels):
                    detections_count += 1
                    current_valid_boxes.append(boxes[i])
                    current_valid_classes.append(class_id)
                    current_valid_scores.append(scores[i])
            # Debug: Mostra outras deteções acima do limiar (opcional, comente se não precisar)
            # else:
            #    logging.debug(f"--- DETECTADO (Score > {DETECTION_THRESHOLD}): Índice: {i}, Score: {scores[i]:.2f}, Classe ID: {class_id}, Etiqueta: '{labels.get(class_id)}'")

    # Atualiza as listas globais APENAS com as deteções de 'person'
    last_valid_boxes = current_valid_boxes
    last_valid_classes = current_valid_classes
    last_valid_scores = current_valid_scores

    return frame_display, detections_count

def draw_last_valid_detections(frame_display, labels):
    """Redesenha as últimas caixas de 'person' válidas (sempre verde)."""
    global last_valid_boxes, last_valid_classes, last_valid_scores
    if last_valid_boxes:
        for i in range(len(last_valid_scores)):
             draw_single_detection(frame_display, last_valid_boxes[i], last_valid_classes[i], last_valid_scores[i], labels)
    return frame_display


# --- Thread de Captura e Deteção ---

def capture_and_detect():
    """Função principal: captura, processa (com saltos) e atualiza frame para stream."""
    global output_frame_display, lock, last_inference_time, last_detections_count

    logging.info("Serviço de IA do FluxoAI a iniciar (Fase 3: Deteção de Pessoas)...")
    logging.info(f"Versão do OpenCV: {cv2.__version__}")

    interpreter, input_details, output_details, model_height, model_width, floating_model = initialize_model()
    labels = load_labels(LABELS_PATH)

    is_rtsp = VIDEO_SOURCE.startswith("rtsp://")
    source_description = f"stream de rede: {VIDEO_SOURCE}" if is_rtsp else f"câmara local no índice: {VIDEO_SOURCE}"
    video_source_arg = VIDEO_SOURCE if is_rtsp else int(VIDEO_SOURCE)

    logging.info(f"A tentar conectar a: {source_description}...")
    cap = cv2.VideoCapture(video_source_arg)
    time.sleep(2)

    if not cap.isOpened(): logging.error(f"ERRO FATAL: Não foi possível abrir vídeo: {VIDEO_SOURCE}"); sys.exit(1)

    logging.info("Fonte de vídeo conectada!"); logging.info(f"Processando 1 em {PROCESS_EVERY_N_FRAMES} frames."); logging.info("Loop iniciado...")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Frame não recebido. Tentando reconectar..."); cap.release(); time.sleep(5)
            cap = cv2.VideoCapture(video_source_arg)
            if not cap.isOpened(): logging.error("Falha ao reconectar. Terminando thread."); break
            else: logging.info("Reconectado com sucesso!"); continue

        frame_count += 1
        frame_display = cv2.resize(frame, (FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY))

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            start_process_time = time.time()
            frame_model_input = cv2.resize(frame, (model_width, model_height))

            boxes, classes, scores, inference_time = detect_objects(
                frame_model_input, interpreter, input_details, output_details, floating_model
            )

            frame_display_processed, detections_count = draw_new_detections_and_update_last(
                frame_display, boxes, classes, scores, labels
            )

            last_inference_time = inference_time
            last_detections_count = detections_count
            end_process_time = time.time(); process_time = end_process_time - start_process_time
            fps = 1 / process_time if process_time > 0 else 0

            with lock: output_frame_display = frame_display_processed.copy()

            res_h, res_w, _ = output_frame_display.shape
            # Log INFO apenas quando processa um frame
            logging.debug(f"Frame {frame_count} PROCESSADO | Res: {res_w}x{res_h} | Pessoas: {detections_count} | Inferência: {inference_time:.3f}s | FPS Proc: {fps:.1f}")

        else:
            # Frame NÃO processado pela IA
            frame_display_with_persist = draw_last_valid_detections(frame_display, labels)
            info_text = f"P: {last_detections_count} ({last_inference_time*1000:.0f}ms @ F{frame_count - (frame_count % PROCESS_EVERY_N_FRAMES)})"
            cv2.putText(frame_display_with_persist, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            with lock: output_frame_display = frame_display_with_persist.copy()
            # Log DEBUG para frames saltados (não aparecerá por padrão)
            logging.debug(f"Frame {frame_count} SALTADO | Exibindo {last_detections_count} caixas anteriores.")


    cap.release()
    logging.info("Loop de captura terminado.")

# --- Servidor Web Flask ---

def generate_frames():
    """Gera frames para o stream HTTP."""
    global output_frame_display, lock
    while True:
        frame_to_encode = None; frame_bytes = None
        with lock:
            if output_frame_display is not None: frame_to_encode = output_frame_display.copy()

        if frame_to_encode is None:
            placeholder = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Aguardando...", (30, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            (flag, encodedImage) = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if flag: frame_bytes = bytearray(encodedImage)
            time.sleep(0.5)
        else:
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if flag: frame_bytes = bytearray(encodedImage)

        if frame_bytes is not None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            logging.warning("Falha ao codificar frame para stream.") # Adicionado log

        time.sleep(0.01)

@app.route("/")
def index():
    """Serve a página HTML."""
    return render_template_string("""
        <!DOCTYPE html><html><head><title>FluxoAI - Deteção ao Vivo</title>
        <style>body{font-family:sans-serif;background-color:#f0f0f0;margin:0;padding:20px;text-align:center;} h1{color:#333;} img{border:1px solid #ccc;background-color:#fff;max-width:90%;height:auto;margin-top:20px;}</style>
        </head><body><h1>FluxoAI - Deteção ao Vivo</h1>
        <img id="stream" src="{{ url_for('video_feed') }}" width="{{ width }}" height="{{ height }}">
        <script>var stream=document.getElementById("stream");stream.onerror=function(){console.log("Erro stream, recarregando...");setTimeout(function(){stream.src="{{ url_for('video_feed') }}?"+new Date().getTime();},5000);};</script>
        </body></html>
    """, width=FRAME_WIDTH_DISPLAY, height=FRAME_HEIGHT_DISPLAY)

@app.route("/video_feed")
def video_feed(): return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Ponto de Entrada Principal ---
if __name__ == '__main__':
    capture_thread = threading.Thread(target=capture_and_detect); capture_thread.daemon = True; capture_thread.start()
    logging.info(f"A iniciar servidor Flask em http://0.0.0.0:5000 com nível de log {log_level_str}")
    # Usa app.run do Flask apenas para desenvolvimento/teste simples. Para produção, considere Gunicorn+Gevent.
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    logging.info("Servidor Flask terminado.")

