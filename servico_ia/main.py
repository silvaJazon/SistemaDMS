# Documentação: Script principal para o Serviço de IA do Projeto FluxoAI
# Fase 3: Deteção de Pessoas (Caixas Persistentes - Correção Stream)

import cv2
import time
import os
import numpy as np
import threading
from flask import Flask, Response, render_template_string
import tflite_runtime.interpreter as tflite
import sys # Para mensagens de erro

# --- Configurações ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
FRAME_WIDTH_DISPLAY = 640  # Largura para exibir no stream
FRAME_HEIGHT_DISPLAY = 480 # Altura para exibir no stream
MODEL_PATH = 'model.tflite'
LABELS_PATH = 'labels.txt'
DETECTION_THRESHOLD = 0.6 # Voltando para 40%
TARGET_LABEL = 'person'
PROCESS_EVERY_N_FRAMES = 3 # Processa 1 em cada 3 frames
JPEG_QUALITY = 95 # Qualidade do JPEG para o stream (0-100, padrão ~95)

# --- Variáveis Globais ---
output_frame_display = None
last_inference_time = 0
last_detections_count = 0
last_valid_boxes = [] # Guarda as últimas caixas válidas (formato [ymin, xmin, ymax, xmax])
last_valid_classes = [] # Guarda as últimas classes válidas
last_valid_scores = [] # Guarda os últimos scores válidos
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
        print(f">>> Etiquetas carregadas ({len(labels)}): {list(labels.values())[:5]}...")
        if TARGET_LABEL not in labels.values():
            print(f"!!! AVISO: A etiqueta alvo '{TARGET_LABEL}' não foi encontrada nas etiquetas!")
        person_id = -1
        for key, value in labels.items():
            if value == TARGET_LABEL:
                person_id = key
                break
        print(f">>> ID da classe '{TARGET_LABEL}': {person_id}")
        return labels
    except FileNotFoundError:
        print(f"!!! ERRO FATAL: Ficheiro de etiquetas não encontrado em {path}")
        sys.exit(1)
    except Exception as e:
        print(f"!!! ERRO ao carregar etiquetas: {e}")
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

        print(f">>> Modelo TFLite carregado: {MODEL_PATH}")
        print(f">>> Input Shape: {input_details[0]['shape']}, Input Type: {input_details[0]['dtype']}")
        print(f">>> Modelo espera input flutuante: {floating_model}")

        return interpreter, input_details, output_details, height, width, floating_model
    except Exception as e:
        print(f"!!! ERRO FATAL ao carregar o modelo TFLite ({MODEL_PATH}): {e}")
        sys.exit(1)


def detect_objects(frame_model_input, interpreter, input_details, output_details, floating_model):
    """Executa a deteção de objetos."""
    input_data = np.expand_dims(frame_model_input, axis=0)
    if floating_model: input_data = (np.float32(input_data) - 127.5) / 127.5
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time(); interpreter.invoke(); inference_time = time.time() - start_time
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
    except IndexError as e:
        print(f"!!! ERRO CRÍTICO ao obter outputs: {e}"); return [], [], [], 0
    return boxes, classes, scores, inference_time

def draw_single_detection(frame_display, box, class_id, score, labels, color=(0, 255, 0), persistent_tag=False):
    """Desenha UMA caixa de deteção no frame."""
    display_height, display_width, _ = frame_display.shape
    label = labels.get(class_id, f'ID:{class_id}')

    if label == TARGET_LABEL:
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * display_width); xmax = int(xmax * display_width)
        ymin = int(ymin * display_height); ymax = int(ymax * display_height)
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(display_width - 1, xmax); ymax = min(display_height - 1, ymax)

        if xmax > xmin and ymax > ymin:
            cv2.rectangle(frame_display, (xmin, ymin), (xmax, ymax), color, 2)
            label_text = f'{label}: {int(score*100)}%'
            # Removido o tag (P) para evitar piscar
            # if persistent_tag: label_text += ' (P)'

            label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(frame_display, (xmin, label_ymin - label_size[1] - 10),
                          (xmin + label_size[0], label_ymin - base_line - 10), (255, 255, 255), cv2.FILLED) # Branco
            cv2.putText(frame_display, label_text, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Preto
            return True # Indica que uma pessoa foi desenhada
    return False # Indica que esta deteção não era uma pessoa


def draw_new_detections_and_update_last(frame_display, boxes, classes, scores, labels):
    """Processa novas deteções: desenha-as e atualiza a lista de últimas válidas."""
    global last_valid_boxes, last_valid_classes, last_valid_scores
    detections_count = 0
    current_valid_boxes = []
    current_valid_classes = []
    current_valid_scores = []

    for i in range(len(scores)):
        if scores[i] > DETECTION_THRESHOLD:
            # Desenha a caixa (se for pessoa) e verifica se foi desenhada
            if draw_single_detection(frame_display, boxes[i], int(classes[i]), scores[i], labels, color=(0, 255, 0)): # Verde para novas
                detections_count += 1
                # Guarda apenas as informações das PESSOAS detectadas
                current_valid_boxes.append(boxes[i])
                current_valid_classes.append(int(classes[i]))
                current_valid_scores.append(scores[i])

    # Atualiza as listas globais
    last_valid_boxes = current_valid_boxes
    last_valid_classes = current_valid_classes
    last_valid_scores = current_valid_scores

    return frame_display, detections_count

def draw_last_valid_detections(frame_display, labels):
    """Redesenha as últimas caixas válidas guardadas no frame atual (sempre verde)."""
    global last_valid_boxes, last_valid_classes, last_valid_scores
    if last_valid_boxes:
        for i in range(len(last_valid_scores)):
             # Sempre desenha em verde
            draw_single_detection(frame_display, last_valid_boxes[i], last_valid_classes[i], last_valid_scores[i], labels, color=(0, 255, 0))

    return frame_display


# --- Thread de Captura e Deteção ---

def capture_and_detect():
    """Função principal: captura, processa (com saltos) e atualiza frame para stream."""
    global output_frame_display, lock, last_inference_time, last_detections_count
    # last_valid_* são atualizados dentro de draw_new_detections_and_update_last

    print(">>> Serviço de IA do FluxoAI a iniciar (Fase 3: Deteção de Pessoas)...")
    print(f">>> Versão do OpenCV: {cv2.__version__}")

    interpreter, input_details, output_details, model_height, model_width, floating_model = initialize_model()
    labels = load_labels(LABELS_PATH)

    is_rtsp = VIDEO_SOURCE.startswith("rtsp://")
    source_description = f"stream de rede: {VIDEO_SOURCE}" if is_rtsp else f"câmara local no índice: {VIDEO_SOURCE}"
    video_source_arg = VIDEO_SOURCE if is_rtsp else int(VIDEO_SOURCE)

    print(f">>> A tentar conectar a: {source_description}...")
    cap = cv2.VideoCapture(video_source_arg)
    time.sleep(2)

    if not cap.isOpened(): print(f"!!! ERRO FATAL: Vídeo: {VIDEO_SOURCE}"); sys.exit(1)

    print(">>> Fonte de vídeo conectada!"); print(f">>> Processando 1 em {PROCESS_EVERY_N_FRAMES} frames."); print(">>> Loop...")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("!!! Frame não recebido. Reconectando..."); cap.release(); time.sleep(5)
            cap = cv2.VideoCapture(video_source_arg)
            if not cap.isOpened(): print("!!! Falha reconexão."); break
            else: print(">>> Reconectado!"); continue

        frame_count += 1
        frame_display = cv2.resize(frame, (FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY))

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            start_process_time = time.time()
            frame_model_input = cv2.resize(frame, (model_width, model_height)) # Usa frame original para IA

            boxes, classes, scores, inference_time = detect_objects(
                frame_model_input, interpreter, input_details, output_details, floating_model
            )

            # Desenha as NOVAS deteções e atualiza as 'last_valid_*'
            frame_display_processed, detections_count = draw_new_detections_and_update_last(
                frame_display, boxes, classes, scores, labels
            )

            last_inference_time = inference_time
            last_detections_count = detections_count

            end_process_time = time.time(); process_time = end_process_time - start_process_time
            fps = 1 / process_time if process_time > 0 else 0

            # Atualiza o frame de saída com as novas deteções
            with lock: output_frame_display = frame_display_processed.copy()

            res_h, res_w, _ = output_frame_display.shape
            print(f">>> Frame {frame_count} PROCESSADO | Res: {res_w}x{res_h} | Pessoas: {detections_count} | Inferência: {inference_time:.3f}s | FPS Proc: {fps:.1f}")

        else:
            # Frame NÃO processado pela IA - Redesenha as ÚLTIMAS caixas válidas (sempre verde)
            frame_display_with_persist = draw_last_valid_detections(frame_display, labels)

            # Adiciona info sobre o último processamento no canto
            info_text = f"P: {last_detections_count} ({last_inference_time*1000:.0f}ms @ F{frame_count - (frame_count % PROCESS_EVERY_N_FRAMES)})"
            cv2.putText(frame_display_with_persist, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # Vermelho para info

            # Atualiza o frame de saída
            with lock: output_frame_display = frame_display_with_persist.copy()


    cap.release()
    print(">>> Loop de captura terminado.")

# --- Servidor Web Flask ---

def generate_frames():
    """Gera frames para o stream HTTP com qualidade JPEG ajustada."""
    global output_frame_display, lock
    while True:
        frame_to_encode = None
        frame_bytes = None # Garante que a variável existe
        with lock:
            if output_frame_display is not None:
                frame_to_encode = output_frame_display.copy()

        if frame_to_encode is None:
            placeholder = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Aguardando...", (30, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            (flag, encodedImage) = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if flag: # Só define frame_bytes se a codificação funcionar
                frame_bytes = bytearray(encodedImage)
            time.sleep(0.5) # Pausa mesmo se a codificação falhar
        else:
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if flag: # Só define frame_bytes se a codificação funcionar
                 frame_bytes = bytearray(encodedImage)

        # Só envia o frame se frame_bytes foi definido com sucesso
        if frame_bytes is not None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')

        time.sleep(0.01) # Pequena pausa para não sobrecarregar CPU com envio


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
    print(">>> A iniciar servidor Flask na porta 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    print(">>> Servidor Flask terminado.")

