# Documentação: Script principal para o Serviço de IA do Projeto FluxoAI
# Fase 3: Deteção de Pessoas (Otimizado)

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
DETECTION_THRESHOLD = 0.4 # REDUZIDO para testes (40%)
TARGET_LABEL = 'person'
PROCESS_EVERY_N_FRAMES = 2 # Processa 1 em cada 2 frames para melhorar performance

# --- Variáveis Globais ---
output_frame_display = None
last_inference_time = 0
last_detections_count = 0
lock = threading.Lock()
app = Flask(__name__)

# --- Funções Auxiliares ---

def load_labels(path):
    """Carrega as etiquetas, tratando possíveis índices 0 ou 1."""
    labels = {}
    try:
        with open(path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                # Assume que o modelo pode retornar classe 0 ou 1 para 'person'
                # (dependendo se há classe 'background')
                labels[i] = line.strip()
        print(f">>> Etiquetas carregadas ({len(labels)}): {list(labels.values())[:5]}...") # Mostra as 5 primeiras
        if TARGET_LABEL not in labels.values():
            print(f"!!! AVISO: A etiqueta alvo '{TARGET_LABEL}' não foi encontrada nas primeiras {len(labels)} etiquetas!")
        return labels
    except FileNotFoundError:
        print(f"!!! ERRO FATAL: Ficheiro de etiquetas não encontrado em {path}")
        sys.exit(1) # Termina o programa se não encontrar as etiquetas
    except Exception as e:
        print(f"!!! ERRO ao carregar etiquetas: {e}")
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

        print(f">>> Modelo TFLite carregado: {MODEL_PATH}")
        print(f">>> Input Shape: {input_details[0]['shape']}, Input Type: {input_details[0]['dtype']}")
        print(f">>> Modelo espera input flutuante: {floating_model}")
        print(f">>> Detalhes dos Outputs: {output_details}") # DEBUG IMPORTANTE

        return interpreter, input_details, output_details, height, width, floating_model
    except Exception as e:
        print(f"!!! ERRO FATAL ao carregar o modelo TFLite ({MODEL_PATH}): {e}")
        sys.exit(1) # Termina o programa se não carregar o modelo


def detect_objects(frame_model_input, interpreter, input_details, output_details, floating_model):
    """Executa a deteção de objetos num frame JÁ REDIMENSIONADO para o modelo."""
    input_data = np.expand_dims(frame_model_input, axis=0)

    if floating_model:
        # Normalização para modelos que esperam float entre -1 e 1
        input_data = (np.float32(input_data) - 127.5) / 127.5
    # else: Modelo uint8 não precisa de normalização extra aqui

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    # Tenta obter os resultados na ordem mais comum para SSD MobileNet
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        # num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0]) # Opcional
    except IndexError:
         # Tenta ordem alternativa (ex: modelos EfficientDet podem ter scores primeiro)
        try:
            scores = interpreter.get_tensor(output_details[0]['index'])[0]
            boxes = interpreter.get_tensor(output_details[1]['index'])[0]
            classes = interpreter.get_tensor(output_details[2]['index'])[0] # Assume 3 outputs
            print("--- Aviso: Ordem alternativa de outputs do modelo detetada (Score, Box, Class).")
        except IndexError as e:
            print(f"!!! ERRO CRÍTICO ao obter outputs do modelo. Verifique 'Detalhes dos Outputs' acima. Erro: {e}")
            return [], [], [], 0 # Retorna vazio se não conseguir interpretar

    # DEBUG: Imprime as primeiras 5 deteções brutas
    # print(f"--- DEBUG: Scores crus: {scores[:5]}")
    # print(f"--- DEBUG: Classes crus: {classes[:5]}")
    # print(f"--- DEBUG: Boxes crus: {boxes[:5]}")


    return boxes, classes, scores, inference_time

def draw_detections(frame_display, boxes, classes, scores, labels, model_input_width, model_input_height):
    """Desenha as caixas no frame de exibição (display)."""
    display_height, display_width, _ = frame_display.shape
    detections_count = 0

    for i in range(len(scores)):
        if scores[i] > DETECTION_THRESHOLD:
            class_id = int(classes[i])
            label = labels.get(class_id, f'ID:{class_id}')

            # DEBUG: Imprime a deteção válida
            # print(f"--- DEBUG: Deteção Válida - Índice: {i}, Score: {scores[i]:.2f}, Classe ID: {class_id}, Etiqueta: {label}")

            if label == TARGET_LABEL:
                detections_count += 1
                # Coordenadas [ymin, xmin, ymax, xmax] normalizadas (0 a 1)
                ymin, xmin, ymax, xmax = boxes[i]

                # Converte coordenadas normalizadas para pixels no frame de DISPLAY
                xmin = int(xmin * display_width)
                xmax = int(xmax * display_width)
                ymin = int(ymin * display_height)
                ymax = int(ymax * display_height)

                # Garante limites
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(display_width - 1, xmax)
                ymax = min(display_height - 1, ymax)

                if xmax > xmin and ymax > ymin:
                    cv2.rectangle(frame_display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Verde
                    label_text = f'{label}: {int(scores[i]*100)}%'
                    label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_ymin = max(ymin, label_size[1] + 10)
                    cv2.rectangle(frame_display, (xmin, label_ymin - label_size[1] - 10),
                                  (xmin + label_size[0], label_ymin - base_line - 10), (255, 255, 255), cv2.FILLED) # Branco
                    cv2.putText(frame_display, label_text, (xmin, label_ymin - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Preto

    return frame_display, detections_count


# --- Thread de Captura e Deteção ---

def capture_and_detect():
    """Função principal: captura, processa (com saltos) e atualiza frame para stream."""
    global output_frame_display, lock, last_inference_time, last_detections_count

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

    if not cap.isOpened():
        print(f"!!! ERRO FATAL: Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")
        sys.exit(1)

    print(">>> Fonte de vídeo conectada com sucesso!")
    print(f">>> A processar 1 em cada {PROCESS_EVERY_N_FRAMES} frames.")
    print(">>> A iniciar loop de captura e deteção...")

    frame_count = 0
    last_processed_frame_with_detections = None

    while True:
        ret, frame = cap.read()
        if not ret:
            # Lógica de reconexão (simplificada)
            print("!!! Frame não recebido. A tentar reconectar em 5s...")
            cap.release()
            time.sleep(5)
            cap = cv2.VideoCapture(video_source_arg)
            if not cap.isOpened():
                print("!!! Falha ao reconectar. A terminar.")
                break
            else:
                print(">>> Reconectado!")
                continue

        frame_count += 1

        # Redimensiona para o tamanho de exibição desejado
        frame_display = cv2.resize(frame, (FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY))

        # --- Otimização: Processa apenas alguns frames ---
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            start_process_time = time.time()

            # Redimensiona para o tamanho esperado pelo MODELO
            frame_model_input = cv2.resize(frame, (model_width, model_height))

            # Executa a deteção
            boxes, classes, scores, inference_time = detect_objects(
                frame_model_input, interpreter, input_details, output_details, floating_model
            )

            # Desenha as deteções no frame de EXIBIÇÃO
            frame_display_with_detections, detections_count = draw_detections(
                frame_display.copy(), boxes, classes, scores, labels, model_width, model_height
            )

            # Guarda o último frame processado e as estatísticas
            last_processed_frame_with_detections = frame_display_with_detections
            last_inference_time = inference_time
            last_detections_count = detections_count

            end_process_time = time.time()
            process_time = end_process_time - start_process_time
            fps = 1 / process_time if process_time > 0 else 0

            # Atualiza frame de saída e imprime log (APENAS quando processa)
            with lock:
                output_frame_display = last_processed_frame_with_detections.copy()

            res_h, res_w, _ = output_frame_display.shape
            print(f">>> Frame {frame_count} PROCESSADO | Res: {res_w}x{res_h} | Pessoas: {detections_count} | Inferência: {inference_time:.3f}s | FPS Proc: {fps:.1f}")

        else:
            # --- Para frames não processados, apenas atualiza o frame de saída com o último resultado ---
            if last_processed_frame_with_detections is not None:
                # Adiciona info sobre o último processamento no canto
                info_text = f"P: {last_detections_count} ({last_inference_time*1000:.0f}ms)"
                cv2.putText(frame_display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            with lock:
                output_frame_display = frame_display.copy() # Mostra o frame atual sem novas deteções

    cap.release()
    print(">>> Loop de captura terminado.")

# --- Servidor Web Flask ---

def generate_frames():
    """Gera frames para o stream HTTP."""
    global output_frame_display, lock
    while True:
        frame_to_encode = None
        with lock:
            if output_frame_display is not None:
                frame_to_encode = output_frame_display.copy()

        if frame_to_encode is None:
            # Frame Placeholder
            placeholder = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Aguardando...", (30, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            (flag, encodedImage) = cv2.imencode(".jpg", placeholder)
            if not flag: continue
            frame_bytes = bytearray(encodedImage)
            time.sleep(0.5)
        else:
            # Codifica o frame com deteções (ou o frame normal se foi saltado)
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode)
            if not flag: continue
            frame_bytes = bytearray(encodedImage)

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')
        # Pequena pausa para não sobrecarregar CPU com encoding/yield
        time.sleep(0.01) # Reduzido para tentar stream mais fluído

@app.route("/")
def index():
    """Serve a página HTML."""
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
                 // Recarrega a imagem se ela falhar (útil se o servidor reiniciar)
                 var stream = document.getElementById("stream");
                 stream.onerror = function() {
                     console.log("Erro no stream, a tentar recarregar em 5s...");
                     setTimeout(function() {
                         stream.src = "{{ url_for('video_feed') }}?" + new Date().getTime(); // Adiciona timestamp para evitar cache
                     }, 5000);
                 };
             </script>
        </body>
        </html>
    """, width=FRAME_WIDTH_DISPLAY, height=FRAME_HEIGHT_DISPLAY)

@app.route("/video_feed")
def video_feed():
    """Serve o stream de vídeo."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    capture_thread = threading.Thread(target=capture_and_detect)
    capture_thread.daemon = True
    capture_thread.start()

    print(">>> A iniciar servidor Flask na porta 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

print(">>> Servidor Flask terminado.")

