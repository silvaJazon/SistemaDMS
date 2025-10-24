# Documentação: Script principal para o Serviço de IA do Projeto FluxoAI
# Fase 3: Deteção de Pessoas

import cv2
import time
import os
import numpy as np
import threading # Para executar a captura em paralelo com o servidor web
from flask import Flask, Response, render_template_string
import tflite_runtime.interpreter as tflite # Importa o motor TFLite

# --- Configurações ---
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0") # Lê da variável de ambiente, default é a câmara 0
FRAME_WIDTH = 640  # Largura desejada para processamento
FRAME_HEIGHT = 480 # Altura desejada para processamento
MODEL_PATH = 'model.tflite'
LABELS_PATH = 'labels.txt'
DETECTION_THRESHOLD = 0.5 # Confiança mínima para mostrar uma deteção (50%)
TARGET_LABEL = 'person' # Objeto que queremos detetar

# --- Variáveis Globais ---
output_frame = None
lock = threading.Lock() # Para acesso seguro ao output_frame por múltiplas threads
app = Flask(__name__)

# --- Funções Auxiliares ---

def load_labels(path):
    """Carrega as etiquetas do ficheiro de texto."""
    try:
        with open(path, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f.readlines())}
    except FileNotFoundError:
        print(f"!!! ERRO FATAL: Ficheiro de etiquetas não encontrado em {path}")
        return None

def initialize_model():
    """Carrega o modelo TFLite e aloca tensores."""
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # Obtém a altura e largura esperadas pelo modelo
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        # Verifica se o modelo espera floats ou inteiros (uint8)
        floating_model = (input_details[0]['dtype'] == np.float32)
        print(">>> Modelo TFLite carregado com sucesso.")
        return interpreter, input_details, output_details, height, width, floating_model
    except Exception as e:
        print(f"!!! ERRO FATAL ao carregar o modelo TFLite ({MODEL_PATH}): {e}")
        return None, None, None, None, None, None

def detect_objects(frame, interpreter, input_details, output_details, model_height, model_width, floating_model):
    """Executa a deteção de objetos num frame."""
    # Redimensiona e prepara a imagem de entrada
    image_resized = cv2.resize(frame, (model_width, model_height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normaliza os pixels se o modelo esperar floats
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    else:
        # Se o modelo esperar uint8, apenas garante o tipo correto
        input_data = np.uint8(input_data)


    # Define o tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Executa a inferência
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    # Obtém os resultados
    # A ordem dos outputs pode variar dependendo do modelo. Verifique a documentação do seu modelo específico.
    # Normalmente para modelos SSD MobileNet do TF Object Detection API:
    # output_details[0]: caixas (locations)
    # output_details[1]: classes
    # output_details[2]: scores
    # output_details[3]: número de deteções (pode não existir em alguns modelos)

    # Tentativa de obter os resultados com base na ordem comum
    try:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Coordenadas das caixas [N, 4]
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Índices das classes [N]
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confianças das deteções [N]
        # num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0]) # Número de deteções válidas
    except IndexError:
         # Alguns modelos podem ter uma ordem diferente ou menos outputs.
         # Ordem alternativa comum (verifique os detalhes do seu 'output_details'):
         try:
             scores = interpreter.get_tensor(output_details[0]['index'])[0]
             boxes = interpreter.get_tensor(output_details[1]['index'])[0]
             # Se houver apenas 2 outputs, a classe pode estar implícita ou ausente
             classes = np.zeros_like(scores) # Placeholder se a classe não estiver explícita
             print("Aviso: Ordem alternativa de outputs do modelo detetada.")
         except IndexError as e:
            print(f"!!! ERRO ao obter outputs do modelo. Detalhes: {output_details}. Erro: {e}")
            return [], [], [], 0 # Retorna vazio se não conseguir interpretar

    # print(f"Raw scores: {scores[:5]}") # Debug: Ver os primeiros scores
    # print(f"Raw classes: {classes[:5]}") # Debug: Ver as primeiras classes
    # print(f"Raw boxes: {boxes[:5]}") # Debug: Ver as primeiras boxes


    return boxes, classes, scores, inference_time

def draw_detections(frame, boxes, classes, scores, labels):
    """Desenha as caixas de deteção no frame."""
    frame_height, frame_width, _ = frame.shape
    detections_count = 0
    valid_detections = 0

    for i in range(len(scores)):
        if scores[i] > DETECTION_THRESHOLD:
            valid_detections += 1
            class_id = int(classes[i])
            label = labels.get(class_id, f'ID:{class_id}') # Usa ID se a etiqueta não for encontrada

            if label == TARGET_LABEL: # Apenas desenha se for o objeto alvo ("person")
                detections_count += 1
                # Obtém as coordenadas da caixa e converte para pixels no frame original
                # Formato da caixa [ymin, xmin, ymax, xmax] normalizado entre 0 e 1
                ymin, xmin, ymax, xmax = boxes[i]
                xmin = int(xmin * frame_width)
                xmax = int(xmax * frame_width)
                ymin = int(ymin * frame_height)
                ymax = int(ymax * frame_height)

                # Garante que as coordenadas estão dentro dos limites da imagem
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(frame_width - 1, xmax)
                ymax = min(frame_height - 1, ymax)

                # Desenha o retângulo apenas se for válido
                if xmax > xmin and ymax > ymin:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Verde

                    # Prepara o texto da etiqueta (classe + confiança)
                    label_text = f'{label}: {int(scores[i]*100)}%'
                    label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_ymin = max(ymin, label_size[1] + 10) # Garante que não sai do topo

                    # Desenha um fundo para a etiqueta
                    cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                                  (xmin + label_size[0], label_ymin - base_line - 10), (255, 255, 255), cv2.FILLED) # Branco
                    # Escreve o texto da etiqueta
                    cv2.putText(frame, label_text, (xmin, label_ymin - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Preto

    # print(f"Total valid detections above threshold: {valid_detections}") # Debug
    return frame, detections_count


# --- Thread de Captura e Deteção ---

def capture_and_detect():
    """Função principal executada em background para capturar e processar vídeo."""
    global output_frame, lock

    print(">>> Serviço de IA do FluxoAI a iniciar (Fase 3: Deteção de Pessoas)...")
    print(f">>> Versão do OpenCV: {cv2.__version__}")

    # Carrega o modelo TFLite
    interpreter, input_details, output_details, model_height, model_width, floating_model = initialize_model()
    if interpreter is None:
        return # Termina a thread se o modelo não carregar

    # Carrega as etiquetas
    labels = load_labels(LABELS_PATH)
    if labels is None:
        return # Termina se não conseguir carregar as etiquetas

    # Tenta abrir a fonte de vídeo
    is_rtsp = VIDEO_SOURCE.startswith("rtsp://")
    source_description = f"stream de rede: {VIDEO_SOURCE}" if is_rtsp else f"câmara local no índice: {VIDEO_SOURCE}"
    video_source_arg = VIDEO_SOURCE if is_rtsp else int(VIDEO_SOURCE)

    print(f">>> A tentar conectar a: {source_description}...")
    cap = cv2.VideoCapture(video_source_arg)
    time.sleep(2) # Pausa para dar tempo à câmara/stream para inicializar

    if not cap.isOpened():
        print(f"!!! ERRO FATAL: Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")
        print("!!! Verifique se a câmara está conectada ou se o URL RTSP e a rede estão corretos.")
        return # Termina a thread se não conseguir abrir

    print(">>> Fonte de vídeo conectada com sucesso!")
    print(">>> A iniciar loop de captura e deteção...")

    frame_count = 0
    last_log_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("!!! Frame não recebido. A verificar ligação...")
            if is_rtsp:
                print(">>> A tentar reconectar ao stream RTSP...")
                cap.release()
                time.sleep(5) # Espera 5 segundos antes de tentar reconectar
                cap = cv2.VideoCapture(video_source_arg)
                if not cap.isOpened():
                    print("!!! Falha ao reconectar. A terminar.")
                    break # Sai do loop se não conseguir reconectar
                else:
                    print(">>> Reconectado com sucesso!")
                    continue # Volta ao início do loop
            else:
                print("!!! Falha ao ler frame da câmara local. A terminar.")
                break # Sai do loop se for câmara local

        frame_count += 1
        start_process_time = time.time()

        # Redimensiona o frame para o tamanho desejado (antes da deteção)
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Executa a deteção
        boxes, classes, scores, inference_time = detect_objects(frame_resized, interpreter, input_details, output_details, model_height, model_width, floating_model)

        # Desenha as caixas de deteção no frame redimensionado
        frame_with_detections, detections_count = draw_detections(frame_resized, boxes, classes, scores, labels)

        # Atualiza o frame de saída para o servidor web (de forma segura)
        with lock:
            output_frame = frame_with_detections.copy()

        end_process_time = time.time()
        process_time = end_process_time - start_process_time
        fps = 1 / process_time if process_time > 0 else 0

        # Imprime um log a cada X segundos (ex: a cada 5 segundos)
        current_time = time.time()
        if current_time - last_log_time >= 5:
            res_h, res_w, _ = frame_with_detections.shape
            print(f">>> Deteção a funcionar! Frame {frame_count} | Res: {res_w}x{res_h} | Pessoas: {detections_count} | Inferência: {inference_time:.3f}s | FPS Proc: {fps:.1f}")
            last_log_time = current_time

    cap.release()
    print(">>> Loop de captura terminado.")

# --- Servidor Web Flask ---

def generate_frames():
    """Gera frames de vídeo para o stream HTTP."""
    global output_frame, lock
    while True:
        frame_to_encode = None
        with lock:
            if output_frame is not None:
                frame_to_encode = output_frame.copy()

        if frame_to_encode is None:
            # Se ainda não temos frames, envia uma imagem placeholder simples ou espera
            # Cria uma imagem preta simples
            black_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(black_frame, "Aguardando video...", (30, FRAME_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            (flag, encodedImage) = cv2.imencode(".jpg", black_frame)
            if not flag:
                continue
            frame_bytes = bytearray(encodedImage)
            time.sleep(0.5) # Espera meio segundo antes de tentar de novo
        else:
            # Codifica o frame atual como JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode)
            if not flag:
                continue
            frame_bytes = bytearray(encodedImage)

        # Envia o frame codificado como parte de uma resposta multipart
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')
        # Pequena pausa para controlar o FPS do stream
        time.sleep(0.03) # Tenta ~30 FPS para o stream

@app.route("/")
def index():
    """Rota principal que serve a página HTML."""
    # Simples HTML para exibir o stream
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
    """, width=FRAME_WIDTH, height=FRAME_HEIGHT)

@app.route("/video_feed")
def video_feed():
    """Rota que serve o stream de vídeo."""
    # Retorna a resposta do gerador de frames
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    # Inicia a thread de captura e deteção em background
    capture_thread = threading.Thread(target=capture_and_detect)
    capture_thread.daemon = True # Permite que o programa principal saia mesmo se a thread estiver a correr
    capture_thread.start()

    # Inicia o servidor Flask
    print(">>> A iniciar servidor Flask na porta 5000...")
    # 'host=0.0.0.0' torna o servidor acessível na rede local
    # 'threaded=True' permite que o Flask lide com múltiplos pedidos (necessário para o stream)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

print(">>> Servidor Flask terminado.")

