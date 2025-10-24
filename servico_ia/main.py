import cv2
import os
import time
import threading
from flask import Flask, Response, render_template_string

# Variáveis globais para partilhar o frame mais recente e o estado
latest_frame = None
lock = threading.Lock() # Para evitar conflitos ao aceder ao latest_frame
camera_status = "A iniciar..."

# --- Configuração do Flask ---
app = Flask(__name__)

# HTML simples para exibir o vídeo
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FluxoAI - Vídeo ao Vivo</title>
</head>
<body>
    <h1>FluxoAI - Vídeo ao Vivo</h1>
    <p>Status da Câmara: <span id="status">{{ status }}</span></p>
    <img id="bg" src="{{ url_for('video_feed') }}" width="640" height="480">

    <script>
        // Atualiza o status periodicamente (opcional)
        // setInterval(function() {
        //     fetch('/status')
        //         .then(response => response.text())
        //         .then(text => document.getElementById('status').innerText = text);
        // }, 2000);
    </script>
</body>
</html>
"""

def generate_frames():
    """Gera frames de vídeo formatados para streaming MJPEG."""
    global latest_frame, lock, camera_status
    while True:
        with lock:
            if latest_frame is None:
                # Se não houver frame, espera um pouco
                time.sleep(0.1)
                continue
            # Codifica o frame para JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", latest_frame)
            if not flag:
                continue

        # Produz o frame no formato MJPEG
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03) # Limita um pouco a taxa de frames para não sobrecarregar

@app.route('/')
def index():
    """Rota principal que serve a página HTML."""
    global camera_status
    return render_template_string(HTML_TEMPLATE, status=camera_status)

@app.route('/video_feed')
def video_feed():
    """Rota que fornece o stream de vídeo."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    """Rota para obter o status atual da câmara (opcional)."""
    global camera_status
    return Response(camera_status, mimetype='text/plain')

# --- Lógica de Captura OpenCV (numa thread separada) ---
def capture_video():
    """Função que corre numa thread separada para capturar vídeo."""
    global latest_frame, lock, camera_status

    print(">>> Serviço de IA do FluxoAI a iniciar (Fase 2.5: Web Stream)...")
    print(f">>> Versão do OpenCV: {cv2.__version__}")

    video_source = os.environ.get("VIDEO_SOURCE", "0") # Default para câmara 0 se não definida

    cap = None
    source_description = ""
    frame_count = 0 # Contador para logs menos frequentes

    while True: # Loop principal para tentar conectar e capturar
        try:
            if cap is None or not cap.isOpened():
                camera_status = f"A tentar conectar a: {video_source}..."
                print(f">>> {camera_status}")

                # Tenta converter para inteiro (câmara local)
                try:
                    source_index = int(video_source)
                    source_description = f"câmara local no índice: {source_index}"
                    cap = cv2.VideoCapture(source_index)
                    # Tenta definir uma resolução menor para webcams USB (pode falhar, mas tentamos)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                except ValueError:
                    # Se não for inteiro, assume que é URL RTSP
                    source_description = f"stream de rede: {video_source}"
                    cap = cv2.VideoCapture(video_source)

                if not cap.isOpened():
                    camera_status = f"ERRO: Não foi possível abrir {source_description}. A tentar novamente em 5s..."
                    print(f"!!! {camera_status}")
                    cap = None # Garante que tentamos reconectar
                    time.sleep(5)
                    continue # Volta ao início do loop while

                camera_status = f"Conectado com sucesso a {source_description}!"
                print(f">>> {camera_status}")
                print(">>> A iniciar loop de captura...")

            # Lê um frame
            ret, frame = cap.read()
            frame_count += 1

            if not ret:
                camera_status = "ERRO: Stream/Câmara desconectado. A tentar reconectar..."
                print(f"!!! {camera_status}")
                cap.release()
                cap = None # Força a reconexão no próximo ciclo
                time.sleep(2)
                continue # Volta ao início do loop while

            # Guarda o frame capturado para a thread do Flask
            with lock:
                latest_frame = frame.copy()
                # Atualiza o status com a resolução (apenas a cada 100 frames para não poluir)
                if frame_count % 100 == 1: # Atualiza no primeiro frame e depois a cada 100
                    h, w = frame.shape[:2]
                    camera_status = f"Captura a funcionar | Resolução: {w}x{h}"
                    print(f">>> {camera_status} (Frame {frame_count})")


            # Pequena pausa para não consumir 100% da CPU
            time.sleep(0.01)

        except Exception as e:
            camera_status = f"ERRO inesperado na captura: {e}. A tentar recuperar..."
            print(f"!!! {camera_status}")
            if cap is not None:
                cap.release()
            cap = None
            time.sleep(5)

# --- Início da Aplicação ---
if __name__ == '__main__':
    # Inicia a thread de captura de vídeo
    capture_thread = threading.Thread(target=capture_video)
    capture_thread.daemon = True # Permite que o programa saia mesmo se a thread estiver a correr
    capture_thread.start()

    # Inicia o servidor Flask
    print(">>> A iniciar servidor Flask na porta 5000...")
    # Use threaded=True para lidar com múltiplos pedidos (necessário para streaming)
    # Use debug=False em produção
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

