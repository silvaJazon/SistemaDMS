import cv2
import time
import os
import sys
import threading
from flask import Flask, Response, render_template_string

# Variável global para armazenar o último frame codificado em JPEG
output_frame = None
# Lock para garantir acesso seguro ao output_frame por múltiplas threads
lock = threading.Lock()

# Cria a aplicação Flask
app = Flask(__name__)

# String HTML simples para a página web
HTML_PAGE = """
<html>
<head>
    <title>FluxoAI - Vídeo ao Vivo</title>
</head>
<body>
    <h1>FluxoAI - Vídeo ao Vivo</h1>
    <img src="{{ url_for('video_feed') }}" width="640" height="480">
</body>
</html>
"""

def capture_video():
    """Função que roda numa thread separada para capturar e processar vídeo."""
    global output_frame, lock

    print(">>> Serviço de IA do FluxoAI a iniciar (Fase 2.5: Web Stream)...")
    print(f">>> Versão do OpenCV: {cv2.__version__}")

    video_source = os.environ.get('VIDEO_SOURCE')
    if video_source is None:
        print("!!! ERRO: VIDEO_SOURCE não definido.", flush=True)
        sys.exit(1)

    is_rtsp = video_source.startswith("rtsp://")
    cap = None
    retry_delay = 5

    while True: # Loop principal para manter a captura a correr e tentar reconectar
        if cap is None or not cap.isOpened():
            try:
                if is_rtsp:
                    print(f">>> A tentar conectar-se ao stream: {video_source}", flush=True)
                    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
                else:
                    camera_index = int(video_source)
                    print(f">>> A tentar conectar-se à câmara local: {camera_index}", flush=True)
                    cap = cv2.VideoCapture(camera_index)

                if cap is None or not cap.isOpened():
                    raise ValueError("Falha ao abrir VideoCapture")

                print(">>> Fonte de vídeo conectada!", flush=True)

            except Exception as e:
                print(f"!!! ERRO ao abrir fonte: {e}", flush=True)
                print(f">>> A tentar novamente em {retry_delay}s...", flush=True)
                cap = None
                time.sleep(retry_delay)
                continue # Volta ao início do loop while True

        # Lê um frame
        ret, frame = cap.read()

        if not ret:
            print("!!! ERRO: Não foi possível ler o frame. A tentar reconectar...", flush=True)
            if cap is not None:
                cap.release()
            cap = None
            time.sleep(retry_delay) # Espera antes de tentar reconectar
            continue # Volta ao início do loop while True

        # --- Área para IA ---
        # Aqui podemos processar o 'frame' com o modelo de IA
        # e desenhar caixas, etc., antes de o codificar.
        # --- Fim da IA ---

        # Codifica o frame como JPEG
        (flag, encoded_image) = cv2.imencode(".jpg", frame)

        # Se a codificação falhar, pula este frame
        if not flag:
            continue

        # Adquire o lock e atualiza o output_frame global
        with lock:
            output_frame = encoded_image.tobytes()

        # Pequena pausa opcional
        # time.sleep(0.01)


def generate_stream():
    """Gerador que produz o stream MJPEG para o Flask."""
    global output_frame, lock

    while True:
        # Espera até que um frame esteja disponível
        with lock:
            if output_frame is None:
                continue
            frame_bytes = output_frame

        # Produz o frame no formato multipart/x-mixed-replace
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Pequena pausa para controlar o fluxo
        time.sleep(0.03) # Aproximadamente 30 FPS


@app.route("/")
def index():
    """Rota principal que serve a página HTML."""
    return render_template_string(HTML_PAGE)

@app.route("/video_feed")
def video_feed():
    """Rota que serve o stream de vídeo MJPEG."""
    # Retorna uma resposta de streaming usando o gerador
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Inicia a thread de captura de vídeo
    video_thread = threading.Thread(target=capture_video)
    video_thread.daemon = True # Permite que o programa principal saia mesmo se a thread estiver a correr
    video_thread.start()

    # Inicia o servidor web Flask
    # host='0.0.0.0' torna o servidor acessível de fora do contentor
    print(">>> A iniciar servidor web na porta 5000...", flush=True)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

