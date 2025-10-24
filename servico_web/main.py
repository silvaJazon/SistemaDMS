# Documentação: Serviço Web para o FluxoAI (Microsserviços)
# Responsabilidade: Receber frames analisados via MQTT e fazer stream para o navegador.

import paho.mqtt.client as mqtt
import os
import time
import logging
import sys
import numpy as np
import cv2 # Para descodificar o frame
import threading
from flask import Flask, Response, render_template_string

# --- Configuração do Logging ---
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - Web - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Reduzir logs do Werkzeug
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# --- Configurações da Aplicação ---
MQTT_BROKER_HOST = os.environ.get('MQTT_BROKER_HOST', 'localhost')
MQTT_BROKER_PORT = int(os.environ.get('MQTT_BROKER_PORT', 1883))
SUBSCRIBE_TOPIC_ANALYZED = os.environ.get('SUBSCRIBE_TOPIC_ANALYZED', 'fluxoai/frames/analyzed')
FLASK_PORT = int(os.environ.get('FLASK_PORT', 5000))
FRAME_WIDTH_DISPLAY = int(os.environ.get('FRAME_WIDTH', 640)) # Largura padrão para placeholder
FRAME_HEIGHT_DISPLAY = int(os.environ.get('FRAME_HEIGHT', 480)) # Altura padrão para placeholder

# --- Variáveis Globais ---
mqtt_client = None
connected_to_mqtt = False
last_analyzed_frame_lock = threading.Lock()
last_analyzed_frame_data = {'frame': None, 'timestamp': 0}
app = Flask(__name__)

# --- Funções MQTT ---

def on_connect_web(client, userdata, flags, rc, properties=None):
    """Callback de conexão."""
    global connected_to_mqtt
    if rc == 0:
        logging.info(f"Conectado ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        logging.info(f"A subscrever ao tópico de frames analisados: {SUBSCRIBE_TOPIC_ANALYZED}")
        client.subscribe(SUBSCRIBE_TOPIC_ANALYZED, qos=0)
        connected_to_mqtt = True
    else:
        logging.error(f"Falha ao conectar ao Broker MQTT, código: {rc}, erro: {mqtt.connack_string(rc)}")
        connected_to_mqtt = False

def on_disconnect_web(client, userdata, rc, properties=None):
    """Callback de desconexão."""
    global connected_to_mqtt
    logging.warning(f"Desconectado do Broker MQTT (código: {rc}).")
    connected_to_mqtt = False

def on_message_web(client, userdata, msg):
    """Callback para mensagens recebidas (frames analisados)."""
    global last_analyzed_frame_data
    if msg.topic == SUBSCRIBE_TOPIC_ANALYZED:
        try:
            logging.debug(f"Frame analisado recebido ({len(msg.payload)} bytes)")
            # Descodifica o JPEG recebido
            frame_bytes = np.frombuffer(msg.payload, dtype=np.uint8)
            frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
            if frame is not None:
                with last_analyzed_frame_lock:
                    last_analyzed_frame_data['frame'] = frame
                    last_analyzed_frame_data['timestamp'] = time.time()
            else:
                logging.warning("Falha ao descodificar frame analisado recebido.")
        except Exception as e:
            logging.warning(f"Erro ao processar frame analisado: {e}", exc_info=True)

def setup_mqtt_web():
    """Configura e inicia o cliente MQTT para o serviço web."""
    global mqtt_client
    client_id = f"fluxoai-web-{os.getpid()}"
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    mqtt_client.on_connect = on_connect_web
    mqtt_client.on_disconnect = on_disconnect_web
    mqtt_client.on_message = on_message_web

    logging.info(f"A tentar conectar ao Broker MQTT: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    try:
        # Loop de tentativa de conexão inicial
        retry_count = 0
        max_retries = 5
        while not connected_to_mqtt and retry_count < max_retries:
             try:
                 mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
                 mqtt_client.loop_start() # Inicia loop em background para MQTT
                 time.sleep(2) # Espera a conexão estabelecer
                 if connected_to_mqtt:
                     break
             except Exception as conn_e:
                 logging.warning(f"Tentativa {retry_count+1}/{max_retries} falhou ao conectar ao MQTT: {conn_e}")
                 retry_count += 1
                 time.sleep(5) # Espera antes de tentar novamente

        if not connected_to_mqtt:
             logging.error(f"Não foi possível conectar ao Broker MQTT após {max_retries} tentativas. O stream de vídeo pode não funcionar.")
             # Considerar se deve sair ou continuar a correr o Flask na esperança que o MQTT volte
             # sys.exit(1)

        # Se conectou, a thread MQTT continua em background
        logging.info("Cliente MQTT configurado e loop iniciado.")

    except Exception as e:
        logging.error(f"Erro CRÍTICO durante a configuração do MQTT: {e}", exc_info=True)
        sys.exit(1)


# --- Servidor Web Flask ---

def generate_web_frames():
    """Gera frames de vídeo para o stream HTTP a partir dos frames MQTT."""
    global last_analyzed_frame_data, last_analyzed_frame_lock
    last_sent_timestamp = 0
    while True:
        frame_to_encode = None
        current_timestamp = 0
        with last_analyzed_frame_lock:
            if last_analyzed_frame_data['frame'] is not None and last_analyzed_frame_data['timestamp'] > last_sent_timestamp:
                frame_to_encode = last_analyzed_frame_data['frame'].copy()
                current_timestamp = last_analyzed_frame_data['timestamp']

        if frame_to_encode is None:
            # Se não há frame novo, envia placeholder ou espera
            black_frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
            cv2.putText(black_frame, "Aguardando video analisado...", (10, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            (flag, encodedImage) = cv2.imencode(".jpg", black_frame)
            if flag:
                frame_bytes = bytearray(encodedImage)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5) # Espera antes de reenviar placeholder
        else:
            # Codifica e envia o frame analisado
            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode)
            if flag:
                frame_bytes = bytearray(encodedImage)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                last_sent_timestamp = current_timestamp # Marca como enviado
            else:
                 logging.warning("Falha ao codificar frame analisado para stream.")


        # Controla o FPS do stream (não precisa ser muito rápido)
        time.sleep(1/30) # Tenta enviar ~30 FPS para o navegador

@app.route("/")
def index_web():
    """Rota principal que serve a página HTML."""
    # Usamos as dimensões padrão, mas poderiam vir do frame real se necessário
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FluxoAI - Vídeo Analisado</title>
            <style>
                body { font-family: sans-serif; background-color: #222; color: #eee; margin: 0; padding: 20px; text-align: center;}
                h1 { color: #eee; }
                img { border: 1px solid #555; background-color: #000; max-width: 95%; height: auto; margin-top: 20px;}
            </style>
        </head>
        <body>
            <h1>FluxoAI - Vídeo Analisado</h1>
            <img id="stream" src="{{ url_for('video_feed_web') }}" width="{{ width }}" height="{{ height }}">
            <script>
                var stream = document.getElementById("stream");
                stream.onerror = function() {
                    console.log("Erro no stream, a tentar recarregar em 5s...");
                    setTimeout(function() {
                        stream.src = "{{ url_for('video_feed_web') }}?" + new Date().getTime();
                    }, 5000);
                };
            </script>
        </body>
        </html>
    """, width=FRAME_WIDTH_DISPLAY, height=FRAME_HEIGHT_DISPLAY)

@app.route("/video_feed")
def video_feed_web():
    """Rota que serve o stream de vídeo."""
    return Response(generate_web_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    logging.info("--- Iniciando Serviço Web FluxoAI ---")

    # Configura e inicia o cliente MQTT em background
    mqtt_thread = threading.Thread(target=setup_mqtt_web, daemon=True)
    mqtt_thread.start()

    # Inicia o servidor Flask (precisa ser na thread principal)
    logging.info(f">>> A iniciar servidor Flask na porta {FLASK_PORT}...")
    try:
        # Usa 'threaded=True' para permitir múltiplos clientes e o stream
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
         logging.error(f"Erro ao iniciar servidor Flask: {e}", exc_info=True)
    finally:
        logging.info(">>> Servidor Flask terminado.")
        if mqtt_client and mqtt_client.is_connected():
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            logging.info("Cliente MQTT desconectado.")
        sys.exit(0)
