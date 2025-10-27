# Documentação: Aplicação Principal Flask (O "Orquestrador")
# Responsabilidades:
# 1. Iniciar e gerir as threads (Câmara, Deteção, EventHandler).
# 2. Servir a interface web (HTML, CSS, JS) para calibração e visualização.
# 3. Fornecer APIs (/api/config, /api/alerts) para a interface web.
# 4. Servir o stream de vídeo (/video_feed).
# 5. Servir as imagens dos alertas (/alerts/images).

import cv2
import time
import os
import numpy as np
import threading
import logging
import sys
from flask import Flask, Response, render_template, jsonify, request, send_from_directory
import queue # Para comunicação entre threads
from datetime import datetime
import json # Para ler o ficheiro JSONL

# Importa os nossos módulos
from camera_thread import CameraThread
from dms_core import DriverMonitor
from event_handler import EventHandler # A "central" de alertas

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

# --- Configuração do Logging ---
# Obtém o nível de log da variável de ambiente ou usa INFO como padrão
default_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
# Configuração básica do logging
logging.basicConfig(level=default_log_level,
                    format='%(asctime)s - DMS - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
# Silencia logs excessivos do Flask em modo INFO
log = logging.getLogger('werkzeug')
if default_log_level == 'INFO':
    log.setLevel(logging.WARNING)
logging.info(f"Nível de log definido para: {default_log_level}")

# --- Configurações da Aplicação ---
# Fonte de vídeo (0 para USB, ou URL RTSP)
VIDEO_SOURCE = os.environ.get('VIDEO_SOURCE', "0")
# Rotação do frame (0, 90, 180, 270) - lido do ambiente
ROTATE_FRAME = int(os.environ.get('ROTATE_FRAME', "0"))
# Dimensões para exibição e processamento
FRAME_WIDTH_DISPLAY = 640  
FRAME_HEIGHT_DISPLAY = 480 
# Qualidade do JPEG para o stream de vídeo
JPEG_QUALITY = 75
# FPS alvo para o loop de deteção (ajustado para ~5 FPS para RPi)
TARGET_FPS = 5
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
# Caminho para guardar os alertas (deve corresponder ao Dockerfile e event_handler.py)
ALERTS_SAVE_PATH = "/app/alerts" 

# --- Variáveis Globais ---
# Frame mais recente processado para exibição no stream
output_frame_stream = None
# Lock para acesso seguro ao output_frame_stream
lock = threading.Lock()
# Flag para indicar se a aplicação deve parar
shutdown_flag = threading.Event()
# Fila para passar eventos (alerta + frame) do loop de deteção para o event_handler
event_queue = queue.Queue(maxsize=10) # Fila com tamanho máximo para evitar consumo excessivo de memória

# --- Inicialização do Flask ---
app = Flask(__name__)

# --- Threads ---
cam_thread = None
dms_monitor = None
event_handler = None
detection_thread = None # Thread para o loop de deteção

# --- Funções Auxiliares ---

def generate_frames():
    """Gera frames de vídeo (JPEG) para o stream HTTP."""
    global output_frame_stream, lock
    
    # Cria um frame placeholder enquanto a câmara não inicia
    placeholder = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Aguardando...", (30, FRAME_HEIGHT_DISPLAY // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    while not shutdown_flag.is_set():
        frame_to_encode = None
        
        with lock:
            if output_frame_stream is not None:
                # Copia o frame mais recente para evitar problemas de concorrência
                frame_to_encode = output_frame_stream.copy()
            else:
                # Usa o placeholder se ainda não houver frame
                frame_to_encode = placeholder

        # Codifica o frame para JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", frame_to_encode, 
                                            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        
        # Garante que a codificação foi bem-sucedida
        if not flag:
            logging.warning("Falha ao codificar frame para JPEG.")
            time.sleep(0.1) # Pausa curta para evitar loop infinito rápido
            continue

        # Produz o frame no formato multipart/x-mixed-replace
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')
        
        # Controla o FPS do stream (aproximadamente)
        time.sleep(1/30) # Tenta enviar a ~30 FPS (o processamento é mais lento)

# --- Loop Principal de Deteção (Executado numa Thread) ---

def detection_loop(cam_thread, dms_monitor, event_queue):
    """
    Loop que executa continuamente a deteção nos frames da câmara.
    """
    global output_frame_stream, lock
    logging.info(f">>> Loop de deteção iniciado (Alvo: {TARGET_FPS} FPS).")
    
    last_process_time = time.time()

    while not shutdown_flag.is_set():
        start_time = time.time()
        
        # Obtém o frame mais recente da thread da câmara
        frame = cam_thread.get_frame()
        
        if frame is None:
            #logging.warning("Frame não recebido da câmara. A aguardar...")
            time.sleep(0.1) # Espera um pouco se não houver frame
            continue
        
        # Converte para escala de cinza (necessário para Dlib)
        try:
            # (CORRIGIDO) Usa COLOR_BGR2GRAY
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        except Exception as e:
             logging.error(f"Erro ao converter frame para cinza: {e}", exc_info=True)
             time.sleep(TARGET_FRAME_TIME) # Pausa antes de tentar o próximo
             continue

        # Processa o frame no dms_core
        try:
            # (CORRIGIDO) Recebe 2 valores: frame com desenhos e lista de eventos
            processed_frame, events = dms_monitor.process_frame(frame.copy(), gray) 
        except Exception as e:
            logging.error(f"!!! Erro fatal no process_frame: {e}", exc_info=True)
            # Em caso de erro grave no core, pausa para evitar spam de logs
            processed_frame = frame.copy() # Mostra o frame original
            cv2.putText(processed_frame, "ERRO NO PROCESSAMENTO!", (10, FRAME_HEIGHT_DISPLAY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            events = []
            time.sleep(1.0) 

        # Atualiza o frame para o stream web
        with lock:
            output_frame_stream = processed_frame.copy()

        # Envia eventos (se houver) para a fila do event_handler
        if events:
            for event in events:
                try:
                    # Envia o dicionário do evento E uma cópia do frame ORIGINAL
                    # (sem os desenhos do alerta, para guardar a imagem "limpa")
                    event_package = {"event": event, "frame": frame.copy()}
                    event_queue.put(event_package, block=False) # Não bloqueia se a fila estiver cheia
                except queue.Full:
                    logging.warning("Fila de eventos cheia! Descartando evento.")
                except Exception as e:
                    logging.error(f"Erro ao colocar evento na fila: {e}", exc_info=True)

        # Controla o FPS do loop de deteção
        elapsed_time = time.time() - start_time
        sleep_time = TARGET_FRAME_TIME - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # Log apenas se o atraso for significativo (ex: mais de 10% do tempo alvo)
             if abs(sleep_time) > TARGET_FRAME_TIME * 0.10:
                # (CORRIGIDO) Usa round() para formatar
                logging.warning(f"!!! LOOP LENTO. Processamento demorou {round(elapsed_time, 2)}s (Alvo era {round(TARGET_FRAME_TIME, 2)}s)")
                
    logging.info(">>> Loop de deteção terminado.")

# --- Rotas Flask ---

@app.route("/")
def index():
    """Serve a página principal de calibração."""
    source_desc = cam_thread.source_description if cam_thread else "Indisponível"
    return render_template("index.html", 
                           width=FRAME_WIDTH_DISPLAY, 
                           height=FRAME_HEIGHT_DISPLAY, 
                           cam_source_desc=source_desc)

@app.route("/video_feed")
def video_feed():
    """Serve o stream de vídeo MJPEG."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Rotas da API de Configuração ---

@app.route("/api/config", methods=['GET'])
def get_config():
    """Retorna as configurações atuais para a interface web."""
    if dms_monitor and cam_thread:
        try:
            dms_settings = dms_monitor.get_settings()
            # (CORRIGIDO) Obtém brilho e rotação da camera_thread
            cam_brightness = cam_thread.get_brightness() 
            cam_rotation = cam_thread.get_rotation()
            
            # Combina tudo num único objeto
            full_config = {
                **dms_settings, 
                "brightness": cam_brightness,
                "rotation": cam_rotation
            }
            return jsonify(full_config)
        except Exception as e:
            logging.error(f"Erro ao obter configurações: {e}", exc_info=True)
            return jsonify({"error": "Falha ao ler configurações"}), 500
    else:
        return jsonify({"error": "Serviço não inicializado"}), 503

@app.route("/api/config", methods=['POST'])
def set_config():
    """Recebe novas configurações da interface web e aplica-as."""
    data = request.json
    if dms_monitor and cam_thread and data:
        try:
            # Separa as configurações do DMS e da câmara
            dms_settings = {
                'ear_threshold': data.get('ear_threshold'),
                'ear_consec_frames': data.get('ear_consec_frames'),
                'distraction_threshold_angle': data.get('distraction_threshold_angle'),
                'distraction_consec_frames': data.get('distraction_consec_frames'),
            }
            # Remove chaves None para não sobrescrever com nada
            dms_settings = {k: v for k, v in dms_settings.items() if v is not None}
            
            # Atualiza o dms_monitor
            dms_monitor.update_settings(dms_settings)
            
            # Atualiza a câmara (brilho e rotação)
            new_brightness = data.get('brightness')
            if new_brightness is not None:
                cam_thread.update_brightness(float(new_brightness))
                
            new_rotation = data.get('rotation')
            if new_rotation is not None:
                cam_thread.update_rotation(int(new_rotation))
                
            logging.info(f"Configurações atualizadas via API: {data}")
            return jsonify({"status": "success", "message": "Configurações aplicadas"})
        except Exception as e:
            logging.error(f"Erro ao aplicar configurações: {e}", exc_info=True)
            return jsonify({"status": "error", "message": "Erro ao aplicar configurações"}), 500
    else:
         return jsonify({"status": "error", "message": "Dados inválidos ou serviço não pronto"}), 400

# --- (NOVO) Rotas para Visualizador de Alertas ---

@app.route("/alerts")
def alerts_page():
    """Serve a nova página HTML para visualizar os alertas."""
    return render_template("alerts.html")

@app.route("/api/alerts", methods=['GET'])
def get_alerts():
    """Lê o ficheiro JSONL e retorna a lista de alertas."""
    alerts_data = []
    try:
        # Usa o caminho definido no event_handler
        log_file = event_handler.log_file_path if event_handler else os.path.join(ALERTS_SAVE_PATH, "alerts_log.jsonl") 
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        alerts_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        logging.warning(f"Ignorando linha inválida no JSONL: {line.strip()}")
        return jsonify(alerts_data)
    except Exception as e:
        logging.error(f"Erro ao ler ficheiro de alertas: {e}", exc_info=True)
        return jsonify({"error": "Falha ao ler o log de alertas"}), 500

@app.route('/alerts/images/<filename>')
def serve_alert_image(filename):
    """Serve os ficheiros de imagem da pasta de alertas."""
    try:
        # Garante segurança: serve apenas ficheiros de ALERTS_SAVE_PATH
        return send_from_directory(ALERTS_SAVE_PATH, filename)
    except FileNotFoundError:
        logging.warning(f"Tentativa de acesso a imagem não encontrada: {filename}")
        return "Imagem não encontrada", 404
    except Exception as e:
        logging.error(f"Erro ao servir imagem {filename}: {e}", exc_info=True)
        return "Erro interno do servidor", 500

# --- Ponto de Entrada Principal ---

if __name__ == '__main__':
    logging.info(">>> Serviço DMS (Refatorado) a iniciar...")
    
    # Verifica se a pasta de alertas existe, se não, cria-a
    if not os.path.exists(ALERTS_SAVE_PATH):
        try:
            os.makedirs(ALERTS_SAVE_PATH)
            logging.info(f"Pasta de alertas criada em: {ALERTS_SAVE_PATH}")
        except OSError as e:
            logging.error(f"!!! ERRO FATAL: Não foi possível criar a pasta de alertas {ALERTS_SAVE_PATH}: {e}")
            sys.exit(1)

    try:
        # 1. Inicia o Gestor de Eventos (precisa da queue e do path)
        event_handler = EventHandler(queue=event_queue, save_path=ALERTS_SAVE_PATH) # Passa o caminho aqui
        event_handler.start()

        # 2. Inicia o DmsCore (precisa do frame_size)
        frame_size = (FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY) 
        dms_monitor = DriverMonitor(frame_size=frame_size)
        
        # 3. Inicia a Thread da Câmara
        cam_thread = CameraThread(VIDEO_SOURCE, 
                                FRAME_WIDTH_DISPLAY, 
                                FRAME_HEIGHT_DISPLAY, 
                                rotation_degrees=ROTATE_FRAME)
        cam_thread.start()
        
        # Espera um pouco para garantir que a câmara conectou e temos o primeiro frame
        logging.info("A aguardar o primeiro frame da câmara...")
        time.sleep(5) # Aumenta a espera inicial
        first_frame = cam_thread.get_frame()
        if first_frame is None:
             # Tenta reconectar ou loga erro mais detalhado
            logging.warning("Primeiro frame ainda não disponível após 5s. A tentar mais 5s...")
            time.sleep(5)
            first_frame = cam_thread.get_frame()
            if first_frame is None:
                 logging.error("!!! ERRO FATAL: Não foi possível obter o primeiro frame da câmara. Verifique a ligação/fonte.")
                 # Decide se quer sair ou continuar tentando
                 # sys.exit(1) # Descomente para sair em caso de falha total
            else:
                 logging.info(">>> Primeiro frame recebido após segunda tentativa!")
        else:
             logging.info(">>> Primeiro frame recebido!")

        # 4. Inicia a Thread de Deteção (só depois de tudo estar pronto)
        if first_frame is not None: # Só inicia se a câmara funcionou
            detection_thread = threading.Thread(target=detection_loop, 
                                                args=(cam_thread, dms_monitor, event_queue))
            detection_thread.daemon = True
            detection_thread.start()
        else:
             logging.warning("Thread de deteção não iniciada devido a falha na obtenção do primeiro frame.")

        # 5. Inicia o Servidor Flask (bloqueia a thread principal)
        logging.info(f">>> A iniciar servidor Flask na porta 5000...")
        # Usa 'threaded=True' para permitir múltiplas conexões
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

    except Exception as e:
        logging.error(f"!!! ERRO FATAL ao iniciar o serviço: {e}", exc_info=True)
    
    finally:
        # --- Rotina de Encerramento ---
        logging.info(">>> A iniciar encerramento do serviço...")
        shutdown_flag.set() # Sinaliza a todas as threads para pararem

        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=2.0) # Espera a thread de deteção terminar
        if cam_thread and cam_thread.is_alive():
            cam_thread.stop()
            cam_thread.join(timeout=2.0) # Espera a thread da câmara terminar
        if event_handler and event_handler.is_alive():
            event_handler.stop()
            event_handler.join(timeout=5.0) # Dá mais tempo ao event_handler para guardar o último evento
            
        logging.info(">>> Serviço DMS terminado.")

