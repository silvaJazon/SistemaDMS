# Documentação: Thread de Captura de Vídeo
# Responsável por ler frames da fonte de vídeo numa thread separada.
# AGORA INCLUI: Controlo de Brilho e Rotação em tempo real.

import cv2
import threading
import time
import logging
import sys
import os

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

class CameraThread(threading.Thread):
    """
    Classe que gere a conexão da câmara numa thread dedicada.
    """
    def __init__(self, video_source_str, frame_width, frame_height, initial_rotation=0):
        threading.Thread.__init__(self)
        self.daemon = True
        
        self.is_rtsp = video_source_str.startswith("rtsp://")
        self.source_description = f"stream de rede: {video_source_str}" if self.is_rtsp else f"câmara local no índice: {video_source_str}"
        self.video_source_arg = video_source_str if self.is_rtsp else int(video_source_str)
        
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.connected = False
        
        # --- Controlo da Câmara ---
        # Define o brilho inicial (1-100). 50 é o padrão.
        # câmaras IR podem precisar de valores mais baixos.
        self.brightness = 17.0 
        
        # Mapeamento de rotação (NOVO)
        self.rotation_degrees = initial_rotation
        self.rotation_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        if self.rotation_degrees != 0:
            logging.info(f">>> ROTAÇÃO DE {self.rotation_degrees} GRAUS APLICADA.")
            
        self.connect_camera()

    def connect_camera(self):
        """Tenta (re)conectar-se à fonte de vídeo."""
        logging.info(f">>> A tentar conectar a: {self.source_description}...")
        try:
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.video_source_arg)
            
            if not self.is_rtsp:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                logging.info(f"Definida resolução da câmara para {self.frame_width}x{self.frame_height}")
                
                # Aplica o brilho (convertido de 0-100 para 0.0-1.0 se necessário, mas OpenCV usa 0-255)
                # Vamos assumir que a API envia 0-100, mapeamos para 0-255.
                # Atualização: a propriedade V4L2 varia (ex: 0-100, 0-255). Vamos usar o valor direto.
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
                logging.info(f"Definido brilho da câmara para {self.brightness}")


            time.sleep(2.0) 

            if not self.cap.isOpened():
                logging.error(f"!!! ERRO: Não foi possível abrir a fonte de vídeo: {self.video_source_arg}")
                self.connected = False
            else:
                logging.info(">>> Fonte de vídeo conectada com sucesso!")
                self.connected = True
                
        except Exception as e:
            logging.error(f"Exceção ao conectar à câmara: {e}", exc_info=True)
            self.connected = False

    def run(self):
        """O loop principal da thread: ler frames continuamente."""
        self.running = True
        
        while self.running:
            if not self.connected or not self.cap or not self.cap.isOpened():
                logging.warning("!!! Ligação de vídeo perdida. A tentar reconectar em 5s...")
                self.connect_camera()
                time.sleep(5.0)
                continue

            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logging.warning("!!! Frame não recebido. A verificar ligação...")
                    if self.is_rtsp:
                        self.connected = False 
                    else:
                        logging.error("Falha ao ler frame da câmara local. A terminar thread.")
                        self.running = False
                    continue
                
                # Redimensiona o frame para o tamanho de exibição padrão
                frame_display = cv2.resize(frame, (self.frame_width, self.frame_height))

                # --- (NOVO) Aplica Rotação ---
                # Obtém o código de rotação (ex: cv2.ROTATE_180)
                rotation_code = self.rotation_map.get(self.rotation_degrees)
                if rotation_code is not None:
                    frame_display = cv2.rotate(frame_display, rotation_code)
                # ------------------------------

                # Atualiza o último frame de forma segura
                with self.lock:
                    self.latest_frame = frame_display.copy()
            
            except Exception as e:
                logging.error(f"Erro no loop da câmara: {e}", exc_info=True)
                self.connected = False
                time.sleep(1.0) 

        logging.info(">>> Thread da câmara terminada.")
        if self.cap:
            self.cap.release()

    def get_frame(self):
        """Retorna o último frame capturado de forma segura."""
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        """Sinaliza à thread para parar."""
        self.running = False

    # --- Métodos de Controlo (Chamados pela API) ---

    def update_brightness(self, brightness_level):
        """Atualiza o brilho da câmara em tempo real."""
        try:
            self.brightness = float(brightness_level)
            if self.cap and not self.is_rtsp:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
                logging.info(f"Brilho da câmara atualizado para: {self.brightness}")
        except Exception as e:
            logging.error(f"Falha ao definir brilho: {e}")

    def get_brightness(self):
        """Obtém o valor de brilho atual (da nossa variável)."""
        return self.brightness

    def update_rotation(self, degrees):
        """Atualiza a rotação em tempo real."""
        try:
            self.rotation_degrees = int(degrees)
            logging.info(f"Rotação da câmara atualizada para: {self.rotation_degrees} graus")
        except Exception as e:
            logging.error(f"Falha ao definir rotação: {e}")

    def get_rotation(self):
        """Obtém o valor de rotação atual."""
        return self.rotation_degrees

