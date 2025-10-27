# Documentação: Thread de Captura de Vídeo
# Esta classe é responsável por ler frames da fonte de vídeo (USB ou RTSP)
# numa thread separada, para não bloquear a aplicação principal.
# NOVO: Agora inclui controlo de rotação e brilho.

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
    def __init__(self, video_source_str, frame_width, frame_height, rotation_degrees=0):
        threading.Thread.__init__(self)
        self.daemon = True # Permite que a thread termine quando a app principal fechar
        
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
        
        # --- NOVO: Parâmetros de Controlo ---
        self.rotation_code = None # Código cv2.ROTATE_...
        if rotation_degrees == 90:
            self.rotation_code = cv2.ROTATE_90_CLOCKWISE
            logging.info(">>> ROTAÇÃO DE 90 GRAUS APLICADA.")
        elif rotation_degrees == 180:
            self.rotation_code = cv2.ROTATE_180
            logging.info(">>> ROTAÇÃO DE 180 GRAUS APLICADA.")
        elif rotation_degrees == 270:
            self.rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
            logging.info(">>> ROTAÇÃO DE 270 GRAUS APLICADA.")
            
        # O brilho da câmara é tipicamente 0-255, com 128 como padrão.
        self.brightness = 128 
        
        self.connect_camera()

    def connect_camera(self):
        """Tenta (re)conectar-se à fonte de vídeo."""
        logging.info(f">>> A tentar conectar a: {self.source_description}...")
        try:
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.video_source_arg)
            
            # Para câmaras USB, tentamos definir os parâmetros
            if not self.is_rtsp:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                logging.info(f"Definida resolução da câmara para {self.frame_width}x{self.frame_height}")
                
                # NOVO: Define o brilho inicial
                # Tenta ler o brilho atual, se falhar, usa o padrão 128
                initial_brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                if initial_brightness == -1: # Algumas câmaras não suportam ler
                    initial_brightness = 128
                self.brightness = initial_brightness
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
                logging.info(f"Definido brilho da câmara para {self.brightness}")

            time.sleep(2.0) # Dá tempo à câmara para inicializar

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
                
                # 1. Redimensiona o frame para o tamanho de exibição padrão
                frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))

                # 2. Aplica a rotação (se definida)
                if self.rotation_code is not None:
                    frame_processed = cv2.rotate(frame_resized, self.rotation_code)
                else:
                    frame_processed = frame_resized

                # 3. Atualiza o último frame de forma segura
                with self.lock:
                    self.latest_frame = frame_processed.copy()
            
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

    # --- NOVO: Métodos de Controlo da Câmara ---
    
    def update_brightness(self, brightness_val):
        """
        Atualiza o brilho da câmara em tempo real.
        Chamado pela API do Flask.
        """
        try:
            # Garante que o valor está no intervalo correto (0-255)
            brightness_val = int(brightness_val)
            if brightness_val < 0: brightness_val = 0
            if brightness_val > 255: brightness_val = 255
            
            with self.lock:
                self.brightness = brightness_val
                # Aplica apenas se a câmara estiver ligada e não for RTSP
                if self.cap and self.cap.isOpened() and not self.is_rtsp:
                    self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
                    logging.info(f"Brilho da câmara atualizado para {self.brightness}")
                    
        except Exception as e:
            logging.error(f"Falha ao definir brilho: {e}", exc_info=True)

    def get_brightness(self):
        """
        Retorna o valor de brilho atual.
        Chamado pela API do Flask.
        """
        with self.lock:
            # Tenta ler o valor real da câmara, se possível
            if self.cap and self.cap.isOpened() and not self.is_rtsp:
                val = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                if val != -1: # Se a leitura for suportada
                    self.brightness = int(val)
            return self.brightness

