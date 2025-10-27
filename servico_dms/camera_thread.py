# Documentação: Thread de Captura de Vídeo
# Esta classe é responsável por ler frames da fonte de vídeo (USB ou RTSP)
# numa thread separada, para não bloquear a aplicação principal.
# (Atualizado para incluir rotação e brilho dinâmicos)

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
        
        # --- (NOVO) Controlo de Rotação e Brilho ---
        self.rotation_code = self._get_rotation_code(rotation_degrees)
        self.initial_brightness = float(os.environ.get('BRIGHTNESS', '17.0')) # Valor padrão de brilho

        self.connect_camera()

    def _get_rotation_code(self, degrees):
        """Converte graus num código cv2.ROTATE_*."""
        degrees = int(degrees)
        if degrees == 90:
            return cv2.ROTATE_90_CLOCKWISE
        elif degrees == 180:
            return cv2.ROTATE_180
        elif degrees == 270:
            return cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            return None # Sem rotação

    def connect_camera(self):
        """Tenta (re)conectar-se à fonte de vídeo."""
        logging.info(f">>> A tentar conectar a: {self.source_description}...")
        try:
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.video_source_arg)
            
            # Para câmaras USB, tentamos definir a resolução e o brilho
            if not self.is_rtsp:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                logging.info(f"Definida resolução da câmara para {self.frame_width}x{self.frame_height}")
                
                # Define o brilho inicial
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.initial_brightness)
                logging.info(f"Definido brilho da câmara para {self.initial_brightness}")

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
                
                # Redimensiona o frame para o tamanho de exibição padrão
                frame_display = cv2.resize(frame, (self.frame_width, self.frame_height))

                # --- (NOVO) Aplica Rotação ---
                # A rotação é aplicada *antes* de ser partilhada
                with self.lock:
                    if self.rotation_code is not None:
                        frame_display = cv2.rotate(frame_display, self.rotation_code)
                # -------------------------------

                # Atualiza o último frame de forma segura
                with self.lock:
                    self.latest_frame = frame_display.copy()
            
            except Exception as e:
                logging.error(f"Erro no loop da câmara: {e}", exc_info=True)
                self.connected = False
                time.sleep(1.0) # Evita spam de logs em caso de erro rápido

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

    # --- (NOVO) Métodos de Controlo da API ---

    def update_brightness(self, value):
        """Atualiza o brilho da câmara (se for USB)."""
        if not self.is_rtsp and self.cap:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, float(value))
            logging.info(f"Brilho da câmara atualizado para: {value}")

    def get_brightness(self):
        """Obtém o brilho atual da câmara (se for USB)."""
        if not self.is_rtsp and self.cap:
            return self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
        return self.initial_brightness # Retorna o padrão se for RTSP

    def update_rotation(self, degrees):
        """Atualiza o ângulo de rotação em tempo real."""
        new_code = self._get_rotation_code(degrees)
        with self.lock:
            self.rotation_code = new_code
        logging.info(f"Rotação da câmara atualizada para: {degrees} graus")

    def get_rotation(self):
        """Obtém o código de rotação atual (para a API saber)."""
        with self.lock:
            if self.rotation_code == cv2.ROTATE_90_CLOCKWISE:
                return 90
            elif self.rotation_code == cv2.ROTATE_180:
                return 180
            elif self.rotation_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
                return 270
            else:
                return 0

