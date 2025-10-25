# Documentação: Thread de Captura de Vídeo
# Esta classe é responsável por ler frames da fonte de vídeo (USB ou RTSP)
# numa thread separada, para não bloquear a aplicação principal.

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
    def __init__(self, video_source_str, frame_width, frame_height):
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
        
        self.connect_camera()

    def connect_camera(self):
        """Tenta (re)conectar-se à fonte de vídeo."""
        logging.info(f">>> A tentar conectar a: {self.source_description}...")
        try:
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.video_source_arg)
            
            # Para câmaras USB, tentamos definir a resolução
            if not self.is_rtsp:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                logging.info(f"Definida resolução da câmara para {self.frame_width}x{self.frame_height}")

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
                        # Se for RTSP, a reconexão é a única solução
                        self.connected = False 
                    else:
                        # Se for USB e falhar, é provável que seja um erro fatal
                        logging.error("Falha ao ler frame da câmara local. A terminar thread.")
                        self.running = False
                    continue
                
                # Redimensiona o frame para o tamanho de exibição padrão
                frame_display = cv2.resize(frame, (self.frame_width, self.frame_height))

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
