# Documentação: Thread de Captura de Vídeo
# Esta classe é responsável por ler frames da fonte de vídeo (USB ou RTSP)
# numa thread separada, para não bloquear a aplicação principal.
# (Atualizado para incluir rotação e brilho dinâmicos e logs DEBUG)

import cv2
import threading
import time
import logging
# import sys (F401 - Removido)
import os

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)


class CameraThread(threading.Thread):
    """
    Classe que gere a conexão da câmara numa thread dedicada.
    """

    def __init__(
        self,
        video_source_str,
        frame_width,
        frame_height,
        rotation_degrees=0,
        stop_event=None,
    ):
        threading.Thread.__init__(self, name="CameraThread")
        self.daemon = True  # Permite que a thread termine quando a app principal fechar

        self.is_rtsp = video_source_str.startswith("rtsp://")
        self.source_description = (
            f"stream de rede: {video_source_str}"
            if self.is_rtsp
            else f"câmara local no índice: {video_source_str}"
        )
        self.video_source_arg = (
            video_source_str if self.is_rtsp else int(video_source_str)
        )

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.connected = False
        self.stop_event = stop_event or threading.Event()

        # --- Controlo de Rotação e Brilho ---
        self.rotation_code = self._get_rotation_code(rotation_degrees)
        self.initial_brightness = float(
            os.environ.get("BRIGHTNESS", "17.0")
        )

        self.connect_camera()

    def _get_rotation_code(self, degrees):
        """Converte graus num código cv2.ROTATE_*."""
        try:
            degrees = int(degrees)
        except (ValueError, TypeError):
            logging.warning(f"Valor de rotação inválido '{degrees}'. Usando 0.")
            degrees = 0

        if degrees == 90:
            return cv2.ROTATE_90_CLOCKWISE
        elif degrees == 180:
            return cv2.ROTATE_180
        elif degrees == 270:
            return cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            return None  # Sem rotação

    def connect_camera(self):
        """Tenta (re)conectar-se à fonte de vídeo."""
        logging.info(f">>> A tentar conectar a: {self.source_description}...")
        try:
            if self.cap:
                logging.debug("CameraThread: A libertar captura anterior...")
                self.cap.release()

            logging.debug(
                f"CameraThread: A chamar cv2.VideoCapture({self.video_source_arg})..."
            )
            self.cap = cv2.VideoCapture(self.video_source_arg)

            if not self.is_rtsp:
                logging.debug(
                    f"CameraThread: A definir resolução {self.frame_width}x{self.frame_height}..."
                )
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                logging.info(
                    f"Definida resolução da câmara para {self.frame_width}x{self.frame_height}"
                )

                logging.debug(
                    f"CameraThread: A definir brilho inicial {self.initial_brightness}..."
                )
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.initial_brightness)
                logging.info(f"Definido brilho da câmara para {self.initial_brightness}")

            logging.debug("CameraThread: A aguardar 2s pela inicialização da câmara...")
            time.sleep(2.0)

            if not self.cap or not self.cap.isOpened():
                logging.error(
                    f"!!! ERRO: Não foi possível abrir a fonte de vídeo: {self.video_source_arg}"
                )
                self.connected = False
            else:
                logging.info(">>> Fonte de vídeo conectada com sucesso!")
                self.connected = True

        except Exception as e:
            logging.error(f"Exceção ao conectar à câmara: {e}", exc_info=True)
            self.connected = False
            if self.cap:  # Garante que libertamos mesmo em caso de erro
                self.cap.release()
                self.cap = None

    def run(self):
        """O loop principal da thread: ler frames continuamente."""
        self.running = True
        frame_read_count = 0

        logging.info(">>> Thread da câmara iniciada.")

        while self.running and not self.stop_event.is_set():
            if not self.connected or not self.cap or not self.cap.isOpened():
                logging.warning(
                    "!!! Ligação de vídeo perdida ou não estabelecida. "
                    "A tentar reconectar em 5s..."
                )
                self.connect_camera()
                self.stop_event.wait(timeout=5.0)
                continue

            try:
                logging.debug("CameraThread: A chamar self.cap.read()...")
                ret, frame = self.cap.read()
                logging.debug(f"CameraThread: self.cap.read() retornou {ret}.")

                if not ret:
                    logging.warning("!!! Frame não recebido (ret=False). A verificar ligação...")
                    if self.is_rtsp:
                        logging.warning(
                            "CameraThread: Fonte RTSP pode ter caído. A tentar reconectar."
                        )
                        self.connected = False
                        self.stop_event.wait(timeout=2.0)
                    else:
                        logging.error(
                            "Falha ao ler frame da câmara local. A terminar thread."
                        )
                        self.running = False
                    continue

                frame_display = cv2.resize(frame, (self.frame_width, self.frame_height))

                # --- Aplica Rotação ---
                current_rotation_code = None
                logging.debug("CameraThread: A adquirir lock para rotação...")
                with self.lock:
                    logging.debug("CameraThread: Lock adquirido para rotação.")
                    current_rotation_code = self.rotation_code
                logging.debug("CameraThread: Lock libertado para rotação.")

                if current_rotation_code is not None:
                    logging.debug(
                        f"CameraThread: A aplicar rotação {current_rotation_code}..."
                    )
                    frame_display = cv2.rotate(frame_display, current_rotation_code)
                # -------------------------------

                # Atualiza o último frame de forma segura
                logging.debug(
                    "CameraThread: A adquirir lock para atualizar latest_frame..."
                )
                with self.lock:
                    logging.debug("CameraThread: Lock adquirido para latest_frame.")
                    self.latest_frame = frame_display.copy()
                logging.debug("CameraThread: Lock libertado para latest_frame.")

                frame_read_count += 1
                if frame_read_count % 100 == 0:
                    logging.debug(
                        f"CameraThread: Frame {frame_read_count} lido e atualizado com sucesso."
                    )

                self.stop_event.wait(timeout=0.01)

            except cv2.error as cv_err:
                logging.error(f"Erro OpenCV no loop da câmara: {cv_err}", exc_info=True)
                self.connected = False
                self.stop_event.wait(timeout=1.0)
            except Exception as e:
                logging.error(f"Erro inesperado no loop da câmara: {e}", exc_info=True)
                self.connected = False
                self.stop_event.wait(timeout=1.0)

        logging.info(
            f">>> Thread da câmara a terminar após ler {frame_read_count} frames."
        )
        if self.cap:
            logging.debug("CameraThread: A libertar self.cap...")
            self.cap.release()
            logging.debug("CameraThread: self.cap libertado.")

    def get_frame(self):
        """Retorna o último frame capturado de forma segura."""
        logging.debug("CameraThread: get_frame() chamado.")
        frame_copy = None
        logging.debug("CameraThread: get_frame() a adquirir lock...")
        with self.lock:
            logging.debug("CameraThread: get_frame() lock adquirido.")
            if self.latest_frame is not None:
                frame_copy = self.latest_frame.copy()
        logging.debug(
            f"CameraThread: get_frame() lock libertado. "
            f"Retornando {'um frame' if frame_copy is not None else 'None'}."
        )
        return frame_copy

    def stop(self):
        """Sinaliza à thread para parar."""
        logging.info("CameraThread: Sinal de paragem recebido.")
        self.running = False
        self.stop_event.set()

    # --- Métodos de Controlo da API ---

    def update_brightness(self, value):
        """Atualiza o brilho da câmara (se for USB)."""
        if not self.is_rtsp and self.cap and self.connected:
            try:
                success = self.cap.set(cv2.CAP_PROP_BRIGHTNESS, float(value))
                if success:
                    logging.info(f"Brilho da câmara atualizado para: {value}")
                else:
                    logging.warning(f"Falha ao definir brilho da câmara para: {value}")
            except Exception as e:
                logging.error(f"Erro ao definir brilho da câmara: {e}")
        elif self.is_rtsp:
            logging.debug("Ajuste de brilho não suportado para RTSP.")
        else:
            logging.warning("Tentativa de ajustar brilho sem câmara conectada.")

    def get_brightness(self):
        """Obtém o brilho atual da câmara (se for USB)."""
        if not self.is_rtsp and self.cap and self.connected:
            try:
                brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                logging.debug(f"CameraThread: Brilho atual lido: {brightness}")
                return brightness
            except Exception as e:
                logging.error(f"Erro ao obter brilho da câmara: {e}")
                return self.initial_brightness  # Retorna padrão em caso de erro
        return self.initial_brightness

    def update_rotation(self, degrees):
        """Atualiza o ângulo de rotação em tempo real."""
        new_code = self._get_rotation_code(degrees)
        logging.debug("CameraThread: update_rotation() a adquirir lock...")
        with self.lock:
            logging.debug("CameraThread: update_rotation() lock adquirido.")
            self.rotation_code = new_code
        logging.debug("CameraThread: update_rotation() lock libertado.")
        logging.info(f"Rotação da câmara atualizada para: {degrees} graus")

    def get_rotation(self):
        """Obtém o ângulo de rotação atual em graus."""
        current_code = None
        logging.debug("CameraThread: get_rotation() a adquirir lock...")
        with self.lock:
            logging.debug("CameraThread: get_rotation() lock adquirido.")
            current_code = self.rotation_code
        logging.debug("CameraThread: get_rotation() lock libertado.")

        if current_code == cv2.ROTATE_90_CLOCKWISE:
            return 90
        elif current_code == cv2.ROTATE_180:
            return 180
        elif current_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
            return 270
        else:
            return 0