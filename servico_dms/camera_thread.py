# Documentação: Thread de Captura de Vídeo Refatorada
# Responsável pela leitura robusta de frames (USB/RTSP) com tipagem e gestão de erros.

import cv2
import numpy as np  # <--- ADICIONADO: Necessário para a tipagem np.ndarray
import threading
import time
import logging
import os
from typing import Optional, Union

# Habilita otimizações do OpenCV (se disponível no hardware)
cv2.setUseOptimized(True)


class CameraThread(threading.Thread):
    """
    Thread dedicada para captura de vídeo.
    Garante que a leitura I/O não bloqueia a thread principal de processamento/web.
    Suporta reconexão automática e ajuste dinâmico de brilho/rotação.
    """

    def __init__(
            self,
            video_source_str: str,
            frame_width: int,
            frame_height: int,
            rotation_degrees: int = 0,
            stop_event: Optional[threading.Event] = None,
    ):
        super().__init__(name="CameraThread")
        self.daemon = True  # A thread morre se o processo principal morrer

        # Identificação da Fonte
        self.is_rtsp: bool = str(video_source_str).startswith("rtsp://")
        self.video_source_arg: Union[str, int] = (
            video_source_str if self.is_rtsp else int(video_source_str)
        )
        self.source_description: str = (
            f"Stream RTSP" if self.is_rtsp else f"USB Cam (Index {self.video_source_arg})"
        )

        # Configurações de Imagem
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Estado Interno
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False
        self.connected = False
        self.stop_event = stop_event or threading.Event()

        # Rotação e Brilho
        self.rotation_code = self._get_rotation_code(rotation_degrees)
        # Tenta ler brilho do ambiente ou usa padrão
        try:
            self.initial_brightness = float(os.environ.get("BRIGHTNESS", "0"))
        except ValueError:
            self.initial_brightness = 0.0

        # Inicia conexão imediatamente
        self._connect_camera()

    def _get_rotation_code(self, degrees: int) -> Optional[int]:
        """Converte graus (0, 90, 180, 270) em constantes do OpenCV."""
        try:
            d = int(degrees)
        except (ValueError, TypeError):
            return None

        if d == 90: return cv2.ROTATE_90_CLOCKWISE
        if d == 180: return cv2.ROTATE_180
        if d == 270: return cv2.ROTATE_90_COUNTERCLOCKWISE
        return None

    def _connect_camera(self) -> None:
        """Gerencia a conexão ou reconexão com a fonte de vídeo."""
        logging.info(f"CamThread: Conectando a {self.source_description}...")

        if self.cap is not None:
            self.cap.release()

        try:
            # Tenta abrir com backend V4L2 explicitamente se for Linux/USB para evitar erros
            if not self.is_rtsp and os.name == 'posix':
                self.cap = cv2.VideoCapture(self.video_source_arg, cv2.CAP_V4L2)
            else:
                self.cap = cv2.VideoCapture(self.video_source_arg)

            if not self.is_rtsp:
                # Configurações de Hardware (apenas USB)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

                # Formato MJPG é mais rápido em USB 2.0
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

                # Define brilho se diferente de zero
                if self.initial_brightness != 0:
                    self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.initial_brightness)

            # Pequena pausa para a câmara "acordar"
            time.sleep(1.0)

            if self.cap.isOpened():
                self.connected = True
                logging.info(f"CamThread: Conectado com sucesso! ({self.frame_width}x{self.frame_height})")
            else:
                self.connected = False
                logging.error(f"CamThread: Falha ao abrir {self.video_source_arg}")

        except Exception as e:
            logging.error(f"CamThread: Erro crítico na conexão: {e}")
            self.connected = False

    def run(self) -> None:
        """Loop principal da thread."""
        self.running = True
        logging.info("CamThread: Loop iniciado.")

        while self.running and not self.stop_event.is_set():
            # Gestão de Reconexão
            if not self.connected or self.cap is None or not self.cap.isOpened():
                logging.warning("CamThread: Câmara desconectada. Tentando reconectar em 3s...")
                time.sleep(3)
                self._connect_camera()
                continue

            try:
                ret, frame = self.cap.read()

                if not ret:
                    logging.warning("CamThread: Frame vazio/corrompido.")
                    self.connected = False  # Força reconexão no próximo loop
                    continue

                # Processamento Básico (Resize e Rotação)
                # Nota: Resize aqui garante consistência, mas gasta CPU.
                # Se a câmara já enviar no tamanho certo (definido no _connect), o resize é rápido.
                if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                # Aplica rotação thread-safe
                with self.lock:
                    rot_code = self.rotation_code

                if rot_code is not None:
                    frame = cv2.rotate(frame, rot_code)

                # Atualiza o frame publicável
                with self.lock:
                    self.latest_frame = frame.copy()

                # Controla o FPS da captura para não saturar a CPU (Sleep leve)
                time.sleep(0.005)

            except Exception as e:
                logging.error(f"CamThread: Erro no loop de leitura: {e}")
                self.connected = False
                time.sleep(1)

        # Fim do loop
        if self.cap:
            self.cap.release()
        logging.info("CamThread: Encerrado.")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Retorna uma cópia do último frame válido ou None.
        Thread-safe.
        """
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def update_rotation(self, degrees: int) -> None:
        """Atualiza a rotação dinamicamente."""
        new_code = self._get_rotation_code(degrees)
        with self.lock:
            self.rotation_code = new_code
        logging.info(f"CamThread: Rotação alterada para {degrees}°")

    def update_brightness(self, value: float) -> None:
        """Atualiza o brilho (Apenas USB)."""
        if self.is_rtsp or not self.connected:
            return

        try:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, float(value))
        except Exception as e:
            logging.warning(f"CamThread: Erro ao ajustar brilho: {e}")

    def get_brightness(self) -> float:
        """Lê o brilho atual (Apenas USB)."""
        if self.is_rtsp or not self.connected:
            return 0.0
        try:
            return self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
        except:
            return 0.0

    def get_rotation(self) -> int:
        """Retorna o ângulo atual em graus."""
        with self.lock:
            c = self.rotation_code

        if c == cv2.ROTATE_90_CLOCKWISE: return 90
        if c == cv2.ROTATE_180: return 180
        if c == cv2.ROTATE_90_COUNTERCLOCKWISE: return 270
        return 0