# Documentação: Thread de Captura de Vídeo (Com Auto-Busca de Dispositivo)

import cv2
import numpy as np
import threading
import time
import logging
import os
from typing import Optional, Union, List
from collections import deque

cv2.setUseOptimized(True)


class CameraThread(threading.Thread):
    def __init__(
            self,
            video_source_str: str,
            frame_width: int,
            frame_height: int,
            rotation_degrees: int = 0,
            stop_event: Optional[threading.Event] = None,
    ):
        super().__init__(name="CameraThread")
        self.daemon = True

        self.is_rtsp: bool = str(video_source_str).startswith("rtsp://")
        # Se for número, guarda como int, mas vamos ignorar se precisarmos de procurar
        self.video_source_arg: Union[str, int] = (
            video_source_str if self.is_rtsp else int(video_source_str)
        )
        self.source_description: str = (
            f"Stream RTSP" if self.is_rtsp else f"USB Cam (Index {self.video_source_arg})"
        )

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.BUFFER_SIZE = 100
        self.frame_buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self.buffer_lock = threading.Lock()

        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False
        self.connected = False
        self.stop_event = stop_event or threading.Event()

        self.rotation_code = self._get_rotation_code(rotation_degrees)
        try:
            self.initial_brightness = float(os.environ.get("BRIGHTNESS", "0"))
        except ValueError:
            self.initial_brightness = 0.0

        self._connect_camera()

    def _get_rotation_code(self, degrees: int) -> Optional[int]:
        try:
            d = int(degrees)
        except:
            return None
        if d == 90: return cv2.ROTATE_90_CLOCKWISE
        if d == 180: return cv2.ROTATE_180
        if d == 270: return cv2.ROTATE_90_COUNTERCLOCKWISE
        return None

    def _connect_camera(self) -> None:
        """Tenta conectar. Se falhar, procura noutros índices USB."""
        logging.info(f"CamThread: A tentar conectar...")

        if self.cap is not None:
            self.cap.release()

        # Se for RTSP, tenta direto
        if self.is_rtsp:
            self._try_open(self.video_source_arg)
            return

        # Se for USB, tenta o índice original primeiro
        if self._try_open(self.video_source_arg):
            return

        # Se falhou, tenta procurar outros índices (0 a 9)
        logging.warning("CamThread: Falha no índice original. A procurar câmaras alternativas...")
        for i in range(10):
            if i == self.video_source_arg: continue  # Já tentámos este
            logging.info(f"CamThread: Tentando /dev/video{i}...")
            if self._try_open(i):
                logging.info(f"CamThread: SUCESSO! Câmara encontrada no índice {i}.")
                self.video_source_arg = i  # Atualiza para o futuro
                return

        logging.error("CamThread: Nenhuma câmara encontrada após varredura.")
        self.connected = False

    def _try_open(self, source) -> bool:
        try:
            if not self.is_rtsp and os.name == 'posix':
                cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            else:
                cap = cv2.VideoCapture(source)

            if not self.is_rtsp:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                if self.initial_brightness != 0:
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, self.initial_brightness)

            # Teste real de leitura
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    self.connected = True
                    return True

            cap.release()
            return False
        except Exception:
            return False

    def run(self) -> None:
        self.running = True
        while self.running and not self.stop_event.is_set():
            if not self.connected or self.cap is None or not self.cap.isOpened():
                time.sleep(5)  # Espera 5s antes de tentar reconectar/procurar
                self._connect_camera()
                continue

            try:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("CamThread: Perda de sinal. Reconectando...")
                    self.connected = False
                    continue

                if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                with self.lock:
                    rot_code = self.rotation_code
                if rot_code is not None: frame = cv2.rotate(frame, rot_code)

                with self.buffer_lock:
                    self.frame_buffer.append(frame)
                with self.lock:
                    self.latest_frame = frame.copy()

                time.sleep(0.005)

            except Exception as e:
                logging.error(f"CamThread: Erro loop: {e}")
                self.connected = False
                time.sleep(1)

        if self.cap: self.cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is not None: return self.latest_frame.copy()
        return None

    def get_recent_frames(self) -> List[np.ndarray]:
        with self.buffer_lock: return list(self.frame_buffer)

    def update_rotation(self, degrees: int) -> None:
        new_code = self._get_rotation_code(degrees)
        with self.lock: self.rotation_code = new_code

    def update_brightness(self, value: float) -> None:
        if not self.is_rtsp and self.connected:
            try:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, float(value))
            except:
                pass

    def get_brightness(self) -> float:
        try:
            return self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
        except:
            return 0.0

    def get_rotation(self) -> int:
        with self.lock:
            c = self.rotation_code
        if c == cv2.ROTATE_90_CLOCKWISE: return 90
        if c == cv2.ROTATE_180: return 180
        if c == cv2.ROTATE_90_COUNTERCLOCKWISE: return 270
        return 0