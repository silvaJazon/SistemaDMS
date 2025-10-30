# Documentação: Núcleo do SistemaDMS (Implementação MediaPipe)
# Focada em Sonolência (EAR) e Bocejo (MAR)

import cv2
import mediapipe as mp
import numpy as np
import logging
import math
import threading
from scipy.spatial import distance as dist
from datetime import datetime
import time
from collections import deque

from dms_base import BaseMonitor

cv2.setUseOptimized(True)

# --- Constantes de Índices do MediaPipe Face Mesh (478 landmarks) ---
MP_LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE_IDX = [362, 385, 387, 263, 380, 373]
MP_MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 87]

class MediaPipeMonitor(BaseMonitor):
    """
    Implementação do BaseMonitor usando MediaPipe Face Mesh.
    Foco: EAR (Sonolência) e MAR (Bocejo).
    A deteção de distração/celular NÃO está implementada nesta versão.
    """

    def __init__(self, frame_size, default_settings: dict = None):
        super().__init__(frame_size, default_settings)
        logging.info("A inicializar o MediaPipeMonitor Core (Modo: EAR + MAR)...")

        # Inicializa o MediaPipe FaceMesh
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logging.info(">>> Modelos MediaPipe FaceMesh carregados.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL MediaPipe: {e}", exc_info=True)
            raise RuntimeError(f"Erro MediaPipe: {e}")

        self.lock = threading.Lock()
        self.drowsiness_counter = 0
        self.yawn_counter = 0
        self.drowsy_alert_active = False
        self.yawn_alert_active = False

        # Configurações (Padrão) - Mantém a estrutura completa
        self.ear_threshold = self.default_settings.get('ear_threshold', 0.25)
        self.ear_frames = self.default_settings.get('ear_frames', 7)
        self.mar_threshold = self.default_settings.get('mar_threshold', 0.40)
        self.mar_frames = self.default_settings.get('mar_frames', 10)
        
        # (ALTERADO) Configurações de Deteção de Celular (desativadas por implementação)
        self.phone_detection_enabled = False
        self.phone_confidence = 0.50
        self.phone_frames = 20

    # ... (Funções _eye_aspect_ratio, _mouth_aspect_ratio, _get_landmarks_from_result permanecem iguais) ...
    def _eye_aspect_ratio(self, eye_landmarks):
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5]); B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        return 0.3 if C < 1e-6 else (A + B) / (2.0 * C)
    def _mouth_aspect_ratio(self, mouth_landmarks):
        A = dist.euclidean(mouth_landmarks[1], mouth_landmarks[7]); B = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])
        C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])
        return 0.0 if C < 1e-6 else (A + B) / (2.0 * C)
    def _get_landmarks_from_result(self, landmarks, indices):
        coords = np.zeros((len(indices), 2), dtype="int")
        for i, idx in enumerate(indices):
            lm = landmarks[idx]
            coords[i] = (int(lm.x * self.frame_width), int(lm.y * self.frame_height))
        return coords


    def process_frame(self, frame, gray):
        """Analisa um frame (MediaPipe EAR + MAR)."""
        logging.debug("DMSCore(MediaPipe): process_frame (EAR + MAR) iniciado.")
        start_time_total = time.time()
        events_list = []
        status_data = {"ear": "-", "mar": "-", "yaw": "-", "pitch": "-", "roll": "-"}
        face_found_this_frame = False

        # MediaPipe espera frames RGB
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
        except cv2.error as e:
            logging.error(f"DMSCore(MediaPipe): Erro ao converter BGR->RGB: {e}")
            return frame, events_list, status_data
        except Exception as e:
            logging.error(f"DMSCore(MediaPipe): Erro .process(): {e}", exc_info=True)
            return frame, events_list, status_data

        logging.debug(f"DMSCore(MediaPipe): Processamento MP {time.time() - start_time_total:.4f}s.")

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            face_found_this_frame = True

            try:
                left_eye_pts = self._get_landmarks_from_result(face_landmarks, MP_LEFT_EYE_IDX)
                right_eye_pts = self._get_landmarks_from_result(face_landmarks, MP_RIGHT_EYE_IDX)
                mouth_pts = self._get_landmarks_from_result(face_landmarks, MP_MOUTH_IDX)
                
                ear_left = self._eye_aspect_ratio(left_eye_pts); ear_right = self._eye_aspect_ratio(right_eye_pts)
                ear = (ear_left + ear_right) / 2.0; logging.debug(f"DMSCore(MediaPipe): EAR={ear:.3f}")
                mar = self._mouth_aspect_ratio(mouth_pts); logging.debug(f"DMSCore(MediaPipe): MAR={mar:.3f}")

                status_data["ear"] = f"{ear:.2f}"; status_data["mar"] = f"{mar:.2f}"

                cv2.drawContours(frame, [cv2.convexHull(left_eye_pts)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye_pts)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(mouth_pts)], -1, (0, 255, 255), 1)

            except Exception as e:
                logging.error(f"DMSCore(MediaPipe): Erro ao processar landmarks: {e}", exc_info=True)
                face_found_this_frame = False

        if face_found_this_frame:
            logging.debug("DMSCore(MediaPipe): Lock alerta...")
            with self.lock:
                logging.debug("DMSCore(MediaPipe): Lock alerta OK.")
                
                # Sonolência
                if ear < self.ear_threshold:
                    self.drowsiness_counter += 1
                    logging.debug(f"DMSCore(MediaPipe): EAR baixo ({ear:.3f}<{self.ear_threshold}), cont={self.drowsiness_counter}/{self.ear_frames}")
                    if self.drowsiness_counter >= self.ear_frames and not self.drowsy_alert_active:
                        self.drowsy_alert_active = True
                        events_list.append({"type": "SONOLENCIA", "value": f"EAR: {ear:.2f}", "timestamp": datetime.now().isoformat() + "Z"})
                        logging.warning("DMSCore(MediaPipe): EVENTO SONOLENCIA.")
                else:
                    if self.drowsiness_counter > 0: logging.debug("DMSCore(MediaPipe): Sonolência reset.")
                    self.drowsiness_counter = 0
                    self.drowsy_alert_active = False

                # Bocejo
                if mar > self.mar_threshold:
                    self.yawn_counter += 1
                    logging.debug(f"DMSCore(MediaPipe): MAR alto ({mar:.3f}>{self.mar_threshold}), cont={self.yawn_counter}/{self.mar_frames}")
                    if self.yawn_counter >= self.mar_frames and not self.yawn_alert_active:
                        self.yawn_alert_active = True
                        events_list.append({"type": "BOCEJO", "value": f"MAR: {mar:.2f}", "timestamp": datetime.now().isoformat() + "Z"})
                        logging.warning("DMSCore(MediaPipe): EVENTO BOCEJO.")
                else:
                    if self.yawn_counter > 0: logging.debug("DMSCore(MediaPipe): Bocejo reset.")
                    self.yawn_counter = 0
                    self.yawn_alert_active = False
                
                # (REMOVIDO) Lógica de distração

            logging.debug("DMSCore(MediaPipe): Lock alerta libertado.")

            if self.drowsy_alert_active:
                cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.yawn_alert_active:
                cv2.putText(frame, "ALERTA: BOCEJO!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        else:
            logging.debug("DMSCore(MediaPipe): Nenhuma face encontrada.")
            with self.lock:
                self.drowsiness_counter = 0; self.drowsy_alert_active = False
                self.yawn_counter = 0; self.yawn_alert_active = False
                # (REMOVIDO) Reset de distração

        total_time = time.time() - start_time_total
        logging.debug(f"DMSCore(MediaPipe): process_frame (EAR + MAR) {total_time:.4f}s.")
        return frame, events_list, status_data

    def update_settings(self, settings):
        """Atualiza configurações."""
        logging.debug(f"DMSCore(MediaPipe): Tentando atualizar conf: {settings}")
        with self.lock:
            try:
                self.ear_threshold = float(settings.get('ear_threshold', self.ear_threshold))
                self.ear_frames = int(settings.get('ear_frames', self.ear_frames))
                self.mar_threshold = float(settings.get('mar_threshold', self.mar_threshold))
                self.mar_frames = int(settings.get('mar_frames', self.mar_frames))
                
                # (ALTERADO) Ignora settings de celular
                new_phone_state = bool(settings.get('phone_detection_enabled', self.phone_detection_enabled))
                if new_phone_state:
                    logging.warning("DMSCore(MediaPipe): Tentativa de ativar deteção de celular, mas não está implementada neste backend.")
                
                # Força a deteção de celular para False
                self.phone_detection_enabled = False
                # (Atualiza valores internos mesmo que desativado)
                self.phone_confidence = float(settings.get('phone_confidence', self.phone_confidence))
                self.phone_frames = int(settings.get('phone_frames', self.phone_frames))


                logging.info(f"Conf DMS Core(MediaPipe): EAR<{self.ear_threshold}({self.ear_frames}f), MAR>{self.mar_threshold}({self.mar_frames}f), Celular: DESATIVADA (Não implementada)")
                return True
            except (ValueError, TypeError) as e:
                logging.error(f"Erro conf MediaPipe (valor inválido?): {e}"); return False
            except Exception as e:
                logging.error(f"Erro inesperado conf MediaPipe: {e}", exc_info=True); return False

    def get_settings(self):
        """Obtém configurações."""
        logging.debug("DMSCore(MediaPipe): get_settings.")
        with self.lock:
            # (ALTERADO) Retorna a nova estrutura de settings
            return {"ear_threshold": self.ear_threshold, "ear_frames": self.ear_frames,
                    "mar_threshold": self.mar_threshold, "mar_frames": self.mar_frames,
                    
                    "phone_detection_enabled": self.phone_detection_enabled, # Sempre False
                    "phone_confidence": self.phone_confidence,
                    "phone_frames": self.phone_frames,
                   }