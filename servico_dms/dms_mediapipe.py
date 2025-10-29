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

# Índices para o cálculo do EAR (6 pontos por olho)
# Formato [P1, P2, P3, P4, P5, P6]
# P1 (canto horizontal), P4 (canto horizontal)
# P2, P3 (pálpebra superior)
# P6, P5 (pálpebra inferior)
MP_LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE_IDX = [362, 385, 387, 263, 380, 373]

# Índices para o cálculo do MAR (8 pontos, equivalentes ao Dlib 60-67)
# P60(78), P61(81), P62(13), P63(311), P64(308), P65(402), P66(14), P67(87)
MP_MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 87]

class MediaPipeMonitor(BaseMonitor):
    """
    Implementação do BaseMonitor usando MediaPipe Face Mesh.
    Foco: EAR (Sonolência) e MAR (Bocejo).
    A deteção de pose/distração NÃO está implementada nesta versão.
    """

    # ================== ALTERAÇÃO (Padrões Centralizados) ==================
    def __init__(self, frame_size, default_settings: dict = None):
        super().__init__(frame_size, default_settings) # (ALTERADO)
        logging.info("A inicializar o MediaPipeMonitor Core (Modo: EAR + MAR)...")
    # ===================================================================

        # Inicializa o MediaPipe FaceMesh
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,  # Refina landmarks dos olhos e lábios
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

        # ================== ALTERAÇÃO (Padrões Centralizados) ==================
        # Define os padrões de EAR/MAR a partir do dict (vindo do app.py)
        # ou usa padrões 'de emergência' (fallbacks) se não forem fornecidos.
        self.ear_threshold = self.default_settings.get('ear_threshold', 0.25)
        self.ear_frames = self.default_settings.get('ear_frames', 7) # Fallback
        self.mar_threshold = self.default_settings.get('mar_threshold', 0.40) # Fallback
        self.mar_frames = self.default_settings.get('mar_frames', 10) # Fallback
        # =======================================================================
        
        # A distração está desativada por *implementação*
        self.distraction_detection_enabled = False
        self.distraction_angle = 40.0
        self.distraction_frames = 35
        self.pitch_down_offset = 20.0
        self.pitch_up_threshold = -15.0
        self.yaw_center_offset = 0.0
        self.pitch_center_offset = 0.0

    def _eye_aspect_ratio(self, eye_landmarks):
        """Calcula EAR (Eye Aspect Ratio) a partir de 6 landmarks."""
        # P2-P6
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        # P3-P5
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        # P1-P4
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Evita divisão por zero se o olho estiver perfeitamente horizontal
        return 0.3 if C < 1e-6 else (A + B) / (2.0 * C)

    def _mouth_aspect_ratio(self, mouth_landmarks):
        """Calcula MAR (Mouth Aspect Ratio) a partir de 8 landmarks (Dlib 60-67)."""
        # P62-P68 (81, 87)
        A = dist.euclidean(mouth_landmarks[1], mouth_landmarks[7])
        # P63-P67 (13, 14)
        B = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])
        A = dist.euclidean(mouth_landmarks[1], mouth_landmarks[7])
        # Dlib P62-P66 (MP 13-14)
        B = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])
        # Dlib P60-P64 (MP 78-308)
        C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])
        
        return 0.0 if C < 1e-6 else (A + B) / (2.0 * C)

    def _get_landmarks_from_result(self, landmarks, indices):
        """Extrai coordenadas (x, y) de landmarks específicas."""
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
            return frame, events_list, status_data # Retorna frame original
        except Exception as e:
            logging.error(f"DMSCore(MediaPipe): Erro .process(): {e}", exc_info=True)
            return frame, events_list, status_data

        logging.debug(f"DMSCore(MediaPipe): Processamento MP {time.time() - start_time_total:.4f}s.")

        if results.multi_face_landmarks:
            # Assume apenas uma cara (configurado no init)
            face_landmarks = results.multi_face_landmarks[0].landmark
            face_found_this_frame = True

            # --- Extrair Landmarks ---
            try:
                left_eye_pts = self._get_landmarks_from_result(face_landmarks, MP_LEFT_EYE_IDX)
                right_eye_pts = self._get_landmarks_from_result(face_landmarks, MP_RIGHT_EYE_IDX)
                mouth_pts = self._get_landmarks_from_result(face_landmarks, MP_MOUTH_IDX)
                
                # --- Calcular EAR e MAR ---
                ear_left = self._eye_aspect_ratio(left_eye_pts)
                ear_right = self._eye_aspect_ratio(right_eye_pts)
                ear = (ear_left + ear_right) / 2.0
                logging.debug(f"DMSCore(MediaPipe): EAR={ear:.3f}")
                
                mar = self._mouth_aspect_ratio(mouth_pts)
                logging.debug(f"DMSCore(MediaPipe): MAR={mar:.3f}")

                status_data["ear"] = f"{ear:.2f}"
                status_data["mar"] = f"{mar:.2f}"

                # --- Desenho (para depuração) ---
                cv2.drawContours(frame, [cv2.convexHull(left_eye_pts)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye_pts)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(mouth_pts)], -1, (0, 255, 255), 1)

            except Exception as e:
                logging.error(f"DMSCore(MediaPipe): Erro ao processar landmarks: {e}", exc_info=True)
                face_found_this_frame = False


        # --- Lógica de Alerta ---
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
                
                # Distração (Sempre resetado, pois não é implementado)
                if self.distraction_counter > 0 or self.distraction_alert_active:
                    self.distraction_counter = 0
                    self.distraction_alert_active = False

            logging.debug("DMSCore(MediaPipe): Lock alerta libertado.")

            # Desenha alertas
            if self.drowsy_alert_active:
                cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.yawn_alert_active:
                cv2.putText(frame, "ALERTA: BOCEJO!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        else: # Nenhuma cara encontrada
            logging.debug("DMSCore(MediaPipe): Nenhuma face encontrada.")
            # Reseta contadores se a cara for perdida
            with self.lock:
                self.drowsiness_counter = 0; self.drowsy_alert_active = False
                self.yawn_counter = 0; self.yawn_alert_active = False
                self.distraction_counter = 0; self.distraction_alert_active = False

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
                
                # Ignora settings de distração, mas loga se tentarem ativar
                new_distraction_state = bool(settings.get('distraction_detection_enabled', self.distraction_detection_enabled))
                if new_distraction_state:
                    logging.warning("DMSCore(MediaPipe): Tentativa de ativar deteção de distração, mas não está implementada neste backend.")
                
                # Força a distração para False
                self.distraction_detection_enabled = False

                logging.info(f"Conf DMS Core(MediaPipe): EAR<{self.ear_threshold}({self.ear_frames}f), MAR>{self.mar_threshold}({self.mar_frames}f), Distração: DESATIVADA (Não implementada)")
                return True
            except (ValueError, TypeError) as e:
                logging.error(f"Erro conf MediaPipe (valor inválido?): {e}"); return False
            except Exception as e:
                logging.error(f"Erro inesperado conf MediaPipe: {e}", exc_info=True); return False

    def get_settings(self):
        """Obtém configurações."""
        logging.debug("DMSCore(MediaPipe): get_settings.")
        with self.lock:
            # Retorna a estrutura completa, mas a distração estará sempre 'False'
            return {"ear_threshold": self.ear_threshold, "ear_frames": self.ear_frames,
                    "mar_threshold": self.mar_threshold, "mar_frames": self.mar_frames,
                    "distraction_detection_enabled": self.distraction_detection_enabled, # Sempre False
                    "distraction_angle": self.distraction_angle,
                    "distraction_frames": self.distraction_frames,
                    "pitch_up_threshold": self.pitch_up_threshold,
                    "pitch_down_offset": self.pitch_down_offset,
                    "yaw_center_offset": self.yaw_center_offset,
                    "pitch_center_offset": self.pitch_center_offset
                   }