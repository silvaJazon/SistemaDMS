# Documentação: Núcleo do SistemaDMS (Driver Monitor System)
# (VERSÃO DE TESTE: Tracking DESATIVADO + Suavização de Ângulos)

import cv2
import dlib
import numpy as np
import logging
import math
import threading
from scipy.spatial import distance as dist
from datetime import datetime
import time # Para medir tempo
import sys # Para sair em caso de erro fatal
from collections import deque # (NOVO) Para suavização

cv2.setUseOptimized(True)

MOUTH_AR_START = 60
MOUTH_AR_END = 68

# (NOVO) Número de frames para suavização dos ângulos
ANGLE_SMOOTHING_FRAMES = 5

class DriverMonitor:
    """
    Classe principal para a deteção de sonolência e distração.
    (VERSÃO DE TESTE: Tracking DESATIVADO + Suavização de Ângulos)
    """

    EYE_AR_LEFT_START = 42
    EYE_AR_LEFT_END = 48
    EYE_AR_RIGHT_START = 36
    EYE_AR_RIGHT_END = 42

    HEAD_MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),        # Ponta do nariz (30)
        (0.0, -330.0, -65.0),   # Queixo (8)
        (-225.0, 170.0, -135.0), # Canto do olho esquerdo (36)
        (225.0, 170.0, -135.0),  # Canto do olho direito (45)
        (-150.0, -150.0, -125.0),# Canto da boca esquerdo (48)
        (150.0, -150.0, -125.0)  # Canto da boca direito (54)
    ])
    HEAD_IMAGE_POINTS_IDX = [30, 8, 36, 45, 48, 54]

    def __init__(self, frame_size):
        logging.info("A inicializar o DriverMonitor Core...")
        self.frame_height, self.frame_width = frame_size
        self.focal_length = self.frame_width
        self.camera_center = (self.frame_width / 2, self.frame_height / 2)
        self.camera_matrix = np.array([[self.focal_length, 0, self.camera_center[0]],
                                       [0, self.focal_length, self.camera_center[1]],
                                       [0, 0, 1]], dtype="double")
        self.dist_coeffs = np.zeros((4,1))

        self.detector = None
        self.predictor = None
        self.initialize_dlib()

        self.lock = threading.Lock()
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        self.yawn_counter = 0
        self.drowsy_alert_active = False
        self.distraction_alert_active = False
        self.yawn_alert_active = False

        # --- Tracking Desativado ---

        # (NOVO) Deques para suavização dos ângulos
        self.yaw_history = deque(maxlen=ANGLE_SMOOTHING_FRAMES)
        self.pitch_history = deque(maxlen=ANGLE_SMOOTHING_FRAMES)

        self.ear_threshold = 0.25
        self.ear_frames = 15
        self.mar_threshold = 0.60
        self.mar_frames = 20
        self.distraction_angle = 35.0
        self.distraction_frames = 25
        self.pitch_down_offset = 15.0
        self.pitch_up_threshold = -10.0

    def initialize_dlib(self):
        """Carrega modelos Dlib."""
        try:
            logging.info(">>> Carregando detetor de faces do Dlib...")
            self.detector = dlib.get_frontal_face_detector()
            model_path = 'shape_predictor_68_face_landmarks.dat'
            logging.info(f">>> Carregando preditor de landmarks faciais ({model_path})...")
            self.predictor = dlib.shape_predictor(model_path)
            logging.info(">>> Modelos Dlib carregados com sucesso.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelos Dlib ({model_path}): {e}", exc_info=True)
            raise RuntimeError(f"Não foi possível carregar modelos Dlib: {e}")

    def _shape_to_np(self, shape, dtype="int"):
        """Converte landmarks Dlib para NumPy."""
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(0, shape.num_parts): coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _eye_aspect_ratio(self, eye):
        """Calcula EAR."""
        A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3]);
        if C < 1e-6: return 0.3
        return (A + B) / (2.0 * C)

    def _mouth_aspect_ratio(self, mouth):
        """Calcula MAR."""
        A = dist.euclidean(mouth[1], mouth[7]); B = dist.euclidean(mouth[3], mouth[5])
        C = dist.euclidean(mouth[0], mouth[4]);
        if C < 1e-6: return 0.0
        return (A + B) / (2.0 * C)

    def _estimate_head_pose(self, shape_np):
        """Estima pose da cabeça."""
        logging.debug("DMSCore: A estimar pose da cabeça...")
        start_time_pose = time.time()
        image_points = np.array([shape_np[i] for i in self.HEAD_IMAGE_POINTS_IDX], dtype="double")
        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                self.HEAD_MODEL_POINTS, image_points, self.camera_matrix,
                self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success: logging.debug("DMSCore: cv2.solvePnP falhou."); return 0, 0, 0
            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            yaw = angles[1]; pitch = angles[0]; roll = angles[2]
            logging.debug(f"DMSCore: Pose RAW (Y={yaw:.1f}, P={pitch:.1f}, R={roll:.1f}) em {time.time() - start_time_pose:.4f}s.") # Log Raw
            return yaw, pitch, roll
        except cv2.error as cv_err: logging.warning(f"DMSCore: Erro OpenCV pose: {cv_err}"); return 0, 0, 0
        except Exception as e: logging.error(f"DMSCore: Erro inesperado pose: {e}", exc_info=True); return 0, 0, 0

    def _reset_counters_and_cooldowns(self):
        """Reinicia contadores e cooldowns."""
        logging.debug("DMSCore: A reiniciar contadores e cooldowns.")
        self.drowsiness_counter = 0; self.distraction_counter = 0; self.yawn_counter = 0
        self.drowsy_alert_active = False; self.distraction_alert_active = False; self.yawn_alert_active = False
        # (NOVO) Limpa também o histórico de ângulos
        self.yaw_history.clear()
        self.pitch_history.clear()

    def process_frame(self, frame, gray):
        """Analisa um frame (SEM TRACKING + Suavização de Ângulos)."""
        logging.debug("DMSCore: process_frame (SEM TRACKING + SMOOTHING) iniciado.")
        start_time_total = time.time()
        events_list = []
        status_data = {"ear": "-", "mar": "-", "yaw": "-", "pitch": "-", "roll": "-"}
        face_found_this_frame = False
        gray_processed = cv2.equalizeHist(gray)
        logging.debug("DMSCore: Histograma equalizado.")
        current_rect = None

        logging.debug("DMSCore: A executar deteção completa (tracking desativado)...")
        start_time_detect = time.time()
        rects = self.detector(gray_processed, 0)
        logging.debug(f"DMSCore: Deteção encontrou {len(rects)} faces em {time.time() - start_time_detect:.4f}s.")

        if rects:
            current_rect = max(rects, key=lambda r: r.width() * r.height())
            face_found_this_frame = True
        else:
            logging.debug("DMSCore: Nenhuma face detetada.")
            # Lock necessário aqui para aceder a _reset_counters_and_cooldowns que limpa os deques
            with self.lock:
                 self._reset_counters_and_cooldowns()

        if face_found_this_frame and current_rect is not None:
            logging.debug("DMSCore: A prever landmarks...")
            start_time_predict = time.time()
            shape = self.predictor(gray, current_rect)
            shape_np = self._shape_to_np(shape)
            logging.debug(f"DMSCore: Landmarks previstos em {time.time() - start_time_predict:.4f}s.")

            leftEye = shape_np[self.EYE_AR_LEFT_START:self.EYE_AR_LEFT_END]
            rightEye = shape_np[self.EYE_AR_RIGHT_START:self.EYE_AR_RIGHT_END]
            ear = (self._eye_aspect_ratio(leftEye) + self._eye_aspect_ratio(rightEye)) / 2.0
            logging.debug(f"DMSCore: EAR={ear:.3f}")
            mouth = shape_np[MOUTH_AR_START:MOUTH_AR_END]
            mar = self._mouth_aspect_ratio(mouth)
            logging.debug(f"DMSCore: MAR={mar:.3f}")
            yaw, pitch, roll = self._estimate_head_pose(shape_np)

            # --- (NOVO) Suavização dos Ângulos ---
            avg_yaw = yaw
            avg_pitch = pitch
            with self.lock: # Protege acesso aos deques
                self.yaw_history.append(yaw)
                self.pitch_history.append(pitch)
                if len(self.yaw_history) == ANGLE_SMOOTHING_FRAMES: # Calcula média só se tiver N frames
                    avg_yaw = np.mean(self.yaw_history)
                    avg_pitch = np.mean(self.pitch_history)
                    logging.debug(f"DMSCore: Ângulos Suavizados (Y={avg_yaw:.1f}, P={avg_pitch:.1f})")
                else:
                    logging.debug(f"DMSCore: Aguardando histórico de ângulos ({len(self.yaw_history)}/{ANGLE_SMOOTHING_FRAMES})")
            # ------------------------------------

            try:
                cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 255), 1)
                pt1 = (int(current_rect.left()), int(current_rect.top()))
                pt2 = (int(current_rect.right()), int(current_rect.bottom()))
                cv2.rectangle(frame, pt1, pt2, (255, 255, 0), 2)
            except Exception as draw_err: logging.warning(f"DMSCore: Erro ao desenhar: {draw_err}")

            logging.debug("DMSCore: A adquirir lock para alerta...")
            with self.lock:
                logging.debug("DMSCore: Lock de alerta adquirido.")
                # Sonolência (usa EAR instantâneo)
                if ear < self.ear_threshold:
                    self.drowsiness_counter += 1
                    logging.debug(f"DMSCore: EAR baixo ({ear:.3f}<{self.ear_threshold}), cont={self.drowsiness_counter}/{self.ear_frames}")
                    if self.drowsiness_counter >= self.ear_frames and not self.drowsy_alert_active:
                        self.drowsy_alert_active = True
                        events_list.append({"type": "SONOLENCIA", "value": f"EAR: {ear:.2f}", "timestamp": datetime.now().isoformat() + "Z"})
                        logging.warning("DMSCore: EVENTO SONOLENCIA.")
                else:
                    if self.drowsiness_counter > 0: logging.debug("DMSCore: Sonolência reset.")
                    self.drowsiness_counter = 0; self.drowsy_alert_active = False
                # Bocejo (usa MAR instantâneo)
                if mar > self.mar_threshold:
                    self.yawn_counter += 1
                    logging.debug(f"DMSCore: MAR alto ({mar:.3f}>{self.mar_threshold}), cont={self.yawn_counter}/{self.mar_frames}")
                    if self.yawn_counter >= self.mar_frames and not self.yawn_alert_active:
                        self.yawn_alert_active = True
                        events_list.append({"type": "BOCEJO", "value": f"MAR: {mar:.2f}", "timestamp": datetime.now().isoformat() + "Z"})
                        logging.warning("DMSCore: EVENTO BOCEJO.")
                else:
                    if self.yawn_counter > 0: logging.debug("DMSCore: Bocejo reset.")
                    self.yawn_counter = 0; self.yawn_alert_active = False
                # Distração (usa ângulos MÉDIOS)
                pitch_down_limit = self.distraction_angle + self.pitch_down_offset
                # (ALTERADO) Usa avg_yaw e avg_pitch
                is_distracted_angle = (abs(avg_yaw) > self.distraction_angle) or \
                                      (avg_pitch < self.pitch_up_threshold) or \
                                      (avg_pitch > pitch_down_limit)
                if is_distracted_angle:
                    self.distraction_counter += 1
                    # Log usa ângulos médios
                    logging.debug(f"DMSCore: Ângulo MÉDIO fora (Y={avg_yaw:.1f}>{self.distraction_angle} OU P={avg_pitch:.1f} fora [{self.pitch_up_threshold},{pitch_down_limit}]), cont={self.distraction_counter}/{self.distraction_frames}")
                    if self.distraction_counter >= self.distraction_frames and not self.distraction_alert_active:
                        self.distraction_alert_active = True
                        # Evento ainda guarda ângulos RAW do frame atual para informação
                        events_list.append({"type": "DISTRACAO", "value": f"Yaw: {yaw:.1f}, Pitch: {pitch:.1f}", "timestamp": datetime.now().isoformat() + "Z"})
                        logging.warning("DMSCore: EVENTO DISTRACAO.")
                else:
                    if self.distraction_counter > 0: logging.debug("DMSCore: Distração reset.")
                    self.distraction_counter = 0; self.distraction_alert_active = False
            logging.debug("DMSCore: Lock de alerta libertado.")

            # Status usa ângulos RAW para feedback imediato na UI
            status_data = { "ear": f"{ear:.2f}", "mar": f"{mar:.2f}", "yaw": f"{yaw:.1f}", "pitch": f"{pitch:.1f}", "roll": f"{roll:.1f}" }

            # Desenha alertas
            if self.drowsy_alert_active: cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.yawn_alert_active: cv2.putText(frame, "ALERTA: BOCEJO!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.distraction_alert_active: cv2.putText(frame, "ALERTA: DISTRACAO!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
             logging.debug("DMSCore: Nenhuma face encontrada neste frame.")
             # Reset já feito se rects estava vazio

        total_time = time.time() - start_time_total
        logging.debug(f"DMSCore: process_frame (SEM TRACKING + SMOOTHING) concluído em {total_time:.4f}s.")
        return frame, events_list, status_data

    def update_settings(self, settings):
        """Atualiza configurações."""
        logging.debug(f"DMSCore: Tentando atualizar conf: {settings}")
        with self.lock:
            try:
                self.ear_threshold = float(settings.get('ear_threshold', self.ear_threshold))
                self.ear_frames = int(settings.get('ear_frames', self.ear_frames))
                self.mar_threshold = float(settings.get('mar_threshold', self.mar_threshold))
                self.mar_frames = int(settings.get('mar_frames', self.mar_frames))
                self.distraction_angle = float(settings.get('distraction_angle', self.distraction_angle))
                self.distraction_frames = int(settings.get('distraction_frames', self.distraction_frames))
                self.pitch_up_threshold = float(settings.get('pitch_up_threshold', self.pitch_up_threshold))
                self.pitch_down_offset = float(settings.get('pitch_down_offset', self.pitch_down_offset))
                logging.info(f"Conf DMS Core atualizada: EAR<{self.ear_threshold}({self.ear_frames}f), MAR>{self.mar_threshold}({self.mar_frames}f), Yaw>{self.distraction_angle}({self.distraction_frames}f), Pitch<({self.pitch_up_threshold})>({self.distraction_angle + self.pitch_down_offset})")
                return True
            except (ValueError, TypeError) as e: logging.error(f"Erro conf (valor inválido?): {e}"); return False
            except Exception as e: logging.error(f"Erro inesperado conf: {e}", exc_info=True); return False

    def get_settings(self):
        """Obtém configurações."""
        logging.debug("DMSCore: get_settings.")
        with self.lock:
            return {"ear_threshold": self.ear_threshold, "ear_frames": self.ear_frames,
                    "mar_threshold": self.mar_threshold, "mar_frames": self.mar_frames,
                    "distraction_angle": self.distraction_angle, "distraction_frames": self.distraction_frames,
                    "pitch_up_threshold": self.pitch_up_threshold, "pitch_down_offset": self.pitch_down_offset}

