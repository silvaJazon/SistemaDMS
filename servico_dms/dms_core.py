# Documentação: Núcleo do SistemaDMS (Implementação Dlib)
# (VERSÃO: Híbrido + Pose Suave + Offsets + Distração Opcional)

import cv2
import dlib
import numpy as np
import logging
import math
import threading
from scipy.spatial import distance as dist
from datetime import datetime
import time
import sys
from collections import deque

# (NOVO) Importa a classe base
from dms_base import BaseMonitor

cv2.setUseOptimized(True)

MOUTH_AR_START = 60
MOUTH_AR_END = 68
FRAMES_FOR_REDETECTION = 10
TRACKER_CONFIDENCE_THRESHOLD = 5.5
ANGLE_SMOOTHING_FRAMES = 7

# (ALTERADO) Renomeia a classe e herda de BaseMonitor
class DlibMonitor(BaseMonitor):
    """
    Classe principal para a deteção de sonolência e distração (Backend: Dlib).
    (VERSÃO: Híbrido + Pose Suave + Offsets + Distração Opcional)
    """
    EYE_AR_LEFT_START = 42; EYE_AR_LEFT_END = 48
    EYE_AR_RIGHT_START = 36; EYE_AR_RIGHT_END = 42

    HEAD_MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])
    HEAD_IMAGE_POINTS_IDX = [30, 8, 36, 45, 48, 54]

    def __init__(self, frame_size):
        # (NOVO) Chama o __init__ da classe base
        super().__init__(frame_size) 
        
        # (ALTERADO) Log específico do Dlib
        logging.info("A inicializar o DlibMonitor Core (Modo: Pose 3D Suavizada + Offsets + Distração Opcional)...")
        # O resto do __init__ permanece igual
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
        self.drowsiness_counter = 0; self.distraction_counter = 0; self.yawn_counter = 0
        self.drowsy_alert_active = False; self.distraction_alert_active = False; self.yawn_alert_active = False
        self.face_tracker = None; self.tracking_active = False; self.tracked_rect = None; self.frame_since_detection = 0
        self.yaw_history = deque(maxlen=ANGLE_SMOOTHING_FRAMES)
        self.pitch_history = deque(maxlen=ANGLE_SMOOTHING_FRAMES)

        # Configurações (Padrão)
        self.ear_threshold = 0.25; self.ear_frames = 15
        self.mar_threshold = 0.60; self.mar_frames = 20
        # (CORRIGIDO) Garante que o padrão é False
        self.distraction_detection_enabled = False
        self.distraction_angle = 40.0 # Yaw
        self.distraction_frames = 35
        self.pitch_down_offset = 20.0
        self.pitch_up_threshold = -15.0
        self.yaw_center_offset = 0.0
        self.pitch_center_offset = 0.0

    def initialize_dlib(self):
        """Carrega modelos Dlib."""
        try:
            logging.info(">>> Carregando detetor faces (Dlib)...")
            self.detector = dlib.get_frontal_face_detector()
            model_path = 'shape_predictor_68_face_landmarks.dat'
            logging.info(f">>> Carregando preditor landmarks ({model_path})...")
            self.predictor = dlib.shape_predictor(model_path)
            logging.info(">>> Modelos Dlib carregados.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL Dlib ({model_path}): {e}", exc_info=True)
            raise RuntimeError(f"Erro Dlib: {e}")

    def _shape_to_np(self, shape, dtype="int"):
        """Converte landmarks Dlib para NumPy."""
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(shape.num_parts): coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _eye_aspect_ratio(self, eye):
        """Calcula EAR."""
        A=dist.euclidean(eye[1],eye[5]); B=dist.euclidean(eye[2],eye[4]); C=dist.euclidean(eye[0],eye[3])
        return 0.3 if C < 1e-6 else (A + B) / (2.0 * C)

    def _mouth_aspect_ratio(self, mouth):
        """Calcula MAR."""
        A=dist.euclidean(mouth[1],mouth[7]); B=dist.euclidean(mouth[3],mouth[5]); C=dist.euclidean(mouth[0],mouth[4])
        return 0.0 if C < 1e-6 else (A + B) / (2.0 * C)

    def _estimate_head_pose(self, shape_np):
        """Estima pose da cabeça (Yaw, Pitch, Roll)."""
        logging.debug("DMSCore(Dlib): Estimando pose...")
        start_time_pose = time.time()
        image_points = np.array([shape_np[i] for i in self.HEAD_IMAGE_POINTS_IDX], dtype="double")
        try:
            (success, rotation_vector, _) = cv2.solvePnP(self.HEAD_MODEL_POINTS, image_points,
                                                          self.camera_matrix, self.dist_coeffs,
                                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if not success: logging.debug("DMSCore(Dlib): solvePnP falhou."); return None, None, None
            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            yaw = angles[1]; pitch = angles[0]; roll = angles[2]
            logging.debug(f"DMSCore(Dlib): Pose RAW (Y={yaw:.1f}, P={pitch:.1f}, R={roll:.1f}) {time.time()-start_time_pose:.4f}s.")
            return yaw, pitch, roll
        except cv2.error as e: logging.warning(f"DMSCore(Dlib): Erro OpenCV pose: {e}"); return None, None, None
        except Exception as e: logging.error(f"DMSCore(Dlib): Erro pose: {e}", exc_info=True); return None, None, None

    def _reset_counters_and_cooldowns(self):
        """Reinicia contadores, cooldowns e histórico de ângulos."""
        logging.debug("DMSCore(Dlib): Reset counters/cooldowns/history.")
        self.drowsiness_counter=0; self.distraction_counter=0; self.yawn_counter=0
        self.drowsy_alert_active=False; self.distraction_alert_active=False; self.yawn_alert_active=False
        self.yaw_history.clear(); self.pitch_history.clear()


    def process_frame(self, frame, gray):
        """Analisa um frame (Híbrido + Pose 3D Suavizada + Offsets + Distração Opcional)."""
        logging.debug("DMSCore(Dlib): process_frame (HÍBRIDO + POSE SUAVE + OFFSETS + DIST_OPCIONAL) iniciado.")
        start_time_total = time.time()
        events_list = []
        status_data = {"ear": "-", "mar": "-", "yaw": "-", "pitch": "-", "roll": "-"}
        face_found_this_frame = False
        gray_processed = cv2.equalizeHist(gray) # Para deteção
        current_rect = None

        # --- Lógica de Deteção/Tracking ---
        logging.debug("DMSCore(Dlib): Lock tracking...")
        with self.lock:
            # (NOVO) Lê estado de ativação da distração DENTRO do lock
            # para evitar condições de corrida se for alterado enquanto processa
            distraction_enabled = self.distraction_detection_enabled

            logging.debug("DMSCore(Dlib): Lock tracking OK.")
            needs_detection = (not self.tracking_active) or (self.frame_since_detection >= FRAMES_FOR_REDETECTION)
            logging.debug(f"DMSCore(Dlib): Track={self.tracking_active}, Frames={self.frame_since_detection}, Detect={needs_detection}")
            if needs_detection:
                logging.debug("DMSCore(Dlib): Detecção completa...")
                start_time_detect = time.time()
                rects = self.detector(gray_processed, 0)
                logging.debug(f"DMSCore(Dlib): Deteção {len(rects)} faces {time.time()-start_time_detect:.4f}s.")
                if rects:
                    current_rect = max(rects, key=lambda r: r.width()*r.height())
                    if self.face_tracker is None: self.face_tracker = dlib.correlation_tracker()
                    logging.debug("DMSCore(Dlib): Iniciando tracker...")
                    self.face_tracker.start_track(frame, current_rect)
                    self.tracking_active=True; self.tracked_rect=current_rect; self.frame_since_detection=0; face_found_this_frame=True
                else:
                    logging.debug("DMSCore(Dlib): Nenhuma face detetada.")
                    if self.tracking_active: self._reset_counters_and_cooldowns()
                    self.tracking_active=False; self.tracked_rect=None
            elif self.tracking_active:
                logging.debug("DMSCore(Dlib): Tracking...")
                start_time_track = time.time()
                confidence = self.face_tracker.update(frame)
                logging.debug(f"DMSCore(Dlib): Track conf={confidence:.2f} {time.time()-start_time_track:.4f}s.")
                if confidence > TRACKER_CONFIDENCE_THRESHOLD:
                    self.tracked_rect = self.face_tracker.get_position()
                    current_rect = dlib.rectangle(int(self.tracked_rect.left()), int(self.tracked_rect.top()),
                                                   int(self.tracked_rect.right()), int(self.tracked_rect.bottom()))
                    self.frame_since_detection += 1; face_found_this_frame = True
                else:
                    logging.debug(f"DMSCore(Dlib): Tracker perdeu face (conf {confidence:.2f}).")
                    self._reset_counters_and_cooldowns(); self.tracking_active=False; self.tracked_rect=None
        logging.debug("DMSCore(Dlib): Lock tracking libertado.")

        # --- Processamento dos Landmarks e Pose ---
        if face_found_this_frame and current_rect is not None:
            logging.debug("DMSCore(Dlib): Prevendo landmarks...")
            start_time_predict = time.time()
            shape = self.predictor(gray, current_rect)
            shape_np = self._shape_to_np(shape)
            logging.debug(f"DMSCore(Dlib): Landmarks {time.time()-start_time_predict:.4f}s.")

            # Cálculos EAR, MAR
            leftEye=shape_np[self.EYE_AR_LEFT_START:self.EYE_AR_LEFT_END]; rightEye=shape_np[self.EYE_AR_RIGHT_START:self.EYE_AR_RIGHT_END]
            ear = (self._eye_aspect_ratio(leftEye) + self._eye_aspect_ratio(rightEye)) / 2.0; logging.debug(f"DMSCore(Dlib): EAR={ear:.3f}")
            mouth = shape_np[MOUTH_AR_START:MOUTH_AR_END]; mar = self._mouth_aspect_ratio(mouth); logging.debug(f"DMSCore(Dlib): MAR={mar:.3f}")

            # (ALTERADO) Cálculo de Pose e Suavização só se a distração estiver ATIVA
            yaw, pitch, roll = None, None, None
            avg_yaw_centered, avg_pitch_centered = None, None
            if distraction_enabled:
                yaw, pitch, roll = self._estimate_head_pose(shape_np)
                if yaw is not None and pitch is not None:
                    with self.lock:
                        self.yaw_history.append(yaw); self.pitch_history.append(pitch)
                        if len(self.yaw_history) == ANGLE_SMOOTHING_FRAMES:
                            avg_yaw = np.mean(self.yaw_history); avg_pitch = np.mean(self.pitch_history)
                            avg_yaw_centered = avg_yaw - self.yaw_center_offset
                            avg_pitch_centered = avg_pitch - self.pitch_center_offset
                            logging.debug(f"DMSCore(Dlib): Ângulos Suaves (Y={avg_yaw:.1f}, P={avg_pitch:.1f}), Centrados (Yc={avg_yaw_centered:.1f}, Pc={avg_pitch_centered:.1f})")
                        else: logging.debug(f"DMSCore(Dlib): Aguardando histórico ({len(self.yaw_history)}/{ANGLE_SMOOTHING_FRAMES})")
                else: # Se a pose falhou, limpa histórico
                    with self.lock: self.yaw_history.clear(); self.pitch_history.clear()
            else: # Se distração desativada, limpa histórico e reseta contador/flag
                 with self.lock:
                      if len(self.yaw_history) > 0 or len(self.pitch_history) > 0:
                          self.yaw_history.clear(); self.pitch_history.clear()
                          logging.debug("DMSCore(Dlib): Histórico de ângulos limpo (distração desativada).")
                      if self.distraction_counter > 0 or self.distraction_alert_active:
                          self.distraction_counter = 0; self.distraction_alert_active = False
                          logging.debug("DMSCore(Dlib): Contador/flag de distração resetado (distração desativada).")


            # Desenho (igual)
            try:
                cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1); cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 255), 1)
                pt1=(int(current_rect.left()), int(current_rect.top())); pt2=(int(current_rect.right()), int(current_rect.bottom()))
                cv2.rectangle(frame, pt1, pt2, (255, 255, 0), 2)
            except Exception as e: logging.warning(f"DMSCore(Dlib): Erro desenho: {e}")

            # --- Lógica de Alerta ---
            logging.debug("DMSCore(Dlib): Lock alerta...")
            with self.lock:
                logging.debug("DMSCore(Dlib): Lock alerta OK.")
                # Sonolência (igual)
                if ear < self.ear_threshold:
                    self.drowsiness_counter += 1; logging.debug(f"DMSCore(Dlib): EAR baixo ({ear:.3f}<{self.ear_threshold}), cont={self.drowsiness_counter}/{self.ear_frames}")
                    if self.drowsiness_counter >= self.ear_frames and not self.drowsy_alert_active:
                        self.drowsy_alert_active=True; events_list.append({"type":"SONOLENCIA", "value":f"EAR: {ear:.2f}", "timestamp":datetime.now().isoformat()+"Z"}); logging.warning("DMSCore(Dlib): EVENTO SONOLENCIA.")
                else:
                    if self.drowsiness_counter > 0: logging.debug("DMSCore(Dlib): Sonolência reset.")
                    self.drowsiness_counter=0; self.drowsy_alert_active=False
                # Bocejo (igual)
                if mar > self.mar_threshold:
                    self.yawn_counter += 1; logging.debug(f"DMSCore(Dlib): MAR alto ({mar:.3f}>{self.mar_threshold}), cont={self.yawn_counter}/{self.mar_frames}")
                    if self.yawn_counter >= self.mar_frames and not self.yawn_alert_active:
                        self.yawn_alert_active=True; events_list.append({"type":"BOCEJO", "value":f"MAR: {mar:.2f}", "timestamp":datetime.now().isoformat()+"Z"}); logging.warning("DMSCore(Dlib): EVENTO BOCEJO.")
                else:
                    if self.yawn_counter > 0: logging.debug("DMSCore(Dlib): Bocejo reset.")
                    self.yawn_counter=0; self.yawn_alert_active=False

                # (ALTERADO) Distração (só executa se ativa E ângulos médios disponíveis)
                if distraction_enabled and avg_yaw_centered is not None and avg_pitch_centered is not None:
                    pitch_down_limit_centered = self.pitch_down_offset
                    pitch_up_limit_centered = self.pitch_up_threshold
                    is_distracted_angle = (abs(avg_yaw_centered) > self.distraction_angle) or \
                                          (avg_pitch_centered < pitch_up_limit_centered) or \
                                          (avg_pitch_centered > pitch_down_limit_centered)
                    details = f"Yaw(c): {avg_yaw_centered:.1f}, Pitch(c): {avg_pitch_centered:.1f}"
                    if is_distracted_angle:
                        self.distraction_counter += 1
                        logging.debug(f"DMSCore(Dlib): Ângulo CENTRADO fora ({details} vs Yaw>{self.distraction_angle}, Pitch<({pitch_up_limit_centered})>({pitch_down_limit_centered})), cont={self.distraction_counter}/{self.distraction_frames}")
                        if self.distraction_counter >= self.distraction_frames and not self.distraction_alert_active:
                            self.distraction_alert_active = True
                            events_list.append({"type": "DISTRACAO", "value": f"Yaw: {yaw:.1f}, Pitch: {pitch:.1f}", "timestamp": datetime.now().isoformat() + "Z"})
                            logging.warning("DMSCore(Dlib): EVENTO DISTRACAO.")
                    else:
                        if self.distraction_counter > 0: logging.debug("DMSCore(Dlib): Distração (centrada) reset.")
                        self.distraction_counter = 0; self.distraction_alert_active = False
                else: # Se distração desativada OU histórico incompleto, garante reset
                     if self.distraction_counter > 0 or self.distraction_alert_active: # Só loga se estava ativo
                         log_reason = "distração desativada" if not distraction_enabled else "histórico incompleto"
                         logging.debug(f"DMSCore(Dlib): Distração reset ({log_reason}).")
                         self.distraction_counter = 0; self.distraction_alert_active = False
            logging.debug("DMSCore(Dlib): Lock alerta libertado.")

            # Status usa ângulos RAW (ou "-" se distração desativada)
            status_data = {"ear": f"{ear:.2f}", "mar": f"{mar:.2f}",
                           "yaw": f"{yaw:.1f}" if distraction_enabled and yaw is not None else "-",
                           "pitch": f"{pitch:.1f}" if distraction_enabled and pitch is not None else "-",
                           "roll": f"{roll:.1f}" if distraction_enabled and roll is not None else "-"}
            # Desenha alertas
            if self.drowsy_alert_active: cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.yawn_alert_active: cv2.putText(frame, "ALERTA: BOCEJO!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # Só desenha alerta de distração se estiver ativo E habilitado
            if distraction_enabled and self.distraction_alert_active:
                 cv2.putText(frame, "ALERTA: DISTRACAO!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
             logging.debug("DMSCore(Dlib): Nenhuma face encontrada/rastreada.")
             # Reset já feito dentro do lock de tracking ou no início do loop se rects vazio

        total_time = time.time() - start_time_total
        logging.debug(f"DMSCore(Dlib): process_frame (HÍBRIDO + POSE SUAVE + OFFSETS + DIST_OPCIONAL) {total_time:.4f}s.")
        return frame, events_list, status_data

    def update_settings(self, settings):
        """Atualiza configurações."""
        logging.debug(f"DMSCore(Dlib): Tentando atualizar conf: {settings}")
        with self.lock:
            try:
                self.ear_threshold = float(settings.get('ear_threshold', self.ear_threshold))
                self.ear_frames = int(settings.get('ear_frames', self.ear_frames))
                self.mar_threshold = float(settings.get('mar_threshold', self.mar_threshold))
                self.mar_frames = int(settings.get('mar_frames', self.mar_frames))
                # (NOVO) Atualiza flag de ativação
                self.distraction_detection_enabled = bool(settings.get('distraction_detection_enabled', self.distraction_detection_enabled))
                self.distraction_angle = float(settings.get('distraction_angle', self.distraction_angle)) # Yaw
                self.distraction_frames = int(settings.get('distraction_frames', self.distraction_frames))
                self.pitch_up_threshold = float(settings.get('pitch_up_threshold', self.pitch_up_threshold))
                self.pitch_down_offset = float(settings.get('pitch_down_offset', self.pitch_down_offset))
                self.yaw_center_offset = float(settings.get('yaw_center_offset', self.yaw_center_offset))
                self.pitch_center_offset = float(settings.get('pitch_center_offset', self.pitch_center_offset))

                # Log atualizado com estado de ativação
                distraction_status = "ATIVADA" if self.distraction_detection_enabled else "DESATIVADA"
                logging.info(f"Conf DMS Core(Dlib): EAR<{self.ear_threshold}({self.ear_frames}f), MAR>{self.mar_threshold}({self.mar_frames}f), Distração:{distraction_status} [Yaw>{self.distraction_angle}({self.distraction_frames}f, Off:{self.yaw_center_offset:.1f}), Pitch<({self.pitch_up_threshold})>({self.pitch_down_offset})({self.distraction_frames}f, Off:{self.pitch_center_offset:.1f})]")

                # (NOVO) Se a distração foi desativada, limpa o histórico de ângulos imediatamente
                if not self.distraction_detection_enabled:
                    self.yaw_history.clear()
                    self.pitch_history.clear()
                    self.distraction_counter = 0 # Reseta contador também
                    self.distraction_alert_active = False # Reseta flag

                return True
            except (ValueError, TypeError) as e: logging.error(f"Erro conf Dlib (valor inválido?): {e}"); return False
            except Exception as e: logging.error(f"Erro inesperado conf Dlib: {e}", exc_info=True); return False

    def get_settings(self):
        """Obtém configurações."""
        logging.debug("DMSCore(Dlib): get_settings.")
        with self.lock:
            # (NOVO) Retorna flag de ativação
            return {"ear_threshold": self.ear_threshold, "ear_frames": self.ear_frames,
                    "mar_threshold": self.mar_threshold, "mar_frames": self.mar_frames,
                    "distraction_detection_enabled": self.distraction_detection_enabled, # NOVO
                    "distraction_angle": self.distraction_angle, # Yaw
                    "distraction_frames": self.distraction_frames,
                    "pitch_up_threshold": self.pitch_up_threshold,
                    "pitch_down_offset": self.pitch_down_offset,
                    "yaw_center_offset": self.yaw_center_offset,
                    "pitch_center_offset": self.pitch_center_offset
                   }

