# Documentação: Núcleo do SistemaDMS (Driver Monitor System)
# (VERSÃO: Usa Posição Relativa dos Olhos para Distração, Tracking DESATIVADO)

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
# Removido deque - não precisamos mais de suavização de ângulos

cv2.setUseOptimized(True)

MOUTH_AR_START = 60
MOUTH_AR_END = 68

class DriverMonitor:
    """
    Classe principal para a deteção de sonolência e distração.
    (VERSÃO: Usa Posição Relativa dos Olhos para Distração, Tracking DESATIVADO)
    """

    EYE_AR_LEFT_START = 42
    EYE_AR_LEFT_END = 48
    EYE_AR_RIGHT_START = 36
    EYE_AR_RIGHT_END = 42
    # (NOVO) Índices para centroide dos olhos
    EYE_CENTER_LEFT_IDX = [37, 38, 40, 41] # Pontos ao redor da pupila esquerda
    EYE_CENTER_RIGHT_IDX = [43, 44, 46, 47] # Pontos ao redor da pupila direita

    # --- Variáveis relacionadas a SolvePnP REMOVIDAS ---
    # HEAD_MODEL_POINTS, HEAD_IMAGE_POINTS_IDX, camera_matrix, dist_coeffs

    def __init__(self, frame_size):
        logging.info("A inicializar o DriverMonitor Core (Modo: Posição Relativa dos Olhos)...")
        self.frame_height, self.frame_width = frame_size
        # Variáveis da câmara (focal_length, etc.) removidas - não usadas

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

        # --- Histórico de ângulos REMOVIDO ---
        # self.yaw_history, self.pitch_history

        # --- Configurações de Calibração (Valores Padrão) ---
        self.ear_threshold = 0.25
        self.ear_frames = 15
        self.mar_threshold = 0.60
        self.mar_frames = 20
        # (NOVO) Limites para posição relativa dos olhos (0.0 a 1.0)
        self.eye_pos_h_threshold_min = 0.35 # Abaixo disto = olhar esquerda
        self.eye_pos_h_threshold_max = 0.65 # Acima disto = olhar direita
        self.eye_pos_v_threshold_max = 0.60 # Acima disto = olhar baixo (lembrar: Y aumenta p/ baixo)
        self.eye_pos_v_threshold_min = 0.30 # Abaixo disto = olhar cima
        # Mantém frames de distração
        self.distraction_frames = 25
        # Variáveis de pitch REMOVIDAS


    def initialize_dlib(self):
        """Carrega modelos Dlib."""
        # ... (igual a antes)
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
        # ... (igual a antes)
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(0, shape.num_parts): coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _eye_aspect_ratio(self, eye):
        """Calcula EAR."""
        # ... (igual a antes)
        A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3]);
        if C < 1e-6: return 0.3
        return (A + B) / (2.0 * C)

    def _mouth_aspect_ratio(self, mouth):
        """Calcula MAR."""
        # ... (igual a antes)
        A = dist.euclidean(mouth[1], mouth[7]); B = dist.euclidean(mouth[3], mouth[5])
        C = dist.euclidean(mouth[0], mouth[4]);
        if C < 1e-6: return 0.0
        return (A + B) / (2.0 * C)

    # --- Função _estimate_head_pose REMOVIDA ---

    # (NOVO) Função para calcular posição relativa dos olhos
    def _get_relative_eye_position(self, shape_np, face_rect):
        """Calcula a posição (X, Y) do centroide dos olhos relativa à caixa da face."""
        try:
            # Calcula centroide do olho esquerdo
            left_eye_pts = shape_np[self.EYE_CENTER_LEFT_IDX]
            left_eye_center = left_eye_pts.mean(axis=0)

            # Calcula centroide do olho direito
            right_eye_pts = shape_np[self.EYE_CENTER_RIGHT_IDX]
            right_eye_center = right_eye_pts.mean(axis=0)

            # Ponto médio entre os dois olhos
            eyes_midpoint = (left_eye_center + right_eye_center) / 2.0

            # Obtém dimensões da caixa da face
            face_x = face_rect.left()
            face_y = face_rect.top()
            face_w = face_rect.width()
            face_h = face_rect.height()

            if face_w <= 0 or face_h <= 0:
                logging.warning("DMSCore: Caixa da face inválida (largura/altura zero).")
                return None, None # Retorna None se a caixa for inválida

            # Calcula posição relativa (0.0 a 1.0)
            relative_x = (eyes_midpoint[0] - face_x) / face_w
            relative_y = (eyes_midpoint[1] - face_y) / face_h

            logging.debug(f"DMSCore: Posição Relativa Olhos (X={relative_x:.2f}, Y={relative_y:.2f})")
            return relative_x, relative_y

        except Exception as e:
            logging.error(f"DMSCore: Erro ao calcular posição relativa dos olhos: {e}", exc_info=True)
            return None, None # Retorna None em caso de erro

    def _reset_counters_and_cooldowns(self):
        """Reinicia contadores e cooldowns."""
        # ... (igual a antes, sem limpar histórico de ângulos)
        logging.debug("DMSCore: A reiniciar contadores e cooldowns.")
        self.drowsiness_counter = 0; self.distraction_counter = 0; self.yawn_counter = 0
        self.drowsy_alert_active = False; self.distraction_alert_active = False; self.yawn_alert_active = False


    def process_frame(self, frame, gray):
        """Analisa um frame (Posição Relativa Olhos, Sem Tracking)."""
        logging.debug("DMSCore: process_frame (POS_REL_OLHOS) iniciado.")
        start_time_total = time.time()
        events_list = []
        # (ALTERADO) Status agora inclui eye_x, eye_y
        status_data = {"ear": "-", "mar": "-", "eye_x": "-", "eye_y": "-"}
        face_found_this_frame = False
        gray_processed = cv2.equalizeHist(gray)
        logging.debug("DMSCore: Histograma equalizado.")
        current_rect = None

        logging.debug("DMSCore: A executar deteção completa...")
        start_time_detect = time.time()
        rects = self.detector(gray_processed, 0)
        logging.debug(f"DMSCore: Deteção encontrou {len(rects)} faces em {time.time() - start_time_detect:.4f}s.")

        if rects:
            current_rect = max(rects, key=lambda r: r.width() * r.height())
            face_found_this_frame = True
        else:
            logging.debug("DMSCore: Nenhuma face detetada.")
            with self.lock: self._reset_counters_and_cooldowns()

        if face_found_this_frame and current_rect is not None:
            logging.debug("DMSCore: A prever landmarks...")
            start_time_predict = time.time()
            shape = self.predictor(gray, current_rect)
            shape_np = self._shape_to_np(shape)
            logging.debug(f"DMSCore: Landmarks previstos em {time.time() - start_time_predict:.4f}s.")

            # --- Cálculos ---
            leftEye = shape_np[self.EYE_AR_LEFT_START:self.EYE_AR_LEFT_END]
            rightEye = shape_np[self.EYE_AR_RIGHT_START:self.EYE_AR_RIGHT_END]
            ear = (self._eye_aspect_ratio(leftEye) + self._eye_aspect_ratio(rightEye)) / 2.0
            logging.debug(f"DMSCore: EAR={ear:.3f}")
            mouth = shape_np[MOUTH_AR_START:MOUTH_AR_END]
            mar = self._mouth_aspect_ratio(mouth)
            logging.debug(f"DMSCore: MAR={mar:.3f}")

            # (NOVO) Calcula posição relativa dos olhos
            relative_eye_x, relative_eye_y = self._get_relative_eye_position(shape_np, current_rect)

            # --- Desenho ---
            try:
                cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 255), 1)
                pt1 = (int(current_rect.left()), int(current_rect.top()))
                pt2 = (int(current_rect.right()), int(current_rect.bottom()))
                cv2.rectangle(frame, pt1, pt2, (255, 255, 0), 2)
                # (Opcional) Desenha o ponto médio dos olhos
                if relative_eye_x is not None and relative_eye_y is not None:
                    mid_x = int(current_rect.left() + relative_eye_x * current_rect.width())
                    mid_y = int(current_rect.top() + relative_eye_y * current_rect.height())
                    cv2.circle(frame, (mid_x, mid_y), 3, (0, 0, 255), -1) # Ponto vermelho
            except Exception as draw_err: logging.warning(f"DMSCore: Erro ao desenhar: {draw_err}")

            logging.debug("DMSCore: A adquirir lock para alerta...")
            with self.lock:
                logging.debug("DMSCore: Lock de alerta adquirido.")
                # Sonolência (igual)
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
                # Bocejo (igual)
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

                # (NOVO) Distração (baseada na posição relativa dos olhos)
                is_distracted_pos = False
                details = "-"
                if relative_eye_x is not None and relative_eye_y is not None:
                    is_distracted_pos = (relative_eye_x < self.eye_pos_h_threshold_min) or \
                                        (relative_eye_x > self.eye_pos_h_threshold_max) or \
                                        (relative_eye_y < self.eye_pos_v_threshold_min) or \
                                        (relative_eye_y > self.eye_pos_v_threshold_max)
                    details = f"X:{relative_eye_x:.2f}, Y:{relative_eye_y:.2f}" # Detalhes para o evento

                if is_distracted_pos:
                    self.distraction_counter += 1
                    logging.debug(f"DMSCore: Posição Rel Olhos fora ({details} vs H:[{self.eye_pos_h_threshold_min},{self.eye_pos_h_threshold_max}], V:[{self.eye_pos_v_threshold_min},{self.eye_pos_v_threshold_max}]), cont={self.distraction_counter}/{self.distraction_frames}")
                    if self.distraction_counter >= self.distraction_frames and not self.distraction_alert_active:
                        self.distraction_alert_active = True
                        events_list.append({"type": "DISTRACAO", "value": details, "timestamp": datetime.now().isoformat() + "Z"})
                        logging.warning("DMSCore: EVENTO DISTRACAO.")
                else:
                    if self.distraction_counter > 0: logging.debug("DMSCore: Distração reset.")
                    self.distraction_counter = 0; self.distraction_alert_active = False
            logging.debug("DMSCore: Lock de alerta libertado.")

            # (ALTERADO) Status agora inclui eye_x, eye_y (RAW)
            status_data = {
                "ear": f"{ear:.2f}", "mar": f"{mar:.2f}",
                "eye_x": f"{relative_eye_x:.2f}" if relative_eye_x is not None else "-",
                "eye_y": f"{relative_eye_y:.2f}" if relative_eye_y is not None else "-"
            }

            # Desenha alertas
            if self.drowsy_alert_active: cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.yawn_alert_active: cv2.putText(frame, "ALERTA: BOCEJO!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.distraction_alert_active: cv2.putText(frame, "ALERTA: DISTRACAO!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
             logging.debug("DMSCore: Nenhuma face encontrada neste frame.")
             # Reset já feito se rects estava vazio

        total_time = time.time() - start_time_total
        logging.debug(f"DMSCore: process_frame (POS_REL_OLHOS) concluído em {total_time:.4f}s.")
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
                # (NOVO) Atualiza limites de posição relativa
                self.eye_pos_h_threshold_min = float(settings.get('eye_pos_h_threshold_min', self.eye_pos_h_threshold_min))
                self.eye_pos_h_threshold_max = float(settings.get('eye_pos_h_threshold_max', self.eye_pos_h_threshold_max))
                self.eye_pos_v_threshold_min = float(settings.get('eye_pos_v_threshold_min', self.eye_pos_v_threshold_min))
                self.eye_pos_v_threshold_max = float(settings.get('eye_pos_v_threshold_max', self.eye_pos_v_threshold_max))
                self.distraction_frames = int(settings.get('distraction_frames', self.distraction_frames))

                # Log atualizado com novos parâmetros
                logging.info(f"Conf DMS Core atualizada: EAR<{self.ear_threshold}({self.ear_frames}f), MAR>{self.mar_threshold}({self.mar_frames}f), EyePos H:[{self.eye_pos_h_threshold_min},{self.eye_pos_h_threshold_max}] V:[{self.eye_pos_v_threshold_min},{self.eye_pos_v_threshold_max}] ({self.distraction_frames}f)")
                return True
            except (ValueError, TypeError) as e: logging.error(f"Erro conf (valor inválido?): {e}"); return False
            except Exception as e: logging.error(f"Erro inesperado conf: {e}", exc_info=True); return False

    def get_settings(self):
        """Obtém configurações."""
        logging.debug("DMSCore: get_settings.")
        with self.lock:
            # (NOVO) Retorna os novos parâmetros de posição
            return {"ear_threshold": self.ear_threshold, "ear_frames": self.ear_frames,
                    "mar_threshold": self.mar_threshold, "mar_frames": self.mar_frames,
                    "eye_pos_h_threshold_min": self.eye_pos_h_threshold_min,
                    "eye_pos_h_threshold_max": self.eye_pos_h_threshold_max,
                    "eye_pos_v_threshold_min": self.eye_pos_v_threshold_min,
                    "eye_pos_v_threshold_max": self.eye_pos_v_threshold_max,
                    "distraction_frames": self.distraction_frames
                   }

