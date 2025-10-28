# Documentação: Núcleo do SistemaDMS (Driver Monitor System)
# Responsável por toda a lógica de análise de imagem (IA).
# (Atualizado com lógica de tracking, bocejo, correção de MAR, debug e ajuste para câmara alta)

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

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

# Constantes para índices dos landmarks da boca
MOUTH_AR_START = 60 # Índice inicial dos pontos da boca (base 0)
MOUTH_AR_END = 68   # Índice final dos pontos da boca (exclusivo)

# Constante para a deteção/redeteção de faces
FRAMES_FOR_REDETECTION = 10 # Executa a deteção completa a cada X frames

class DriverMonitor:
    """
    Classe principal para a deteção de sonolência e distração.
    """

    # --- Índices dos Landmarks Faciais (Dlib 68 pontos) ---
    EYE_AR_LEFT_START = 42
    EYE_AR_LEFT_END = 48
    EYE_AR_RIGHT_START = 36
    EYE_AR_RIGHT_END = 42

    # --- Modelo 3D da Cabeça (para SolvePnP) ---
    # Ajustado ligeiramente para potentially melhor estabilidade com ângulos altos
    HEAD_MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),        # Ponta do nariz (30)
        (0.0, -330.0, -65.0),   # Queixo (8)
        (-225.0, 170.0, -135.0), # Canto do olho esquerdo (36)
        (225.0, 170.0, -135.0),  # Canto do olho direito (45)
        (-150.0, -150.0, -125.0),# Canto da boca esquerdo (48)
        (150.0, -150.0, -125.0)  # Canto da boca direito (54)
    ])

    # Índices dos pontos 2D correspondentes no Dlib (base 0)
    HEAD_IMAGE_POINTS_IDX = [30, 8, 36, 45, 48, 54]

    def __init__(self, frame_size):
        logging.info("A inicializar o DriverMonitor Core...")

        # Parâmetros da câmara (baseado no tamanho do frame)
        self.frame_height, self.frame_width = frame_size
        self.focal_length = self.frame_width # Aproximação comum
        self.camera_center = (self.frame_width / 2, self.frame_height / 2)

        self.camera_matrix = np.array([
            [self.focal_length, 0, self.camera_center[0]],
            [0, self.focal_length, self.camera_center[1]],
            [0, 0, 1]
        ], dtype="double")

        # Assumindo sem distorção de lente
        self.dist_coeffs = np.zeros((4,1))

        # --- Modelos Dlib ---
        self.detector = None
        self.predictor = None
        self.initialize_dlib()

        # --- Estado da Deteção ---
        self.lock = threading.Lock() # Protege as configurações e contadores/estado de tracking

        # Contadores de frames
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        self.yawn_counter = 0 # Contador para bocejo

        # Lógica de Cooldown de Alerta
        self.drowsy_alert_active = False
        self.distraction_alert_active = False
        self.yawn_alert_active = False # Cooldown para bocejo

        # --- Lógica de Tracking ---
        self.face_tracker = None        # Objeto dlib.correlation_tracker
        self.tracking_active = False    # Se estamos a rastrear uma face
        self.tracked_rect = None        # O retângulo (dlib.rectangle) da face rastreada
        self.frame_since_detection = 0  # Contador de frames desde a última deteção

        # --- Configurações de Calibração (Valores Padrão) ---
        self.ear_threshold = 0.25      # Limite do Eye Aspect Ratio
        self.ear_frames = 15           # Nº de frames consecutivos para alarme de sonolência
        self.mar_threshold = 0.60      # Limite do Mouth Aspect Ratio
        self.mar_frames = 20           # Nº de frames consecutivos para alarme de bocejo
        # (AJUSTADO) Aumenta o ângulo padrão para compensar a câmara alta
        self.distraction_angle = 35.0
        self.distraction_frames = 25   # Nº de frames consecutivos para alarme de distração
        # (NOVO) Offset para deteção de olhar para baixo (considerando câmara alta)
        self.PITCH_DOWN_OFFSET = 15.0
        # (NOVO) Limite para deteção de olhar para cima
        self.PITCH_UP_THRESHOLD = -10.0


    def initialize_dlib(self):
        """Carrega o detetor de face e o preditor de landmarks do Dlib."""
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


    # --- Métodos de Deteção Auxiliares ---

    def _shape_to_np(self, shape, dtype="int"):
        """Converte o objeto de landmarks do Dlib para um array NumPy."""
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _eye_aspect_ratio(self, eye):
        """Calcula a distância euclidiana entre os pontos verticais e horizontais do olho."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])

        if C < 1e-6: return 0.3
        ear = (A + B) / (2.0 * C)
        return ear

    def _mouth_aspect_ratio(self, mouth):
        """Calcula o Mouth Aspect Ratio (MAR)."""
        A = dist.euclidean(mouth[1], mouth[7]) # Pontos 62 e 68 (base 1)
        B = dist.euclidean(mouth[3], mouth[5]) # Pontos 64 e 66 (base 1)
        C = dist.euclidean(mouth[0], mouth[4]) # Pontos 61 e 65 (base 1)

        if C < 1e-6: return 0.0
        mar = (A + B) / (2.0 * C)
        return mar


    def _estimate_head_pose(self, shape_np):
        """Estima a pose da cabeça (para onde o motorista está a olhar)."""
        logging.debug("DMSCore: A estimar pose da cabeça...")
        start_time_pose = time.time()

        image_points = np.array([shape_np[i] for i in self.HEAD_IMAGE_POINTS_IDX], dtype="double")

        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                self.HEAD_MODEL_POINTS,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                 logging.debug("DMSCore: cv2.solvePnP falhou.")
                 return 0, 0, 0

            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

            yaw = angles[1]
            pitch = angles[0]
            roll = angles[2]

            # (AJUSTE OPCIONAL) Pode ser necessário inverter yaw ou pitch dependendo da câmara/setup
            # yaw = -yaw

            logging.debug(f"DMSCore: Pose estimada (Yaw={yaw:.1f}, Pitch={pitch:.1f}, Roll={roll:.1f}) em {time.time() - start_time_pose:.4f}s.")
            return yaw, pitch, roll
        except cv2.error as cv_err:
            logging.warning(f"DMSCore: Erro OpenCV em solvePnP/RQDecomp3x3: {cv_err}")
            return 0, 0, 0
        except Exception as e:
            logging.error(f"DMSCore: Erro inesperado ao calcular pose da cabeça: {e}", exc_info=True)
            return 0, 0, 0

    def _reset_counters_and_cooldowns(self):
        """Reinicia todos os contadores de frames e flags de cooldown."""
        # Não precisa de lock aqui, pois é chamada DENTRO de blocos que já têm lock
        logging.debug("DMSCore: A reiniciar contadores e cooldowns.")
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        self.yawn_counter = 0
        self.drowsy_alert_active = False
        self.distraction_alert_active = False
        self.yawn_alert_active = False


    # --- Métodos Principais (Processamento e Configuração) ---

    def process_frame(self, frame, gray):
        """
        Função principal de processamento. Analisa um frame e retorna o
        frame processado, quaisquer eventos de alerta e dados de status.
        (Otimizado com Tracking e ajuste para câmara alta)
        """
        logging.debug("DMSCore: process_frame iniciado.")
        start_time_total = time.time()

        events_list = []
        status_data = {"ear": "-", "mar": "-", "yaw": "-", "pitch": "-", "roll": "-"} # Padrão
        face_found_this_frame = False

        # Aplica equalização de histograma
        gray_processed = cv2.equalizeHist(gray)
        logging.debug("DMSCore: Histograma equalizado.")

        # --- Lógica de Deteção/Tracking ---
        current_rect = None

        logging.debug("DMSCore: A adquirir lock para tracking...")
        with self.lock:
             logging.debug("DMSCore: Lock de tracking adquirido.")
             needs_detection = (not self.tracking_active) or (self.frame_since_detection >= FRAMES_FOR_REDETECTION)
             logging.debug(f"DMSCore: Tracking ativo={self.tracking_active}, Frames desde deteção={self.frame_since_detection}, Precisa deteção={needs_detection}")

             if needs_detection:
                 logging.debug("DMSCore: A executar deteção completa de face...")
                 start_time_detect = time.time()
                 rects = self.detector(gray_processed, 0)
                 logging.debug(f"DMSCore: Deteção encontrou {len(rects)} faces em {time.time() - start_time_detect:.4f}s.")

                 if rects:
                     current_rect = max(rects, key=lambda r: r.width() * r.height())
                     if self.face_tracker is None:
                          self.face_tracker = dlib.correlation_tracker()
                     logging.debug("DMSCore: A iniciar/reiniciar tracker...")
                     self.face_tracker.start_track(frame, current_rect)
                     self.tracking_active = True
                     self.tracked_rect = current_rect
                     self.frame_since_detection = 0
                     face_found_this_frame = True
                     # Reinicia contadores APENAS se o tracking estava inativo antes (evita resets desnecessários)
                     # if not self.tracking_active: self._reset_counters_and_cooldowns() # Comentado - reset só na perda
                 else:
                     logging.debug("DMSCore: Nenhuma face detetada na deteção completa.")
                     if self.tracking_active: # Só reinicia se estava ativo antes
                          self._reset_counters_and_cooldowns()
                     self.tracking_active = False
                     self.tracked_rect = None

             elif self.tracking_active:
                 logging.debug("DMSCore: A executar tracking...")
                 start_time_track = time.time()
                 confidence = self.face_tracker.update(frame)
                 logging.debug(f"DMSCore: Tracking atualizado (confiança={confidence:.2f}) em {time.time() - start_time_track:.4f}s.")

                 if confidence > 6.0:
                     self.tracked_rect = self.face_tracker.get_position()
                     current_rect = dlib.rectangle(
                         int(self.tracked_rect.left()), int(self.tracked_rect.top()),
                         int(self.tracked_rect.right()), int(self.tracked_rect.bottom())
                     )
                     self.frame_since_detection += 1
                     face_found_this_frame = True
                 else:
                     logging.debug("DMSCore: Tracker perdeu a face (confiança baixa).")
                     self._reset_counters_and_cooldowns()
                     self.tracking_active = False
                     self.tracked_rect = None

        logging.debug("DMSCore: Lock de tracking libertado.")


        # --- Processamento dos Landmarks ---
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
            logging.debug(f"DMSCore: EAR calculado: {ear:.3f}")

            mouth = shape_np[MOUTH_AR_START:MOUTH_AR_END]
            mar = self._mouth_aspect_ratio(mouth)
            logging.debug(f"DMSCore: MAR calculado: {mar:.3f}")

            yaw, pitch, roll = self._estimate_head_pose(shape_np)


            # --- Desenho ---
            try:
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)
                pt1 = (int(current_rect.left()), int(current_rect.top()))
                pt2 = (int(current_rect.right()), int(current_rect.bottom()))
                cv2.rectangle(frame, pt1, pt2, (255, 255, 0), 2)
            except Exception as draw_err:
                 logging.warning(f"DMSCore: Erro ao desenhar contornos: {draw_err}")


            # --- Lógica de Alerta (com Lock e Cooldown) ---
            logging.debug("DMSCore: A adquirir lock para lógica de alerta...")
            with self.lock:
                logging.debug("DMSCore: Lock de alerta adquirido.")
                # Sonolência
                if ear < self.ear_threshold:
                    self.drowsiness_counter += 1
                    logging.debug(f"DMSCore: EAR baixo ({ear:.3f} < {self.ear_threshold}), contador={self.drowsiness_counter}/{self.ear_frames}")
                    if self.drowsiness_counter >= self.ear_frames and not self.drowsy_alert_active:
                        self.drowsy_alert_active = True
                        events_list.append({
                            "type": "SONOLENCIA", "value": f"EAR: {ear:.2f}",
                            "timestamp": datetime.now().isoformat() + "Z"
                        })
                        logging.warning("DMSCore: EVENTO SONOLENCIA disparado.")
                else:
                    if self.drowsiness_counter > 0:
                        logging.debug("DMSCore: Condição de sonolência terminada ou contador reiniciado.")
                        self.drowsiness_counter = 0
                    self.drowsy_alert_active = False

                # Bocejo
                if mar > self.mar_threshold:
                    self.yawn_counter += 1
                    logging.debug(f"DMSCore: MAR alto ({mar:.3f} > {self.mar_threshold}), contador={self.yawn_counter}/{self.mar_frames}")
                    if self.yawn_counter >= self.mar_frames and not self.yawn_alert_active:
                        self.yawn_alert_active = True
                        events_list.append({
                            "type": "BOCEJO", "value": f"MAR: {mar:.2f}",
                            "timestamp": datetime.now().isoformat() + "Z"
                        })
                        logging.warning("DMSCore: EVENTO BOCEJO disparado.")
                else:
                    if self.yawn_counter > 0:
                         logging.debug("DMSCore: Condição de bocejo terminada ou contador reiniciado.")
                         self.yawn_counter = 0
                    self.yawn_alert_active = False

                # Distração (AJUSTADO PARA CÂMARA ALTA)
                # Verifica Yaw (lateral) OU Pitch muito para cima OU Pitch muito para baixo
                is_distracted_angle = (abs(yaw) > self.distraction_angle) or \
                                      (pitch < self.PITCH_UP_THRESHOLD) or \
                                      (pitch > (self.distraction_angle + self.PITCH_DOWN_OFFSET))

                if is_distracted_angle:
                    self.distraction_counter += 1
                    logging.debug(f"DMSCore: Ângulo fora (Yaw={yaw:.1f} > {self.distraction_angle} OU Pitch={pitch:.1f} fora de [{self.PITCH_UP_THRESHOLD}, {self.distraction_angle + self.PITCH_DOWN_OFFSET}]), contador={self.distraction_counter}/{self.distraction_frames}")
                    if self.distraction_counter >= self.distraction_frames and not self.distraction_alert_active:
                        self.distraction_alert_active = True
                        events_list.append({
                            "type": "DISTRACAO", "value": f"Yaw: {yaw:.1f}, Pitch: {pitch:.1f}",
                            "timestamp": datetime.now().isoformat() + "Z"
                        })
                        logging.warning("DMSCore: EVENTO DISTRACAO disparado.")
                else:
                    if self.distraction_counter > 0:
                         logging.debug("DMSCore: Condição de distração terminada ou contador reiniciado.")
                         self.distraction_counter = 0
                    self.distraction_alert_active = False

            logging.debug("DMSCore: Lock de alerta libertado.")

            # --- Atualiza dados de status ---
            status_data = { "ear": f"{ear:.2f}", "mar": f"{mar:.2f}", "yaw": f"{yaw:.1f}", "pitch": f"{pitch:.1f}", "roll": f"{roll:.1f}" }

            # --- Desenha Alertas Visuais ---
            if self.drowsy_alert_active:
                cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.yawn_alert_active:
                cv2.putText(frame, "ALERTA: BOCEJO!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.distraction_alert_active:
                cv2.putText(frame, "ALERTA: DISTRACAO!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)


        else: # Nenhuma face encontrada/rastreada
             logging.debug("DMSCore: Nenhuma face encontrada/rastreada neste frame.")
             # O reset já acontece dentro do lock de tracking quando a face é perdida

        total_time = time.time() - start_time_total
        logging.debug(f"DMSCore: process_frame concluído em {total_time:.4f}s.")

        return frame, events_list, status_data

    def update_settings(self, settings):
        """Atualiza as configurações de calibração de forma segura (thread-safe)."""
        logging.debug(f"DMSCore: Tentando atualizar configurações: {settings}")
        with self.lock:
            try:
                self.ear_threshold = float(settings.get('ear_threshold', self.ear_threshold))
                self.ear_frames = int(settings.get('ear_frames', self.ear_frames))
                self.mar_threshold = float(settings.get('mar_threshold', self.mar_threshold))
                self.mar_frames = int(settings.get('mar_frames', self.mar_frames))
                self.distraction_angle = float(settings.get('distraction_angle', self.distraction_angle))
                self.distraction_frames = int(settings.get('distraction_frames', self.distraction_frames))
                # (Nota: PITCH_DOWN_OFFSET e PITCH_UP_THRESHOLD não são configuráveis pela UI por enquanto)
                logging.info(f"Configurações do DMS Core atualizadas para: EAR<{self.ear_threshold} ({self.ear_frames}f), MAR>{self.mar_threshold} ({self.mar_frames}f), Angle>{self.distraction_angle} ({self.distraction_frames}f)")
                return True
            except (ValueError, TypeError) as e:
                logging.error(f"Erro ao atualizar configurações do DMS Core (valor inválido?): {e}")
                return False
            except Exception as e:
                logging.error(f"Erro inesperado ao atualizar configurações do DMS Core: {e}", exc_info=True)
                return False

    def get_settings(self):
        """Obtém as configurações atuais de forma segura (thread-safe)."""
        logging.debug("DMSCore: get_settings chamado.")
        with self.lock:
            return {
                "ear_threshold": self.ear_threshold,
                "ear_frames": self.ear_frames,
                "mar_threshold": self.mar_threshold,
                "mar_frames": self.mar_frames,
                "distraction_angle": self.distraction_angle,
                "distraction_frames": self.distraction_frames
            }

