# Documentação: Núcleo do SistemaDMS (Driver Monitor System)
# Responsável por toda a lógica de análise de imagem (IA).
# (Atualizado com lógica de tracking, bocejo e debug detalhado)

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

# (NOVO) Constantes para índices dos landmarks da boca
MOUTH_AR_START = 60 # Índice inicial dos pontos da boca
MOUTH_AR_END = 68   # Índice final dos pontos da boca (exclusivo)

# (NOVO) Constante para a deteção/redeteção de faces
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
    HEAD_MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),       # Ponta do nariz (30)
        (0.0, -330.0, -65.0),  # Queixo (8)
        (-225.0, 170.0, -135.0),# Canto do olho esquerdo (36)
        (225.0, 170.0, -135.0), # Canto do olho direito (45)
        (-150.0, -150.0, -125.0),# Canto da boca esquerdo (48)
        (150.0, -150.0, -125.0) # Canto da boca direito (54)
    ])

    # Índices dos pontos 2D correspondentes no Dlib
    HEAD_IMAGE_POINTS_IDX = [30, 8, 36, 45, 48, 54]

    def __init__(self, frame_size):
        logging.info("A inicializar o DriverMonitor Core...")

        # Parâmetros da câmara (baseado no tamanho do frame)
        self.frame_height, self.frame_width = frame_size
        self.focal_length = self.frame_width
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
        self.lock = threading.Lock() # Protege as configurações e contadores

        # Contadores de frames
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        self.yawn_counter = 0 # (NOVO) Contador para bocejo

        # Lógica de Cooldown de Alerta
        self.drowsy_alert_active = False
        self.distraction_alert_active = False
        self.yawn_alert_active = False # (NOVO) Cooldown para bocejo

        # --- (NOVO) Lógica de Tracking ---
        self.face_tracker = None        # Objeto dlib.correlation_tracker
        self.tracking_active = False    # Se estamos a rastrear uma face
        self.tracked_rect = None        # O retângulo (dlib.rectangle) da face rastreada
        self.frame_since_detection = 0  # Contador de frames desde a última deteção

        # --- Configurações de Calibração (Valores Padrão) ---
        self.ear_threshold = 0.25      # Limite do Eye Aspect Ratio
        self.ear_frames = 15           # Nº de frames consecutivos para alarme de sonolência
        self.mar_threshold = 0.60      # (NOVO) Limite do Mouth Aspect Ratio
        self.mar_frames = 20           # (NOVO) Nº de frames consecutivos para alarme de bocejo
        self.distraction_angle = 30.0  # Ângulo (graus) para alarme de distração
        self.distraction_frames = 25   # Nº de frames consecutivos para alarme de distração

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
            # Considerar usar sys.exit(1) ou levantar a exceção
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
        # Pontos verticais
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # Ponto horizontal
        C = dist.euclidean(eye[0], eye[3])

        if C < 1e-6: return 0.3 # Evita divisão por zero ou valores muito pequenos
        ear = (A + B) / (2.0 * C)
        return ear

    # (NOVO) Função para calcular o MAR
    def _mouth_aspect_ratio(self, mouth):
        """Calcula o Mouth Aspect Ratio (MAR)."""
        # Distâncias verticais (lábio superior ao inferior)
        A = dist.euclidean(mouth[2], mouth[10]) # 62, 68 (índices 0-based) -> 61, 67 (pontos 1-based)
        B = dist.euclidean(mouth[4], mouth[8])  # 64, 66 -> 63, 65

        # Distância horizontal (canto a canto)
        C = dist.euclidean(mouth[0], mouth[6])  # 60, 64 -> 60, 64 (pontos 1-based)

        if C < 1e-6: return 0.0 # Evita divisão por zero

        mar = (A + B) / (2.0 * C)
        return mar


    def _estimate_head_pose(self, shape_np):
        """Estima a pose da cabeça (para onde o motorista está a olhar)."""
        logging.debug("DMSCore: A estimar pose da cabeça...") # NOVO
        start_time_pose = time.time() # NOVO

        # Pontos 2D correspondentes do Dlib
        image_points = np.array([shape_np[i] for i in self.HEAD_IMAGE_POINTS_IDX], dtype="double")

        try:
            # Resolve a pose da cabeça (PnP)
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                self.HEAD_MODEL_POINTS,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE # (ou SOLVEPNP_IPPE para alternativas)
            )

            if not success:
                 logging.debug("DMSCore: cv2.solvePnP falhou.") # NOVO
                 return 0, 0, 0

            # Converte o vetor de rotação em ângulos de Euler (pitch, yaw, roll)
            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            # decompõe a matriz de rotação
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)

            # Ângulos em graus (Yaw é o mais importante para distração lateral)
            # A ordem pode depender da convenção (X, Y, Z)
            # Geralmente: pitch (X), yaw (Y), roll (Z)
            yaw = angles[1]
            pitch = angles[0]
            roll = angles[2]

            logging.debug(f"DMSCore: Pose estimada (Yaw={yaw:.1f}, Pitch={pitch:.1f}, Roll={roll:.1f}) em {time.time() - start_time_pose:.4f}s.") # NOVO
            return yaw, pitch, roll
        except cv2.error as cv_err: # (NOVO) Captura erros OpenCV específicos
            logging.warning(f"DMSCore: Erro OpenCV em solvePnP/RQDecomp3x3: {cv_err}")
            return 0, 0, 0
        except Exception as e:
            logging.error(f"DMSCore: Erro inesperado ao calcular pose da cabeça: {e}", exc_info=True) # NOVO: Adiciona exc_info
            return 0, 0, 0

    # (NOVO) Função para reiniciar contadores e cooldowns
    def _reset_counters_and_cooldowns(self):
        """Reinicia todos os contadores de frames e flags de cooldown."""
        logging.debug("DMSCore: A reiniciar contadores e cooldowns.") # NOVO
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
        (Otimizado com Tracking)
        """
        logging.debug("DMSCore: process_frame iniciado.") # NOVO
        start_time_total = time.time() # NOVO

        events_list = []
        status_data = {"ear": "-", "mar": "-", "yaw": "-", "pitch": "-", "roll": "-"} # Padrão
        face_found_this_frame = False # NOVO

        # Aplica equalização de histograma
        gray_processed = cv2.equalizeHist(gray)
        logging.debug("DMSCore: Histograma equalizado.") # NOVO

        # --- Lógica de Deteção/Tracking ---
        current_rect = None # O retângulo da face neste frame

        # Verifica se deve fazer deteção completa ou tracking
        # NOVO: Lock para proteger estado de tracking
        logging.debug("DMSCore: A adquirir lock para tracking...") # NOVO
        with self.lock:
             logging.debug("DMSCore: Lock de tracking adquirido.") # NOVO
             needs_detection = (not self.tracking_active) or (self.frame_since_detection >= FRAMES_FOR_REDETECTION)
             logging.debug(f"DMSCore: Tracking ativo={self.tracking_active}, Frames desde deteção={self.frame_since_detection}, Precisa deteção={needs_detection}") # NOVO

             if needs_detection:
                 logging.debug("DMSCore: A executar deteção completa de face...") # NOVO
                 start_time_detect = time.time() # NOVO
                 rects = self.detector(gray_processed, 0)
                 logging.debug(f"DMSCore: Deteção encontrou {len(rects)} faces em {time.time() - start_time_detect:.4f}s.") # NOVO

                 if rects:
                     # Assume a maior face como sendo a do condutor
                     current_rect = max(rects, key=lambda r: r.width() * r.height())
                     # Inicia ou reinicia o tracker
                     if self.face_tracker is None:
                          self.face_tracker = dlib.correlation_tracker()
                     logging.debug("DMSCore: A iniciar/reiniciar tracker...") # NOVO
                     self.face_tracker.start_track(frame, current_rect) # Usa o frame original (colorido) para tracking
                     self.tracking_active = True
                     self.tracked_rect = current_rect
                     self.frame_since_detection = 0 # Reinicia contador
                     face_found_this_frame = True
                 else:
                     # Nenhuma face detetada, desativa tracking e reinicia contadores
                     logging.debug("DMSCore: Nenhuma face detetada na deteção completa.") # NOVO
                     self.tracking_active = False
                     self.tracked_rect = None
                     self._reset_counters_and_cooldowns() # Reinicia tudo se a face for perdida

             elif self.tracking_active:
                 # Faz apenas tracking (mais leve)
                 logging.debug("DMSCore: A executar tracking...") # NOVO
                 start_time_track = time.time() # NOVO
                 # Atualiza o tracker e obtém a confiança
                 # Nota: O update deve usar o frame no qual start_track foi chamado (frame colorido)
                 confidence = self.face_tracker.update(frame)
                 logging.debug(f"DMSCore: Tracking atualizado (confiança={confidence:.2f}) em {time.time() - start_time_track:.4f}s.") # NOVO

                 if confidence > 7.0: # Limite de confiança (experimental)
                     self.tracked_rect = self.face_tracker.get_position()
                     current_rect = dlib.rectangle(
                         int(self.tracked_rect.left()), int(self.tracked_rect.top()),
                         int(self.tracked_rect.right()), int(self.tracked_rect.bottom())
                     )
                     self.frame_since_detection += 1
                     face_found_this_frame = True
                 else:
                     # Tracker perdeu a face (confiança baixa)
                     logging.debug("DMSCore: Tracker perdeu a face (confiança baixa).") # NOVO
                     self.tracking_active = False
                     self.tracked_rect = None
                     self._reset_counters_and_cooldowns() # Reinicia tudo

             # Se tracking estava inativo e não detetamos nada, reinicia (redundante mas seguro)
             if not face_found_this_frame and not self.tracking_active:
                  self._reset_counters_and_cooldowns()

        logging.debug("DMSCore: Lock de tracking libertado.") # NOVO


        # --- Processamento dos Landmarks (APENAS se uma face foi encontrada/rastreada) ---
        if face_found_this_frame and current_rect is not None:
            logging.debug("DMSCore: A prever landmarks...") # NOVO
            start_time_predict = time.time() # NOVO
            # Usa o frame cinzento para predição de landmarks (geralmente mais robusto)
            shape = self.predictor(gray, current_rect)
            shape_np = self._shape_to_np(shape)
            logging.debug(f"DMSCore: Landmarks previstos em {time.time() - start_time_predict:.4f}s.") # NOVO

            # --- 1. Verificação de Sonolência (EAR) ---
            leftEye = shape_np[self.EYE_AR_LEFT_START:self.EYE_AR_LEFT_END]
            rightEye = shape_np[self.EYE_AR_RIGHT_START:self.EYE_AR_RIGHT_END]
            leftEAR = self._eye_aspect_ratio(leftEye)
            rightEAR = self._eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            logging.debug(f"DMSCore: EAR calculado: {ear:.3f}") # NOVO

            # --- 2. Verificação de Bocejo (MAR) ---
            mouth = shape_np[MOUTH_AR_START:MOUTH_AR_END]
            mar = self._mouth_aspect_ratio(mouth)
            logging.debug(f"DMSCore: MAR calculado: {mar:.3f}") # NOVO

            # --- 3. Verificação de Distração (Pose da Cabeça) ---
            yaw, pitch, roll = self._estimate_head_pose(shape_np)
            # Log já está dentro da função _estimate_head_pose


            # Desenha os contornos dos olhos e boca (debug visual)
            try: # NOVO: Adiciona try/except para desenho
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1) # Ciano para boca
                # Desenha o retângulo da face (rastreado ou detetado)
                pt1 = (int(current_rect.left()), int(current_rect.top()))
                pt2 = (int(current_rect.right()), int(current_rect.bottom()))
                cv2.rectangle(frame, pt1, pt2, (255, 255, 0), 2) # Azul claro
            except Exception as draw_err:
                 logging.warning(f"DMSCore: Erro ao desenhar contornos: {draw_err}")


            # --- Lógica de Alerta (com Lock e Cooldown) ---
            logging.debug("DMSCore: A adquirir lock para lógica de alerta...") # NOVO
            with self.lock:
                logging.debug("DMSCore: Lock de alerta adquirido.") # NOVO
                # Sonolência
                if ear < self.ear_threshold:
                    self.drowsiness_counter += 1
                    logging.debug(f"DMSCore: EAR baixo ({ear:.3f} < {self.ear_threshold}), contador={self.drowsiness_counter}/{self.ear_frames}") # NOVO
                    if self.drowsiness_counter >= self.ear_frames and not self.drowsy_alert_active:
                        self.drowsy_alert_active = True
                        events_list.append({
                            "type": "SONOLENCIA", "value": f"EAR: {ear:.2f}",
                            "timestamp": datetime.now().isoformat() + "Z"
                        })
                        logging.warning("DMSCore: EVENTO SONOLENCIA disparado.") # NOVO
                else:
                    if self.drowsy_alert_active: # Log apenas se estava ativo
                        logging.debug("DMSCore: Condição de sonolência terminada.") # NOVO
                    self.drowsiness_counter = 0
                    self.drowsy_alert_active = False

                # Bocejo
                if mar > self.mar_threshold:
                    self.yawn_counter += 1
                    logging.debug(f"DMSCore: MAR alto ({mar:.3f} > {self.mar_threshold}), contador={self.yawn_counter}/{self.mar_frames}") # NOVO
                    if self.yawn_counter >= self.mar_frames and not self.yawn_alert_active:
                        self.yawn_alert_active = True
                        events_list.append({
                            "type": "BOCEJO", "value": f"MAR: {mar:.2f}",
                            "timestamp": datetime.now().isoformat() + "Z"
                        })
                        logging.warning("DMSCore: EVENTO BOCEJO disparado.") # NOVO
                else:
                    if self.yawn_alert_active:
                         logging.debug("DMSCore: Condição de bocejo terminada.") # NOVO
                    self.yawn_counter = 0
                    self.yawn_alert_active = False

                # Distração
                # (NOVO) Lógica mais clara para distração (qualquer um dos ângulos fora do limite)
                is_distracted_angle = abs(yaw) > self.distraction_angle or \
                                      abs(pitch) > self.distraction_angle # Usamos o mesmo ângulo para yaw e pitch agora
                if is_distracted_angle:
                    self.distraction_counter += 1
                    logging.debug(f"DMSCore: Ângulo fora ({yaw=:.1f}, {pitch=:.1f} > {self.distraction_angle}), contador={self.distraction_counter}/{self.distraction_frames}") # NOVO
                    if self.distraction_counter >= self.distraction_frames and not self.distraction_alert_active:
                        self.distraction_alert_active = True
                        events_list.append({
                            "type": "DISTRACAO", "value": f"Yaw: {yaw:.1f}, Pitch: {pitch:.1f}",
                            "timestamp": datetime.now().isoformat() + "Z"
                        })
                        logging.warning("DMSCore: EVENTO DISTRACAO disparado.") # NOVO
                else:
                    if self.distraction_alert_active:
                         logging.debug("DMSCore: Condição de distração terminada.") # NOVO
                    self.distraction_counter = 0
                    self.distraction_alert_active = False

            logging.debug("DMSCore: Lock de alerta libertado.") # NOVO

            # --- Atualiza dados de status para a UI ---
            status_data = { "ear": f"{ear:.2f}", "mar": f"{mar:.2f}", "yaw": f"{yaw:.1f}", "pitch": f"{pitch:.1f}", "roll": f"{roll:.1f}" }

            # --- Desenha Alertas Visuais Finais ---
            # (O alerta VISUAL permanece ativo enquanto a condição persistir)
            # A leitura do estado 'active' não precisa de lock se só for escrita dentro do lock
            if self.drowsy_alert_active:
                cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.yawn_alert_active:
                cv2.putText(frame, "ALERTA: BOCEJO!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Ciano
            if self.distraction_alert_active:
                cv2.putText(frame, "ALERTA: DISTRACAO!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # Laranja


        else: # Nenhuma face encontrada/rastreada neste frame
             logging.debug("DMSCore: Nenhuma face encontrada/rastreada neste frame.")
             # Mantém status_data com "-"

        total_time = time.time() - start_time_total
        logging.debug(f"DMSCore: process_frame concluído em {total_time:.4f}s.") # NOVO

        return frame, events_list, status_data

    def update_settings(self, settings):
        """Atualiza as configurações de calibração de forma segura (thread-safe)."""
        logging.debug(f"DMSCore: Tentando atualizar configurações: {settings}") # NOVO
        with self.lock:
            try:
                # Usa .get com o valor atual como padrão para evitar sobrescrever se a chave não existir
                self.ear_threshold = float(settings.get('ear_threshold', self.ear_threshold))
                self.ear_frames = int(settings.get('ear_frames', self.ear_frames))
                self.mar_threshold = float(settings.get('mar_threshold', self.mar_threshold)) # (NOVO)
                self.mar_frames = int(settings.get('mar_frames', self.mar_frames))       # (NOVO)
                self.distraction_angle = float(settings.get('distraction_angle', self.distraction_angle))
                self.distraction_frames = int(settings.get('distraction_frames', self.distraction_frames))
                logging.info(f"Configurações do DMS Core atualizadas para: EAR<{self.ear_threshold} ({self.ear_frames}f), MAR>{self.mar_threshold} ({self.mar_frames}f), Angle>{self.distraction_angle} ({self.distraction_frames}f)") # (NOVO) Log mais claro
                return True
            except (ValueError, TypeError) as e: # (NOVO) Captura erros de conversão
                logging.error(f"Erro ao atualizar configurações do DMS Core (valor inválido?): {e}")
                return False
            except Exception as e:
                logging.error(f"Erro inesperado ao atualizar configurações do DMS Core: {e}", exc_info=True)
                return False

    def get_settings(self):
        """Obtém as configurações atuais de forma segura (thread-safe)."""
        logging.debug("DMSCore: get_settings chamado.") # NOVO
        with self.lock:
            # (NOVO) Inclui MAR
            return {
                "ear_threshold": self.ear_threshold,
                "ear_frames": self.ear_frames,
                "mar_threshold": self.mar_threshold,
                "mar_frames": self.mar_frames,
                "distraction_angle": self.distraction_angle,
                "distraction_frames": self.distraction_frames
            }

