# Documentação: Núcleo do SistemaDMS (Driver Monitor System)
# Responsável por toda a lógica de análise de imagem (IA).
# (Atualizado com tracking de face e deteção de bocejo)

import cv2
import dlib
import numpy as np
import logging
import math
import threading
from scipy.spatial import distance as dist
from datetime import datetime
import sys # Para sys.exit

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

class DriverMonitor:
    """
    Classe principal para a deteção de sonolência, distração e bocejo.
    """

    # --- Constantes ---
    FRAMES_FOR_REDETECTION = 10 # Executa a deteção completa a cada X frames

    # --- Índices dos Landmarks Faciais (Dlib 68 pontos) ---
    EYE_AR_LEFT_START = 42
    EYE_AR_LEFT_END = 48
    EYE_AR_RIGHT_START = 36
    EYE_AR_RIGHT_END = 42
    # (NOVO) Pontos da boca (usaremos os lábios internos para MAR)
    MOUTH_AR_START = 60 # Canto esquerdo interno
    MOUTH_AR_END = 68   # Canto direito interno (exclusivo)

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
        self.lock = threading.Lock() # Protege as configurações e o estado

        # Contadores de frames
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        self.yawn_counter = 0 # (NOVO) Contador para bocejo

        # Cooldown de Alerta
        self.drowsy_alert_active = False
        self.distraction_alert_active = False
        self.yawn_alert_active = False # (NOVO) Cooldown para bocejo

        # --- Estado do Tracking ---
        self.face_tracker = None
        self.tracking_active = False
        self.current_face_rect = None
        self.frame_count_since_detection = 0

        # --- Configurações de Calibração (Valores Padrão) ---
        self.ear_threshold = 0.25      # Limite do Eye Aspect Ratio
        self.ear_frames = 15           # Nº de frames consecutivos para alarme de sonolência
        self.distraction_angle = 30.0  # Ângulo (graus) para alarme de distração
        self.distraction_frames = 25   # Nº de frames consecutivos para alarme de distração
        self.mar_threshold = 0.60      # (NOVO) Limite do Mouth Aspect Ratio para bocejo
        self.mar_frames = 20           # (NOVO) Nº de frames consecutivos para alarme de bocejo

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
            # Considerar usar sys.exit(1) se os modelos forem essenciais
            raise RuntimeError(f"Falha ao carregar modelos Dlib: {e}")


    # --- Métodos de Deteção Auxiliares ---

    def _shape_to_np(self, shape, dtype="int"):
        """Converte o objeto de landmarks do Dlib para um array NumPy."""
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _eye_aspect_ratio(self, eye):
        """Calcula o EAR."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        if C == 0: return 0.3
        ear = (A + B) / (2.0 * C)
        return ear

    def _mouth_aspect_ratio(self, mouth):
        """(NOVO) Calcula o MAR (Mouth Aspect Ratio)."""
        # Pontos verticais (lábio interno)
        A = dist.euclidean(mouth[1], mouth[7]) # 61 -> 67
        B = dist.euclidean(mouth[2], mouth[6]) # 62 -> 66
        C = dist.euclidean(mouth[3], mouth[5]) # 63 -> 65
        # Pontos horizontais (lábio interno)
        D = dist.euclidean(mouth[0], mouth[4]) # 60 -> 64

        if D == 0: return 0.2 # Valor baixo se a boca estiver fechada horizontalmente
        mar = (A + B + C) / (3.0 * D) # Média das verticais / horizontal
        return mar


    def _estimate_head_pose(self, shape_np):
        """Estima a pose da cabeça (para onde o motorista está a olhar)."""
        image_points = np.array([shape_np[i] for i in self.HEAD_IMAGE_POINTS_IDX], dtype="double")

        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                self.HEAD_MODEL_POINTS,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE # Pode experimentar SOLVEPNP_EPNP ou outros
            )
            if not success:
                logging.debug("solvePnP falhou.")
                return 0, 0, 0

            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            # Tenta decompor a matriz de rotação
            try:
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
                yaw = angles[1]
                pitch = angles[0]
                roll = angles[2]
                return yaw, pitch, roll
            except Exception as decomp_e:
                logging.debug(f"Erro na decomposição RQDecomp3x3: {decomp_e}")
                return 0,0,0


        except Exception as e:
            # Captura exceções mais específicas se necessário (ex: cv2.error)
            logging.debug(f"Erro ao calcular pose da cabeça: {e}")
            return 0, 0, 0

    # --- Métodos Principais (Processamento e Configuração) ---

    def process_frame(self, frame, gray):
        """
        Função principal de processamento. Analisa um frame e retorna o
        frame processado, eventos de alerta e dados de status.
        """
        events_list = []
        status_data = {}
        processed_frame = frame.copy() # Começa com uma cópia

        # Aplica equalização de histograma
        gray_processed = cv2.equalizeHist(gray)

        # --- Lógica de Tracking/Redeteção ---
        rect_to_process = None
        perform_detection = False

        with self.lock: # Protege o estado do tracker
            if not self.tracking_active or self.frame_count_since_detection >= self.FRAMES_FOR_REDETECTION:
                perform_detection = True
                self.frame_count_since_detection = 0 # Reinicia contador
            else:
                self.frame_count_since_detection += 1

            if perform_detection:
                self.tracking_active = False # Desativa antes de tentar detetar
                self.face_tracker = None
                self.current_face_rect = None

                rects = self.detector(gray_processed, 0)
                if rects:
                    # Assume a maior face como sendo a do condutor
                    rect = max(rects, key=lambda r: r.width() * r.height())
                    self.current_face_rect = rect
                    rect_to_process = rect # Guarda para processar landmarks

                    # Inicializa o tracker
                    self.face_tracker = dlib.correlation_tracker()
                    self.face_tracker.start_track(gray_processed, rect)
                    self.tracking_active = True
                    logging.debug("DETEÇÃO: Face encontrada e tracker iniciado.")
                else:
                    logging.debug("DETEÇÃO: Nenhuma face encontrada.")
                    # Reinicia contadores e cooldowns se nenhuma face for detetada
                    self._reset_counters_and_cooldowns()

            elif self.tracking_active and self.face_tracker:
                # --- Faz o Tracking ---
                confidence = self.face_tracker.update(gray_processed)
                if confidence > 7.0: # Limiar de confiança (ajustável)
                    tracked_pos = self.face_tracker.get_position()
                    rect = dlib.rectangle(int(tracked_pos.left()), int(tracked_pos.top()),
                                          int(tracked_pos.right()), int(tracked_pos.bottom()))
                    self.current_face_rect = rect # Atualiza a posição atual
                    rect_to_process = rect # Guarda para processar landmarks
                    logging.debug(f"TRACKING: Posição atualizada (Conf: {confidence:.2f})")
                else:
                    # Tracker perdeu a face
                    self.tracking_active = False
                    self.face_tracker = None
                    self.current_face_rect = None
                    logging.debug(f"TRACKING: Perdido (Conf: {confidence:.2f}). Próxima frame fará redeteção.")
                    # Reinicia contadores e cooldowns se a face for perdida
                    self._reset_counters_and_cooldowns()

        # --- Processamento de Landmarks (Só se tivermos uma face) ---
        if rect_to_process:
            try:
                shape = self.predictor(gray, rect_to_process) # Usa 'gray' original para landmarks
                shape_np = self._shape_to_np(shape)

                # Desenha o retângulo da face (debug visual)
                pt1 = (rect_to_process.left(), rect_to_process.top())
                pt2 = (rect_to_process.right(), rect_to_process.bottom())
                cv2.rectangle(processed_frame, pt1, pt2, (255, 255, 0), 1) # Ciano para tracking

                # --- 1. Verificação de Sonolência (EAR) ---
                leftEye = shape_np[self.EYE_AR_LEFT_START:self.EYE_AR_LEFT_END]
                rightEye = shape_np[self.EYE_AR_RIGHT_START:self.EYE_AR_RIGHT_END]
                leftEAR = self._eye_aspect_ratio(leftEye)
                rightEAR = self._eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # Desenha contornos dos olhos
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(processed_frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(processed_frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # --- (NOVO) 2. Verificação de Bocejo (MAR) ---
                mouth = shape_np[self.MOUTH_AR_START:self.MOUTH_AR_END]
                mar = self._mouth_aspect_ratio(mouth)

                # Desenha contorno da boca (lábios internos)
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(processed_frame, [mouthHull], -1, (255, 0, 255), 1) # Magenta

                # --- 3. Verificação de Distração (Pose da Cabeça) ---
                yaw, pitch, roll = self._estimate_head_pose(shape_np)

                # --- Lógica de Alerta (com Cooldown, dentro do lock) ---
                with self.lock:
                    # Sonolência (EAR)
                    if ear < self.ear_threshold:
                        self.drowsiness_counter += 1
                        if self.drowsiness_counter >= self.ear_frames and not self.drowsy_alert_active:
                            self.drowsy_alert_active = True
                            events_list.append({
                                "type": "SONOLENCIA",
                                "value": f"EAR: {ear:.2f}",
                                "timestamp": datetime.now().isoformat() + "Z"
                            })
                    else:
                        self.drowsiness_counter = 0
                        self.drowsy_alert_active = False

                    # (NOVO) Bocejo (MAR)
                    if mar > self.mar_threshold:
                        self.yawn_counter += 1
                        if self.yawn_counter >= self.mar_frames and not self.yawn_alert_active:
                            self.yawn_alert_active = True
                            events_list.append({
                                "type": "BOCEJO",
                                "value": f"MAR: {mar:.2f}",
                                "timestamp": datetime.now().isoformat() + "Z"
                            })
                    else:
                        self.yawn_counter = 0
                        self.yawn_alert_active = False

                    # Distração (Pose)
                    # (Ajuste: Considera Pitch positivo alto como olhar para baixo)
                    if abs(yaw) > self.distraction_angle or pitch > (self.distraction_angle + 5): # Pitch mais sensível
                        self.distraction_counter += 1
                        if self.distraction_counter >= self.distraction_frames and not self.distraction_alert_active:
                            self.distraction_alert_active = True
                            events_list.append({
                                "type": "DISTRACAO",
                                "value": f"Yaw: {yaw:.1f}, Pitch: {pitch:.1f}",
                                "timestamp": datetime.now().isoformat() + "Z"
                            })
                    else:
                        self.distraction_counter = 0
                        self.distraction_alert_active = False

                # --- Desenha Alertas Visuais Finais ---
                # (O alerta VISUAL permanece ativo enquanto a condição persistir)
                alert_y_pos = 30
                if self.drowsy_alert_active:
                    cv2.putText(processed_frame, "ALERTA: SONOLENCIA!", (10, alert_y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    alert_y_pos += 25
                if self.yawn_alert_active: # (NOVO)
                    cv2.putText(processed_frame, "ALERTA: BOCEJO!", (10, alert_y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    alert_y_pos += 25
                if self.distraction_alert_active:
                    cv2.putText(processed_frame, "ALERTA: DISTRACAO!", (10, alert_y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


                # Atualiza dados de status para a UI
                status_data = {
                    "ear": f"{ear:.2f}",
                    "mar": f"{mar:.2f}", # (NOVO)
                    "yaw": f"{yaw:.1f}",
                    "pitch": f"{pitch:.1f}",
                    "roll": f"{roll:.1f}"
                }
            except Exception as landmark_error:
                 # Erro ao processar landmarks (ex: face muito perto da borda)
                 logging.warning(f"Erro ao processar landmarks/pose: {landmark_error}")
                 # Se falhar aqui, não atualiza status nem dispara alertas
                 # Mantém o retângulo desenhado se foi detetado/trackeado
                 if self.current_face_rect:
                    pt1 = (self.current_face_rect.left(), self.current_face_rect.top())
                    pt2 = (self.current_face_rect.right(), self.current_face_rect.bottom())
                    cv2.rectangle(processed_frame, pt1, pt2, (0, 255, 255), 1) # Amarelo para erro


        # Se não houve deteção nem tracking, processed_frame é a cópia original
        # e status_data está vazio.

        return processed_frame, events_list, status_data

    def _reset_counters_and_cooldowns(self):
        """Reinicia contadores e flags de cooldown (chamado quando face é perdida)."""
        with self.lock:
            self.drowsiness_counter = 0
            self.distraction_counter = 0
            self.yawn_counter = 0
            self.drowsy_alert_active = False
            self.distraction_alert_active = False
            self.yawn_alert_active = False

    def update_settings(self, settings):
        """Atualiza as configurações de calibração de forma segura (thread-safe)."""
        with self.lock:
            try:
                # Usa .get() com o valor atual como padrão para evitar erros se a chave não existir
                self.ear_threshold = float(settings.get('ear_threshold', self.ear_threshold))
                self.ear_frames = int(settings.get('ear_frames', self.ear_frames))
                self.distraction_angle = float(settings.get('distraction_angle', self.distraction_angle))
                self.distraction_frames = int(settings.get('distraction_frames', self.distraction_frames))
                # (NOVO) Parâmetros de Bocejo
                self.mar_threshold = float(settings.get('mar_threshold', self.mar_threshold))
                self.mar_frames = int(settings.get('mar_frames', self.mar_frames))
                logging.info(f"Configurações do DMS atualizadas: EAR Thr={self.ear_threshold}, Frames={self.ear_frames}; "
                             f"Dist Angle={self.distraction_angle}, Frames={self.distraction_frames}; "
                             f"MAR Thr={self.mar_threshold}, Frames={self.mar_frames}") # Log mais detalhado
                return True
            except (ValueError, TypeError) as e: # Captura erros de conversão
                logging.error(f"Erro ao atualizar configurações do DMS (valor inválido?): {e} - Settings recebidas: {settings}")
                return False
            except Exception as e:
                logging.error(f"Erro inesperado ao atualizar configurações do DMS: {e}")
                return False

    def get_settings(self):
        """Obtém as configurações atuais de forma segura (thread-safe)."""
        with self.lock:
            return {
                "ear_threshold": self.ear_threshold,
                "ear_frames": self.ear_frames,
                "distraction_angle": self.distraction_angle,
                "distraction_frames": self.distraction_frames,
                "mar_threshold": self.mar_threshold, # (NOVO)
                "mar_frames": self.mar_frames       # (NOVO)
            }

