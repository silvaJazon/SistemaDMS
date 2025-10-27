# Documentação: Núcleo de Deteção do Driver Monitor System
# Responsável por:
# 1. Carregar os modelos Dlib.
# 2. Processar um frame (gray) e encontrar rosto/olhos.
# 3. Calcular EAR e Pose da Cabeça.
# 4. Receber atualizações de calibração.
# 5. Desenhar no frame (display).
# NOVO: Retorna uma lista de "eventos" (alertas) para serem processados.

import cv2
import dlib
import numpy as np
import math
import logging
from scipy.spatial import distance as dist
import threading # NOVO: Adicionado Lock para settings

class DriverMonitor:
    
    # --- Constantes de Modelo ---
    MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'
    # Índices dos landmarks do Dlib
    EYE_AR_LEFT_START = 42
    EYE_AR_LEFT_END = 48
    EYE_AR_RIGHT_START = 36
    EYE_AR_RIGHT_END = 42

    def __init__(self, frame_size):
        logging.info("A inicializar o DriverMonitor Core...")
        
        # --- Parâmetros de Calibração (Valores Padrão) ---
        self.settings_lock = threading.Lock() # Protege o acesso às configurações
        self.frame_size = frame_size
        self.ear_threshold = 0.25
        self.ear_consec_frames = 15
        self.distraction_threshold_angle = 30.0
        self.distraction_consec_frames = 25
        self.apply_histogram_equalization = True # Para câmaras IR

        # --- Contadores de Alerta ---
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        
        # --- Modelos 3D para Pose da Cabeça ---
        self.model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Ponta do nariz (30)
                                    (0.0, -330.0, -65.0),        # Queixo (8)
                                    (-225.0, 170.0, -135.0),     # Canto do olho esquerdo (36)
                                    (225.0, 170.0, -135.0),      # Canto do olho direito (45)
                                    (-150.0, -150.0, -125.0),    # Canto da boca esquerdo (48)
                                    (150.0, -150.0, -125.0)     # Canto da boca direito (54)
                                ])
        
        # Parâmetros da câmara (assumidos)
        focal_length = self.frame_size[1]
        center = (self.frame_size[1]/2, self.frame_size[0]/2)
        self.camera_matrix = np.array(
                                    [[focal_length, 0, center[0]],
                                    [0, focal_length, center[1]],
                                    [0, 0, 1]], dtype = "double"
                                    )
        self.dist_coeffs = np.zeros((4,1)) # Assumindo sem distorção de lente

        # --- Carregar Modelos Dlib ---
        self.detector = None
        self.predictor = None
        self.initialize_dlib_models()

    # --- Métodos de Inicialização ---
    
    def initialize_dlib_models(self):
        """Carrega o detetor de face e o preditor de landmarks do Dlib."""
        try:
            logging.info(">>> Carregando detetor de faces do Dlib...")
            self.detector = dlib.get_frontal_face_detector()
            logging.info(f">>> Carregando preditor de landmarks faciais ({self.MODEL_PATH})...")
            self.predictor = dlib.shape_predictor(self.MODEL_PATH)
            logging.info(">>> Modelos Dlib carregados com sucesso.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelos Dlib ({self.MODEL_PATH}): {e}", exc_info=True)
            raise e # Propaga o erro para parar a aplicação

    # --- Métodos de Configuração (API) ---

    def update_settings(self, ear_thresh=None, ear_frames=None, distraction_angle=None, distraction_frames=None):
        """Atualiza as configurações de calibração (chamado pela API)."""
        with self.settings_lock:
            if ear_thresh:
                self.ear_threshold = float(ear_thresh)
            if ear_frames:
                self.ear_consec_frames = int(ear_frames)
            if distraction_angle:
                self.distraction_threshold_angle = float(distraction_angle)
            if distraction_frames:
                self.distraction_consec_frames = int(distraction_frames)
            
            logging.info(f"Configurações atualizadas: EAR_T={self.ear_threshold}, EAR_F={self.ear_consec_frames}, DIST_A={self.distraction_threshold_angle}, DIST_F={self.distraction_consec_frames}")

    def get_settings(self):
        """Retorna as configurações atuais (chamado pela API)."""
        with self.settings_lock:
            return {
                "ear_threshold": self.ear_threshold,
                "ear_consec_frames": self.ear_consec_frames,
                "distraction_threshold_angle": self.distraction_threshold_angle,
                "distraction_consec_frames": self.distraction_consec_frames
            }

    # --- Métodos de Cálculo (Auxiliares) ---
    
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
        if C == 0: return 0.3 # Evita divisão por zero
        ear = (A + B) / (2.0 * C)
        return ear

    def _estimate_head_pose(self, shape_np):
        """Estima a pose da cabeça (para onde o motorista está a olhar)."""
        
        # Pontos 2D correspondentes do Dlib
        image_points = np.array([
                                shape_np[30],     # Ponta do nariz
                                shape_np[8],      # Queixo
                                shape_np[36],     # Canto do olho esquerdo
                                shape_np[45],     # Canto do olho direito
                                shape_np[48],     # Canto da boca esquerdo
                                shape_np[54]      # Canto da boca direito
                            ], dtype="double")
        
        try:
            # Resolve a pose da cabeça
            (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

            yaw = angles[1]   # Olhar para os lados (positivo = direita)
            pitch = angles[0] # Olhar para cima/baixo (positivo = baixo)
            roll = angles[2]  # Inclinação
            
            return yaw, pitch, roll
        except Exception as e:
            logging.debug(f"Erro ao calcular pose da cabeça: {e}")
            return 0, 0, 0 # Retorna 0 se falhar

    # --- Método Principal de Processamento ---

    def process_frame(self, frame_bgr, frame_gray):
        """
        Função principal que processa um único frame.
        Retorna: (frame_processado, dados_status, lista_eventos)
        """
        
        # NOVO: Lista para armazenar eventos desta frame
        events_triggered = []
        
        # Obtém configurações atuais de forma segura
        with self.settings_lock:
            ear_thresh = self.ear_threshold
            ear_frames = self.ear_consec_frames
            distraction_angle = self.distraction_threshold_angle
            distraction_frames = self.distraction_consec_frames
            
        # Variáveis de estado para esta frame
        status = {"face_detected": False, "ear": 0.0, "yaw": 0.0}
        alarm_drowsy = False
        alarm_distraction = False

        # 1. (Opcional) Melhoria para IR
        if self.apply_histogram_equalization:
             frame_gray = cv2.equalizeHist(frame_gray)

        # 2. Deteção de Faces
        rects = self.detector(frame_gray, 0)
        
        # 3. Loop sobre as faces detetadas (deve ser apenas 1)
        if len(rects) > 0:
            status["face_detected"] = True
            rect = rects[0] # Pega apenas a primeira face
            
            shape = self.predictor(frame_gray, rect)
            shape_np = self._shape_to_np(shape)

            # --- 4. Verificação de Sonolência (EAR) ---
            leftEye = shape_np[self.EYE_AR_LEFT_START:self.EYE_AR_LEFT_END]
            rightEye = shape_np[self.EYE_AR_RIGHT_START:self.EYE_AR_RIGHT_END]
            leftEAR = self._eye_aspect_ratio(leftEye)
            rightEAR = self._eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            status["ear"] = ear

            # Desenha os contornos dos olhos
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame_bgr, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame_bgr, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < ear_thresh:
                self.drowsiness_counter += 1
                if self.drowsiness_counter >= ear_frames:
                    alarm_drowsy = True
                    # NOVO: Adiciona evento à lista
                    if self.drowsiness_counter == ear_frames: # Dispara apenas uma vez
                        logging.warning(f"DETEÇÃO DE SONOLÊNCIA (EAR: {ear:.2f})")
                        events_triggered.append({"type": "SONOLENCIA", "value": ear})
            else:
                self.drowsiness_counter = 0

            # --- 5. Verificação de Distração (Pose da Cabeça) ---
            yaw, pitch, roll = self._estimate_head_pose(shape_np)
            status["yaw"] = yaw
            
            # Verifica se está a olhar para os lados (Yaw) ou muito para baixo (Pitch)
            if abs(yaw) > distraction_angle or pitch > (distraction_angle / 2): # Mais sensível a olhar para baixo
                self.distraction_counter += 1
                if self.distraction_counter >= distraction_frames:
                    alarm_distraction = True
                    # NOVO: Adiciona evento à lista
                    if self.distraction_counter == distraction_frames: # Dispara apenas uma vez
                        logging.warning(f"DETEÇÃO DE DISTRAÇÃO (Yaw: {yaw:.1f}, Pitch: {pitch:.1f})")
                        events_triggered.append({"type": "DISTRACAO", "value": yaw if abs(yaw) > distraction_angle else pitch})
            else:
                self.distraction_counter = 0
                
            # Desenha informações de debug no frame
            cv2.putText(frame_bgr, f"EAR: {ear:.2f}", (self.frame_size[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Yaw: {yaw:.1f}", (self.frame_size[1] - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Se nenhuma face for detetada, reinicia os contadores
        if not status["face_detected"]:
             self.drowsiness_counter = 0
             self.distraction_counter = 0

        # --- 6. Desenha os Alertas Visuais Finais ---
        if alarm_drowsy:
             cv2.putText(frame_bgr, "ALERTA: SONOLENCIA!", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if alarm_distraction:
             cv2.putText(frame_bgr, "ALERTA: DISTRACAO!", (10, 60),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # --- 7. Retorna os resultados ---
        return frame_bgr, status, events_triggered

