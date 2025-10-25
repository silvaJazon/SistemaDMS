# Documentação: Núcleo de Processamento do SistemaDMS
# Responsável por:
# 1. Carregar os modelos de IA (Dlib)
# 2. Processar os frames de vídeo para detetar (Sonolência, Distração)

import cv2
import time
import logging
import numpy as np
import dlib
from scipy.spatial import distance as dist
import math

# --- Constantes de Caminho dos ModelOS ---
DLIB_MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'

# --- Constantes de Deteção ---

# 1. Sonolência (EAR)
EAR_THRESHOLD = 0.25      # Limite do Eye Aspect Ratio (Valores mais baixos = mais difícil de detetar)
EAR_CONSEC_FRAMES = 15    # Frames consecutivos para alarme (Valores mais altos = menos sensível)

# 2. Distração (Pose da Cabeça)
DISTRACTION_THRESHOLD_ANGLE = 30.0 # Ângulo (graus) para alarme (Valores mais altos = menos sensível)
DISTRACTION_CONSEC_FRAMES = 25     # Frames consecutivos para alarme (Valores mais altos = menos sensível)

# Índices Dlib para os olhos
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

class DriverMonitor:
    """
    Classe principal que encapsula toda a lógica de deteção de IA (apenas Dlib).
    """
    
    def __init__(self, frame_size):
        logging.info("A inicializar o DriverMonitor Core...")
        
        self.frame_height = frame_size[0]
        self.frame_width = frame_size[1]

        # Modelos Dlib
        self.dlib_detector = None
        self.dlib_predictor = None
        
        # Contadores de Alerta
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        
        self.initialize_dlib_models()

    def initialize_dlib_models(self):
        """Carrega o detetor de face e o preditor de landmarks do Dlib."""
        try:
            logging.info(">>> Carregando detetor de faces do Dlib...")
            self.dlib_detector = dlib.get_frontal_face_detector()
            
            logging.info(f">>> Carregando preditor de landmarks faciais ({DLIB_MODEL_PATH})...")
            self.dlib_predictor = dlib.shape_predictor(DLIB_MODEL_PATH)
            
            logging.info(">>> Modelos Dlib carregados com sucesso.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelos Dlib: {e}", exc_info=True)
            raise e

    def _eye_aspect_ratio(self, eye):
        """Calcula a distância euclidiana entre os pontos verticais e horizontais do olho."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def _estimate_head_pose(self, shape_np):
        """Estima a pose da cabeça (para onde o motorista está a olhar)."""
        
        model_points = np.array([
                                (0.0, 0.0, 0.0),      # Ponta do nariz (30)
                                (0.0, -330.0, -65.0),  # Queixo (8)
                                (-225.0, 170.0, -135.0), # Canto do olho esquerdo (36)
                                (225.0, 170.0, -135.0),  # Canto do olho direito (45)
                                (-150.0, -150.0, -125.0), # Canto da boca esquerdo (48)
                                (150.0, -150.0, -125.0)  # Canto da boca direito (54)
                            ])
        
        focal_length = self.frame_width
        center = (self.frame_width / 2, self.frame_height / 2)
        camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = "double"
                                )
        
        dist_coeffs = np.zeros((4,1)) 

        image_points = np.array([
                                shape_np[30], 
                                shape_np[8],  
                                shape_np[36], 
                                shape_np[45], 
                                shape_np[48], 
                                shape_np[54]  
                            ], dtype="double")
        
        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            
            yaw = angles[1]   
            pitch = angles[0] 
            roll = angles[2]  
            
            return yaw, pitch, roll
        except Exception as e:
            logging.debug(f"Erro ao calcular pose da cabeça: {e}")
            return 0, 0, 0

    def _shape_to_np(self, shape):
        """Converte o objeto de landmarks do Dlib para um array NumPy."""
        coords = np.zeros((shape.num_parts, 2), dtype="int")
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def process_frame(self, frame, gray):
        """
        Função principal que processa um único frame.
        """
        
        # --- 1. Deteção de Rosto e Landmarks (Dlib) ---
        rects = self.dlib_detector(gray, 0)
        
        alarm_drowsy = False
        alarm_distraction = False

        if len(rects) > 0:
            rect = rects[0]
            shape = self.dlib_predictor(gray, rect)
            shape_np = self._shape_to_np(shape)

            # --- 1a. Verificação de Sonolência (EAR) ---
            leftEye = shape_np[lStart:lEnd]
            rightEye = shape_np[rStart:rEnd]
            leftEAR = self._eye_aspect_ratio(leftEye)
            rightEAR = self._eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EAR_THRESHOLD:
                self.drowsiness_counter += 1
                if self.drowsiness_counter >= EAR_CONSEC_FRAMES:
                    alarm_drowsy = True
                    logging.warning(f"DETEÇÃO DE SONOLÊNCIA (EAR: {ear:.2f})")
            else:
                self.drowsiness_counter = 0

            # --- 1b. Verificação de Distração (Pose da Cabeça) ---
            yaw, pitch, roll = self._estimate_head_pose(shape_np)
            
            if abs(yaw) > DISTRACTION_THRESHOLD_ANGLE or pitch > DISTRACTION_THRESHOLD_ANGLE:
                self.distraction_counter += 1
                if self.distraction_counter >= DISTRACTION_CONSEC_FRAMES:
                    alarm_distraction = True
                    logging.warning(f"DETEÇÃO DE DISTRAÇÃO (Yaw: {yaw:.1f}, Pitch: {pitch:.1f})")
            else:
                self.distraction_counter = 0
                
            cv2.putText(frame, f"EAR: {ear:.2f}", (self.frame_width - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (self.frame_width - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        else:
            # Se nenhum rosto for detetado, reinicia os contadores
            self.drowsiness_counter = 0
            self.distraction_counter = 0


        # --- 4. Desenha os Alertas Visuais Finais ---
        if alarm_drowsy:
             cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if alarm_distraction:
             cv2.putText(frame, "ALERTA: DISTRACAO!", (10, 60),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame, (alarm_drowsy, alarm_distraction, False) # Retorna 'False' para o alarme de celular

