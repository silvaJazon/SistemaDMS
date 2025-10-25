# Documentação: Núcleo de Processamento do SistemaDMS
# Contém a classe 'DriverMonitor' que encapsula toda a lógica
# de deteção de face, landmarks, EAR (sonolência) e pose da cabeça (distração).

import cv2
import dlib
import numpy as np
import logging
import sys
import os
from scipy.spatial import distance as dist
import math

# --- Configurações de Alerta ---
# Estas configurações podem ser movidas para um ficheiro de config ou para a UI no futuro
EAR_THRESHOLD = 0.25 # Limite do Eye Aspect Ratio para considerar "fechado"
EAR_CONSEC_FRAMES = 15 # Número de frames consecutivos com olhos fechados para disparar o alarme

DISTRACTION_THRESHOLD_ANGLE = 30.0 # Ângulo (em graus) para considerar "distraído"
DISTRACTION_CONSEC_FRAMES = 25 # Número de frames consecutivos distraído para disparar o alarme

# Índices dos landmarks do Dlib
EYE_AR_LEFT_START = 42
EYE_AR_LEFT_END = 48
EYE_AR_RIGHT_START = 36
EYE_AR_RIGHT_END = 42

MODEL_PATH = 'shape_predictor_68_face_landmarks.dat' # Modelo Dlib

class DriverMonitor:
    """
    Classe principal para o monitoramento do motorista.
    Engloba a inicialização dos modelos e o processamento de cada frame.
    """
    
    def __init__(self):
        logging.info("A inicializar o DriverMonitor Core...")
        
        # Índices dos landmarks
        self.lStart = EYE_AR_LEFT_START
        self.lEnd = EYE_AR_LEFT_END
        self.rStart = EYE_AR_RIGHT_START
        self.rEnd = EYE_AR_RIGHT_END

        # Contadores de Alerta
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        
        # Modelos
        self.detector = None
        self.predictor = None
        self.initialize_model()

    def initialize_model(self):
        """Carrega o detetor de face e o preditor de landmarks do Dlib."""
        try:
            logging.info(">>> Carregando detetor de faces do Dlib...")
            self.detector = dlib.get_frontal_face_detector()
            
            logging.info(f">>> Carregando preditor de landmarks faciais ({MODEL_PATH})...")
            if not os.path.exists(MODEL_PATH):
                logging.error(f"!!! ERRO FATAL: Modelo de landmarks '{MODEL_PATH}' não encontrado.")
                raise FileNotFoundError
                
            self.predictor = dlib.shape_predictor(MODEL_PATH)
            logging.info(">>> Modelos Dlib carregados com sucesso.")
            
        except FileNotFoundError:
            logging.error("O ficheiro do modelo não foi encontrado. O Dockerfile descarregou-o?")
            sys.exit(1)
        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelos Dlib: {e}", exc_info=True)
            sys.exit(1)

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
        ear = (A + B) / (2.0 * C)
        return ear

    def _estimate_head_pose(self, shape_np, frame_size):
        """Estima a pose da cabeça (para onde o motorista está a olhar)."""
        
        # Pontos de referência faciais 3D (modelo genérico)
        model_points = np.array([
                                (0.0, 0.0, 0.0),             # Ponta do nariz (30)
                                (0.0, -330.0, -65.0),        # Queixo (8)
                                (-225.0, 170.0, -135.0),     # Canto do olho esquerdo (36)
                                (225.0, 170.0, -135.0),      # Canto do olho direito (45)
                                (-150.0, -150.0, -125.0),    # Canto da boca esquerdo (48)
                                (150.0, -150.0, -125.0)     # Canto da boca direito (54)
                            ])
        
        # Parâmetros da câmara (assumidos)
        focal_length = frame_size[1]
        center = (frame_size[1]/2, frame_size[0]/2)
        camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = "double"
                                )
        
        dist_coeffs = np.zeros((4,1)) # Assumindo sem distorção de lente

        # Pontos 2D correspondentes do Dlib
        image_points = np.array([
                                shape_np[30],    # Ponta do nariz
                                shape_np[8],     # Queixo
                                shape_np[36],    # Canto do olho esquerdo
                                shape_np[45],    # Canto do olho direito
                                shape_np[48],    # Canto da boca esquerdo
                                shape_np[54]     # Canto da boca direito
                            ], dtype="double")
        
        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

            # Ângulos em graus (Yaw é o mais importante para distração lateral)
            yaw = angles[1]
            pitch = angles[0]
            roll = angles[2]
            
            return yaw, pitch, roll
        except Exception as e:
            logging.debug(f"Erro ao calcular pose da cabeça: {e}")
            return 0, 0, 0 # Retorna 0 se falhar

    def process_frame(self, frame_display, gray):
        """
        Processa um único frame para deteção de sonolência e distração.
        Retorna o frame com as anotações e um dicionário de status.
        """
        
        # Deteção de Faces
        rects = self.detector(gray, 0)
        
        alarm_drowsy = False
        alarm_distraction = False
        status_data = {}

        # Loop sobre as faces detetadas
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape_np = self._shape_to_np(shape)

            # --- 1. Verificação de Sonolência (EAR) ---
            leftEye = shape_np[self.lStart:self.lEnd]
            rightEye = shape_np[self.rStart:self.rEnd]
            leftEAR = self._eye_aspect_ratio(leftEye)
            rightEAR = self._eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            status_data['ear'] = ear

            # Desenha os contornos dos olhos
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame_display, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame_display, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EAR_THRESHOLD:
                self.drowsiness_counter += 1
                if self.drowsiness_counter >= EAR_CONSEC_FRAMES:
                    alarm_drowsy = True
                    logging.warning(f"DETEÇÃO DE SONOLÊNCIA (EAR: {ear:.2f})")
            else:
                self.drowsiness_counter = 0

            # --- 2. Verificação de Distração (Pose da Cabeça) ---
            frame_size = frame_display.shape[:2]
            yaw, pitch, roll = self._estimate_head_pose(shape_np, frame_size)
            
            status_data['yaw'] = yaw
            status_data['pitch'] = pitch
            
            # Verifica se está a olhar para os lados (Yaw) ou muito para baixo (Pitch)
            if abs(yaw) > DISTRACTION_THRESHOLD_ANGLE or pitch > DISTRACTION_THRESHOLD_ANGLE:
                self.distraction_counter += 1
                if self.distraction_counter >= DISTRACTION_CONSEC_FRAMES:
                    alarm_distraction = True
                    logging.warning(f"DETEÇÃO DE DISTRAÇÃO (Yaw: {yaw:.1f}, Pitch: {pitch:.1f})")
            else:
                self.distraction_counter = 0
                
            # Desenha informações de debug no frame
            cv2.putText(frame_display, f"EAR: {ear:.2f}", (frame_display.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Yaw: {yaw:.1f}", (frame_display.shape[1] - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Se nenhuma face for detetada, reinicia os contadores
        if not rects:
            self.drowsiness_counter = 0
            self.distraction_counter = 0

        # Desenha os Alertas Visuais Finais
        if alarm_drowsy:
            cv2.putText(frame_display, "ALERTA: SONOLENCIA!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if alarm_distraction:
            cv2.putText(frame_display, "ALERTA: DISTRACAO!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        status_data['alarm_drowsy'] = alarm_drowsy
        status_data['alarm_distraction'] = alarm_distraction

        # Retorna o frame processado e os dados de status
        return frame_display, status_data
