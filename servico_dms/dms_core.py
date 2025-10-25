# Documentação: Núcleo do Driver Monitor System (DMS)
# Responsável por:
# 1. Carregar os modelos Dlib (detetor de face e preditor de landmarks).
# 2. Calcular o EAR (Eye Aspect Ratio) para sonolência.
# 3. Estimar a Pose da Cabeça para distração.
# 4. Processar um frame de vídeo e retornar os resultados.

import cv2
import time
import os
import numpy as np
import logging 
import sys
import dlib # Biblioteca para deteção de face e landmarks
from scipy.spatial import distance as dist # Para calcular a distância euclidiana
import math # Para a matemática da Pose da CabeA

# --- Configurações de Alerta ---

# 1. Sonolência (EAR)
EAR_THRESHOLD = 0.25      # Limite do Eye Aspect Ratio (Valores mais baixos = mais difícil de detetar)
EAR_CONSEC_FRAMES = 15    # Frames consecutivos para alarme (Valores mais altos = menos sensível)

# 2. Distração (Pose da Cabeça)
DISTRACTION_THRESHOLD_ANGLE = 30.0 # Ângulo (graus) para alarme (Valores mais altos = menos sensível)
DISTRACTION_CONSEC_FRAMES = 25     # Frames consecutivos para alarme (Valores mais altos = menos sensível)

# --- Constantes do Modelo Dlib ---
MODEL_PATH = 'shape_predictor_68_face_landmarks.dat' # Modelo Dlib
EYE_AR_LEFT_START = 42
EYE_AR_LEFT_END = 48
EYE_AR_RIGHT_START = 36
EYE_AR_RIGHT_END = 42

class DriverMonitor:
    """
    Classe principal que encapsula toda a lógica de deteção de IA.
    """
    
    def __init__(self, frame_size):
        logging.info("A inicializar o DriverMonitor Core...")
        
        # Parâmetros
        self.frame_height, self.frame_width = frame_size
        
        # Contadores de Alerta
        self.drowsiness_counter = 0     # Contador para sonolência
        self.distraction_counter = 0    # Contador para distração
        
        # Índices dos landmarks do Dlib
        self.lStart, self.lEnd = (EYE_AR_LEFT_START, EYE_AR_LEFT_END)
        self.rStart, self.rEnd = (EYE_AR_RIGHT_START, EYE_AR_RIGHT_END)
        
        # Modelos
        self.dlib_detector = None
        self.dlib_predictor = None
        self.initialize_dlib_model()

    # --- Funções de Inicialização ---
    
    def initialize_dlib_model(self):
        """Carrega o detetor de face e o preditor de landmarks do Dlib."""
        try:
            logging.info(">>> Carregando detetor de faces do Dlib...")
            self.dlib_detector = dlib.get_frontal_face_detector()
            
            logging.info(f">>> Carregando preditor de landmarks faciais ({MODEL_PATH})...")
            self.dlib_predictor = dlib.shape_predictor(MODEL_PATH)
            
            logging.info(">>> Modelos Dlib carregados com sucesso.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelos Dlib ({MODEL_PATH}): {e}", exc_info=True)
            raise e # Propaga o erro para o app.py

    # --- Funções de Deteção (Helpers) ---

    def _eye_aspect_ratio(self, eye):
        """Calcula a distância euclidiana entre os pontos verticais e horizontais do olho."""
        # Pontos verticais
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # Ponto horizontal
        C = dist.euclidean(eye[0], eye[3])
        
        # Fórmula do EAR
        ear = (A + B) / (2.0 * C)
        return ear

    def _estimate_head_pose(self, shape_landmarks, frame_size):
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
                                shape_landmarks[30],     # Ponta do nariz
                                shape_landmarks[8],      # Queixo
                                shape_landmarks[36],     # Canto do olho esquerdo
                                shape_landmarks[45],     # Canto do olho direito
                                shape_landmarks[48],     # Canto da boca esquerdo
                                shape_landmarks[54]      # Canto da boca direito
                            ], dtype="double")
        
        try:
            # Resolve a pose da cabeça
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            # Converte o vetor de rotação em ângulos de Euler (pitch, yaw, roll)
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

    def _shape_to_np(self, shape, dtype="int"):
        """Converte o objeto de landmarks do Dlib para um array NumPy."""
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(0, range(0, shape.num_parts)):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    # --- Função Principal de Processamento ---

    def process_frame(self, frame_bgr, frame_gray):
        """
        Função principal chamada pelo app.py.
        Recebe um frame a cores (BGR) e um frame a preto e branco (Gray).
        Processa o frame e retorna o frame desenhado e o status.
        """
        
        # Faz uma cópia do frame a cores para desenhar por cima
        frame_display = frame_bgr.copy()
        
        # Flags de Alerta
        alarm_drowsy = False
        alarm_distraction = False
        
        # --- NOVO: TENTATIVA DE MELHORIA PARA IR ---
        # Aplica Equalização de Histograma para aumentar o contraste
        # Isto pode ajudar o Dlib a "ver" as características em imagens IR
        try:
            frame_gray_processed = cv2.equalizeHist(frame_gray)
        except Exception:
            # Fallback caso a equalização falhe (ex: imagem totalmente preta)
            frame_gray_processed = frame_gray
        # -------------------------------------------
        
        # Deteção de Faces (usando o frame processado)
        rects = self.dlib_detector(frame_gray_processed, 0)
        
        # Loop sobre as faces detetadas (deve ser apenas 1, o motorista)
        if len(rects) > 0:
            rect = rects[0] # Pega apenas a primeira face
            
            # Usa o frame processado para encontrar os landmarks
            shape = self.dlib_predictor(frame_gray_processed, rect)
            shape = self._shape_to_np(shape)

            # --- 1. Verificação de Sonolência (EAR) ---
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self._eye_aspect_ratio(leftEye)
            rightEAR = self._eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

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
            yaw, pitch, roll = self._estimate_head_pose(shape, (self.frame_height, self.frame_width))
            
            # Verifica se está a olhar para os lados (Yaw) ou muito para baixo (Pitch)
            if abs(yaw) > DISTRACTION_THRESHOLD_ANGLE or pitch > DISTRACTION_THRESHOLD_ANGLE:
                self.distraction_counter += 1
                if self.distraction_counter >= DISTRACTION_CONSEC_FRAMES:
                    alarm_distraction = True
                    logging.warning(f"DETEÇÃO DE DISTRAÇÃO (Yaw: {yaw:.1f}, Pitch: {pitch:.1f})")
            else:
                self.distraction_counter = 0
                
            # Desenha informações de debug no frame
            cv2.putText(frame_display, f"EAR: {ear:.2f}", (self.frame_width - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Yaw: {yaw:.1f}", (self.frame_width - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # Se nenhuma face for detetada, reinicia os contadores
            self.drowsiness_counter = 0
            self.distraction_counter = 0
            logging.debug("Nenhuma face detetada.")

        # --- Desenha os Alertas Visuais Finais ---
        if alarm_drowsy:
            cv2.putText(frame_display, "ALERTA: SONOLENCIA!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if alarm_distraction:
            cv2.putText(frame_display, "ALERTA: DISTRACAO!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Prepara o dicionário de status
        status_data = {
            "timestamp": time.time(),
            "face_detected": len(rects) > 0,
            "alarm_drowsy": alarm_drowsy,
            "alarm_distraction": alarm_distraction
        }
        
        return frame_display, status_data

