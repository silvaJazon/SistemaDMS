# Documentação: Núcleo do Driver Monitor System (O "Cérebro")
# Responsável por:
# 1. Carregar os modelos Dlib.
# 2. Receber frames e analisá-los (EAR, Pose).
# 3. Manter e atualizar as configurações de deteção.
# 4. Desenhar no frame e retornar os eventos de alerta.

import cv2
import dlib
import numpy as np
import logging
import math
import threading
from scipy.spatial import distance as dist

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

class DriverMonitor:
    """
    Classe principal para toda a lógica de deteção de IA.
    """
    
    def __init__(self, frame_size):
        logging.info("A inicializar o DriverMonitor Core...")
        
        # --- Modelos de IA ---
        self.detector = None
        self.predictor = None
        
        # --- Índices Dlib ---
        self.EYE_AR_LEFT_START = 42
        self.EYE_AR_LEFT_END = 48
        self.EYE_AR_RIGHT_START = 36
        self.EYE_AR_RIGHT_END = 42
        (self.lStart, self.lEnd) = (self.EYE_AR_LEFT_START, self.EYE_AR_LEFT_END)
        (self.rStart, self.rEnd) = (self.EYE_AR_RIGHT_START, self.EYE_AR_RIGHT_END)

        # --- Parâmetros de Pose da Cabeça ---
        self.frame_height, self.frame_width = frame_size
        self.model_points = np.array([
            (0.0, 0.0, 0.0),      # Ponta do nariz (30)
            (0.0, -330.0, -65.0),  # Queixo (8)
            (-225.0, 170.0, -135.0), # Canto do olho esquerdo (36)
            (225.0, 170.0, -135.0),  # Canto do olho direito (45)
            (-150.0, -150.0, -125.0), # Canto da boca esquerdo (48)
            (150.0, -150.0, -125.0)   # Canto da boca direito (54)
        ])
        focal_length = self.frame_width
        center = (self.frame_width / 2, self.frame_height / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        self.dist_coeffs = np.zeros((4, 1)) # Assumindo sem distorção

        # --- Configurações Dinâmicas (Valores Padrão) ---
        # Estes valores serão controlados pela API
        self.lock = threading.Lock()
        self.ear_threshold = 0.25
        self.ear_consec_frames = 15
        self.distraction_threshold_angle = 30.0
        self.distraction_consec_frames = 25
        
        # --- Contadores de Estado ---
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        
        self.initialize_model()

    def update_settings(self, settings):
        """Atualiza as configurações de deteção de forma segura (thread-safe)."""
        with self.lock:
            try:
                self.ear_threshold = float(settings.get('ear_threshold', self.ear_threshold))
                self.ear_consec_frames = int(settings.get('ear_consec_frames', self.ear_consec_frames))
                self.distraction_threshold_angle = float(settings.get('distraction_threshold_angle', self.distraction_threshold_angle))
                self.distraction_consec_frames = int(settings.get('distraction_consec_frames', self.distraction_consec_frames))
            except Exception as e:
                logging.error(f"Erro ao atualizar configurações: {e}", exc_info=True)

    def get_settings(self):
        """Retorna as configurações atuais de forma segura (thread-safe)."""
        with self.lock:
            return {
                "ear_threshold": self.ear_threshold,
                "ear_consec_frames": self.ear_consec_frames,
                "distraction_threshold_angle": self.distraction_threshold_angle,
                "distraction_consec_frames": self.distraction_consec_frames,
            }

    def initialize_model(self):
        """Carrega o detetor de face e o preditor de landmarks do Dlib."""
        try:
            logging.info(">>> Carregando detetor de faces do Dlib...")
            self.detector = dlib.get_frontal_face_detector()
            logging.info(">>> Carregando preditor de landmarks faciais (shape_predictor_68_face_landmarks.dat)...")
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            logging.info(">>> Modelos Dlib carregados com sucesso.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelos Dlib: {e}", exc_info=True)
            raise e # Propaga o erro para parar a aplicação

    # --- Funções Auxiliares de Deteção ---

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

    def _estimate_head_pose(self, shape, frame_size):
        """Estima a pose da cabeça (para onde o motorista está a olhar)."""
        image_points = np.array([
            shape[30], # Ponta do nariz
            shape[8],  # Queixo
            shape[36], # Canto do olho esquerdo
            shape[45], # Canto do olho direito
            shape[48], # Canto da boca esquerdo
            shape[54]  # Canto da boca direito
        ], dtype="double")
        
        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            
            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            
            yaw = angles[1]   # Olhar para os lados (positivo = direita)
            pitch = angles[0] # Olhar para cima/baixo (positivo = baixo)
            roll = angles[2]  # Inclinação da cabeça
            
            return yaw, pitch, roll
        except Exception as e:
            logging.debug(f"Erro ao calcular pose da cabeça: {e}")
            return 0, 0, 0 # Retorna 0 se falhar

    # --- Função Principal de Processamento ---

    def process_frame(self, frame, gray):
        """
        Função principal chamada pelo app.py.
        Processa um frame e retorna o frame com desenhos + lista de eventos.
        """
        
        # (NOVO) Lista de eventos para esta frame
        events = []
        
        # Lê as configurações atuais de forma segura
        current_settings = self.get_settings()
        
        # Tenta melhorar o contraste para câmaras IR
        gray = cv2.equalizeHist(gray) 
        
        # Deteção de Faces
        rects = self.detector(gray, 0)
        
        alarm_drowsy = False
        alarm_distraction = False

        if not rects:
            # Se nenhuma face for detetada, reinicia os contadores
            self.drowsiness_counter = 0
            self.distraction_counter = 0
        
        # Loop sobre as faces detetadas (deve ser apenas 1, o motorista)
        for rect in rects:
            shape = self.predictor(gray, rect)
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
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < current_settings['ear_threshold']:
                self.drowsiness_counter += 1
                if self.drowsiness_counter >= current_settings['ear_consec_frames']:
                    alarm_drowsy = True
                    # (NOVO) Adiciona evento
                    events.append({
                        "type": "SONOLENCIA",
                        "details": {"ear": round(ear, 2), "frames": self.drowsiness_counter}
                    })
            else:
                self.drowsiness_counter = 0

            # --- 2. Verificação de Distração (Pose da Cabeça) ---
            yaw, pitch, roll = self._estimate_head_pose(shape, (self.frame_height, self.frame_width))
            
            # Verifica se está a olhar para os lados (Yaw) ou muito para baixo (Pitch)
            if (abs(yaw) > current_settings['distraction_threshold_angle'] or 
                pitch > current_settings['distraction_threshold_angle']): # Usamos o mesmo threshold para Yaw e Pitch
                
                self.distraction_counter += 1
                if self.distraction_counter >= current_settings['distraction_consec_frames']:
                    alarm_distraction = True
                    # (NOVO) Adiciona evento
                    events.append({
                        "type": "DISTRACAO",
                        "details": {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "frames": self.distraction_counter}
                    })
            else:
                self.distraction_counter = 0
                
            # Desenha informações de debug no frame
            cv2.putText(frame, f"EAR: {ear:.2f}", (self.frame_width - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (self.frame_width - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (self.frame_width - 150, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Desenha os Alertas Visuais Finais
        if alarm_drowsy:
            cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if alarm_distraction:
            cv2.putText(frame, "ALERTA: DISTRACAO!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # (ALTERADO) Retorna o frame processado E a lista de eventos
        return frame, events

