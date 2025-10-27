# Documentação: Núcleo do SistemaDMS (Driver Monitor System)
# Responsável por toda a lógica de análise de imagem (IA).
# (Atualizado com lógica de "cooldown" para eventos)

import cv2
import dlib
import numpy as np
import logging
import math
import threading
from scipy.spatial import distance as dist
from datetime import datetime

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

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
        self.lock = threading.Lock() # Protege as configurações
        
        # Contadores de frames
        self.drowsiness_counter = 0
        self.distraction_counter = 0

        # --- (NOVO) Lógica de Cooldown de Alerta ---
        # Estes booleanos controlam se um alerta já está ATIVO,
        # para evitar disparar 100 eventos para 100 frames seguidos.
        self.drowsy_alert_active = False
        self.distraction_alert_active = False
        
        # --- Configurações de Calibração (Valores Padrão) ---
        self.ear_threshold = 0.25      # Limite do Eye Aspect Ratio
        self.ear_frames = 15           # Nº de frames consecutivos para alarme de sonolência
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
            sys.exit(1)

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
        
        if C == 0: return 0.3 # Evita divisão por zero
        
        ear = (A + B) / (2.0 * C)
        return ear

    def _estimate_head_pose(self, shape_np):
        """Estima a pose da cabeça (para onde o motorista está a olhar)."""
        
        # Pontos 2D correspondentes do Dlib
        image_points = np.array([shape_np[i] for i in self.HEAD_IMAGE_POINTS_IDX], dtype="double")
        
        try:
            # Resolve a pose da cabeça (PnP)
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                self.HEAD_MODEL_POINTS, 
                image_points, 
                self.camera_matrix, 
                self.dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )

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

    # --- Métodos Principais (Processamento e Configuração) ---

    def process_frame(self, frame, gray):
        """
        Função principal de processamento. Analisa um frame e retorna o
        frame processado e quaisquer eventos de alerta.
        """
        
        events_list = [] # Lista para guardar os eventos DESTE frame
        status_data = {} # Dicionário para enviar status para a UI
        
        # (NOVO) Aplica equalização de histograma para melhorar contraste (especialmente útil para IR)
        gray_processed = cv2.equalizeHist(gray)
        
        # Deteção de Faces
        rects = self.detector(gray_processed, 0)
        
        # (NOVO) Se nenhuma face for detetada, reinicia os contadores e o estado de "cooldown"
        if not rects:
            self.drowsiness_counter = 0
            self.distraction_counter = 0
            self.drowsy_alert_active = False
            self.distraction_alert_active = False

        # Loop sobre as faces detetadas (deve ser apenas 1, o motorista)
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape_np = self._shape_to_np(shape)

            # --- 1. Verificação de Sonolência (EAR) ---
            leftEye = shape_np[self.EYE_AR_LEFT_START:self.EYE_AR_LEFT_END]
            rightEye = shape_np[self.EYE_AR_RIGHT_START:self.EYE_AR_RIGHT_END]
            leftEAR = self._eye_aspect_ratio(leftEye)
            rightEAR = self._eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Desenha os contornos dos olhos (debug visual)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            with self.lock: # Garante que as configurações não mudem durante o cálculo
                # --- Lógica de Sonolência (com Cooldown) ---
                if ear < self.ear_threshold:
                    self.drowsiness_counter += 1
                    
                    if self.drowsiness_counter >= self.ear_frames and not self.drowsy_alert_active:
                        # (NOVO) DISPARA O ALERTA (SÓ UMA VEZ)
                        self.drowsy_alert_active = True # Ativa o "cooldown"
                        events_list.append({
                            "type": "SONOLENCIA",
                            "value": f"EAR: {ear:.2f}",
                            "timestamp": datetime.now().isoformat() + "Z"
                        })
                else:
                    # (NOVO) REARMA O ALERTA
                    self.drowsiness_counter = 0
                    self.drowsy_alert_active = False

                # --- 2. Verificação de Distração (Pose da Cabeça) ---
                yaw, pitch, roll = self._estimate_head_pose(shape_np)
                
                # Verifica se está a olhar para os lados (Yaw) ou muito para baixo (Pitch)
                # (O Pitch para baixo costuma ter ângulos positivos altos)
                if abs(yaw) > self.distraction_angle or pitch > (self.distraction_angle + 10):
                    self.distraction_counter += 1
                    
                    if self.distraction_counter >= self.distraction_frames and not self.distraction_alert_active:
                        # (NOVO) DISPARA O ALERTA (SÓ UMA VEZ)
                        self.distraction_alert_active = True # Ativa o "cooldown"
                        events_list.append({
                            "type": "DISTRACAO",
                            "value": f"Yaw: {yaw:.1f}, Pitch: {pitch:.1f}",
                            "timestamp": datetime.now().isoformat() + "Z"
                        })
                else:
                    # (NOVO) REARMA O ALERTA
                    self.distraction_counter = 0
                    self.distraction_alert_active = False
            
            # --- Desenha Alertas Visuais Finais ---
            # (O alerta VISUAL permanece ativo enquanto a condição persistir)
            if self.drowsy_alert_active:
                cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if self.distraction_alert_active:
                cv2.putText(frame, "ALERTA: DISTRACAO!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Atualiza dados de status para a UI
            status_data = {
                "ear": f"{ear:.2f}",
                "yaw": f"{yaw:.1f}",
                "pitch": f"{pitch:.1f}",
                "roll": f"{roll:.1f}"
            }

        return frame, events_list, status_data

    def update_settings(self, settings):
        """Atualiza as configurações de calibração de forma segura (thread-safe)."""
        with self.lock:
            try:
                self.ear_threshold = float(settings.get('ear_threshold', self.ear_threshold))
                self.ear_frames = int(settings.get('ear_frames', self.ear_frames))
                self.distraction_angle = float(settings.get('distraction_angle', self.distraction_angle))
                self.distraction_frames = int(settings.get('distraction_frames', self.distraction_frames))
                logging.info(f"Configurações do DMS atualizadas: {settings}")
                return True
            except Exception as e:
                logging.error(f"Erro ao atualizar configurações do DMS: {e}")
                return False

    def get_settings(self):
        """Obtém as configurações atuais de forma segura (thread-safe)."""
        with self.lock:
            return {
                "ear_threshold": self.ear_threshold,
                "ear_frames": self.ear_frames,
                "distraction_angle": self.distraction_angle,
                "distraction_frames": self.distraction_frames
            }

