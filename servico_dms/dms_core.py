# Documentação: Núcleo do Driver Monitor System (DMS)
# Responsável por toda a lógica de Visão Computacional (Dlib).
# Agora suporta configuração dinâmica (alterar parâmetros em tempo real).

import cv2
import dlib
import numpy as np
import logging
import math
from scipy.spatial import distance as dist
import threading # NOVO: Adicionado lock para segurança da thread

# Habilita otimizações do OpenCV
cv2.setUseOptimized(True)

# --- Constantes ---
MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'
DLIB_UPSAMPLE = 0 # Quantas vezes fazer 'upsample' na imagem (0 = mais rápido)

# Índices dos landmarks (pontos) do Dlib
EYE_AR_LEFT_START = 42
EYE_AR_LEFT_END = 48
EYE_AR_RIGHT_START = 36
EYE_AR_RIGHT_END = 42

class DriverMonitor:
    """
    Esta classe gere toda a lógica de deteção (Rosto, Olhos, Pose).
    """

    def __init__(self, frame_size):
        logging.info("A inicializar o DriverMonitor Core...")
        
        # --- Configurações de Calibração Dinâmica ---
        # Estes são os valores padrão que serão carregados no arranque.
        # Eles podem ser alterados em tempo real pelo método update_settings()
        self.lock = threading.Lock() # Protege o acesso a estas variáveis
        self.ear_threshold = 0.25      # Limite do Eye Aspect Ratio
        self.ear_consec_frames = 15    # Frames consecutivos para alarme de sonolência
        self.distraction_threshold_angle = 30.0 # Ângulo (graus) para alarme de distração
        self.distraction_consec_frames = 25     # Frames consecutivos para alarme de distração
        
        # --- Contadores de Alerta ---
        self.drowsiness_counter = 0    # Contador para sonolência
        self.distraction_counter = 0   # Contador para distração
        
        # --- Parâmetros Internos ---
        self.frame_height, self.frame_width = frame_size
        self.detector = None
        self.predictor = None
        self.is_ready = False
        
        # Índices Dlib (atalhos)
        self.lStart, self.lEnd = (EYE_AR_LEFT_START, EYE_AR_LEFT_END)
        self.rStart, self.rEnd = (EYE_AR_RIGHT_START, EYE_AR_RIGHT_END)

        # Parâmetros da câmara (usados para estimar a pose)
        focal_length = self.frame_width
        cam_center = (self.frame_width / 2, self.frame_height / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, cam_center[0]],
             [0, focal_length, cam_center[1]],
             [0, 0, 1]], dtype="double"
        )
        # Modelo 3D genérico do rosto
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),       # Ponta do nariz (30)
            (0.0, -330.0, -65.0),    # Queixo (8)
            (-225.0, 170.0, -135.0), # Canto do olho esquerdo (36)
            (225.0, 170.0, -135.0),  # Canto do olho direito (45)
            (-150.0, -150.0, -125.0), # Canto da boca esquerdo (48)
            (150.0, -150.0, -125.0)   # Canto da boca direito (54)
        ])
        
        # Carrega os modelos Dlib
        self.initialize_dlib_models()

    def initialize_dlib_models(self):
        """Carrega o detetor de face e o preditor de landmarks."""
        try:
            logging.info(">>> Carregando detetor de faces do Dlib...")
            self.detector = dlib.get_frontal_face_detector()
            logging.info(f">>> Carregando preditor de landmarks faciais ({MODEL_PATH})...")
            self.predictor = dlib.shape_predictor(MODEL_PATH)
            logging.info(">>> Modelos Dlib carregados com sucesso.")
            self.is_ready = True
        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelos Dlib ({MODEL_PATH}): {e}", exc_info=True)
            self.is_ready = False
            
    # --- NOVO: Métodos de Configuração ---
    
    def update_settings(self, ear_thresh, ear_frames, distraction_angle, distraction_frames):
        """
        Atualiza os parâmetros de deteção de forma segura (thread-safe).
        Chamado pela API do Flask.
        """
        with self.lock:
            self.ear_threshold = float(ear_thresh)
            self.ear_consec_frames = int(ear_frames)
            self.distraction_threshold_angle = float(distraction_angle)
            self.distraction_consec_frames = int(distraction_frames)
            
            # Reinicia os contadores para evitar falsos positivos com as novas regras
            self.drowsiness_counter = 0
            self.distraction_counter = 0
            
        logging.info(f"Configurações atualizadas: EAR_T={self.ear_threshold}, EAR_F={self.ear_consec_frames}, "
                     f"DIST_A={self.distraction_threshold_angle}, DIST_F={self.distraction_consec_frames}")

    def get_settings(self):
        """
        Retorna as configurações atuais de forma segura.
        Chamado pela API do Flask.
        """
        with self.lock:
            return {
                "ear_threshold": self.ear_threshold,
                "ear_consec_frames": self.ear_consec_frames,
                "distraction_threshold_angle": self.distraction_threshold_angle,
                "distraction_consec_frames": self.distraction_consec_frames
            }
    
    # --- Funções Auxiliares de Deteção ---

    def _eye_aspect_ratio(self, eye):
        """Calcula a distância euclidiana entre os pontos verticais e horizontais do olho."""
        # distâncias verticais
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # distância horizontal
        C = dist.euclidean(eye[0], eye[3])
        # Fórmula do EAR
        ear = (A + B) / (2.0 * C)
        return ear

    def _shape_to_np(self, shape, dtype="int"):
        """Converte o objeto de landmarks do Dlib para um array NumPy."""
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _estimate_head_pose(self, shape_np):
        """Estima a pose da cabeça (para onde o motorista está a olhar)."""
        
        # Pontos 2D correspondentes do Dlib
        image_points = np.array([
            shape_np[30], # Ponta do nariz
            shape_np[8],  # Queixo
            shape_np[36], # Canto do olho esquerdo
            shape_np[45], # Canto do olho direito
            shape_np[48], # Canto da boca esquerdo
            shape_np[54]  # Canto da boca direito
        ], dtype="double")
        
        dist_coeffs = np.zeros((4,1)) # Assumindo sem distorção de lente

        try:
            # Resolve a pose da cabeça
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_3d, 
                image_points, 
                self.camera_matrix, 
                dist_coeffs, 
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
            
    def _apply_ir_contrast(self, gray_frame):
        """
        Aplica equalização de histograma para melhorar o contraste, 
        especialmente útil para imagens de câmaras IR (Infravermelhos).
        """
        try:
            # Aumenta o contraste
            return cv2.equalizeHist(gray_frame)
        except cv2.error as e:
            # logging.debug(f"Falha na equalização do histograma (provavelmente frame vazio): {e}")
            return gray_frame # Retorna o original se falhar

    # --- Função Principal de Processamento ---

    def process_frame(self, frame, gray):
        """
        Função principal. Recebe um frame e processa-o.
        Retorna o frame com os desenhos (visualização) e os dados de status.
        """
        
        if not self.is_ready:
            # Se os modelos não carregaram, não faz nada
            return frame, {}

        # --- NOVO: Lê os valores de configuração de forma segura ---
        # Copia os valores no início do loop para garantir consistência
        with self.lock:
            ear_thresh = self.ear_threshold
            ear_frames = self.ear_consec_frames
            distraction_angle = self.distraction_threshold_angle
            distraction_frames = self.distraction_consec_frames
        # -----------------------------------------------------------

        # Aplica a melhoria de contraste (para IR)
        # Se a câmara não for IR, esta função tem pouco efeito.
        gray = self._apply_ir_contrast(gray)
        
        # 1. Deteção de Faces (Dlib)
        # DLIB_UPSAMPLE=0 é o mais rápido
        rects = self.detector(gray, DLIB_UPSAMPLE)
        
        alarm_drowsy = False
        alarm_distraction = False
        status_data = {} # Dados para a API

        # Loop sobre as faces detetadas (deve ser apenas 1, o motorista)
        if len(rects) > 0:
            rect = rects[0] # Pega apenas a primeira face
            
            # 2. Deteção de Landmarks (Pontos Faciais)
            shape_dlib = self.predictor(gray, rect)
            shape = self._shape_to_np(shape_dlib)

            # --- 3. Verificação de Sonolência (EAR) ---
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self._eye_aspect_ratio(leftEye)
            rightEAR = self._eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Desenha os contornos dos olhos (visualização)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Lógica do alarme de sonolência
            if ear < ear_thresh:
                self.drowsiness_counter += 1
                if self.drowsiness_counter >= ear_frames:
                    alarm_drowsy = True
                    logging.warning(f"DETEÇÃO DE SONOLÊNCIA (EAR: {ear:.2f})")
            else:
                self.drowsiness_counter = 0

            # --- 4. Verificação de Distração (Pose da Cabeça) ---
            yaw, pitch, roll = self._estimate_head_pose(shape)
            
            # Lógica do alarme de distração
            # (Verifica se está a olhar para os lados ou muito para baixo)
            if abs(yaw) > distraction_angle or pitch > distraction_angle:
                self.distraction_counter += 1
                if self.distraction_counter >= distraction_frames:
                    alarm_distraction = True
                    logging.warning(f"DETEÇÃO DE DISTRAÇÃO (Yaw: {yaw:.1f}, Pitch: {pitch:.1f})")
            else:
                self.distraction_counter = 0
                
            # Prepara dados para a API/Desenho
            status_data['ear'] = ear
            status_data['yaw'] = yaw
            status_data['pitch'] = pitch
            status_data['face_detected'] = True

            # Desenha informações de debug no frame
            cv2.putText(frame, f"EAR: {ear:.2f} (T: {ear_thresh})", (self.frame_width - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f} (T: {distraction_angle})", (self.frame_width - 250, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        else:
            # Se nenhuma face for detetada, reinicia os contadores
            self.drowsiness_counter = 0
            self.distraction_counter = 0
            status_data['face_detected'] = False
            cv2.putText(frame, "ROSTO NAO DETETADO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 5. Desenha os Alertas Visuais Finais
        if alarm_drowsy:
            cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if alarm_distraction:
            cv2.putText(frame, "ALERTA: DISTRACAO!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame, status_data

