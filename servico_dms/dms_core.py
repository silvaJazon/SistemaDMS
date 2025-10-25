# Documentação: Núcleo de Processamento do SistemaDMS
# Responsável por:
# 1. Carregar os modelos de IA (Dlib, MobileNet-SSD)
# 2. Processar os frames de vídeo para detetar (Sonolência, Distração, Celular)

import cv2
import time
import logging
import numpy as np
import dlib
from scipy.spatial import distance as dist
import math

# --- Constantes de Caminho dos Modelos ---
DLIB_MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'
MOBILENET_PROTOTXT_PATH = 'MobileNetSSD_deploy.prototxt'
MOBILENET_MODEL_PATH = 'MobileNetSSD_deploy.caffemodel'

# --- Constantes de Deteção ---

# 1. Sonolência (EAR)
EAR_THRESHOLD = 0.25      # Limite do Eye Aspect Ratio
EAR_CONSEC_FRAMES = 15    # Frames consecutivos para alarme

# 2. Distração (Pose da Cabeça)
DISTRACTION_THRESHOLD_ANGLE = 30.0 # Ângulo (graus) para alarme
DISTRACTION_CONSEC_FRAMES = 25     # Frames consecutivos para alarme

# 3. Celular (MobileNet-SSD)
# (MODO DEBUG) Baixamos a confiança para ver mais deteções
CELLPHONE_MIN_CONFIDENCE = 0.2 
CELLPHONE_CONSEC_FRAMES = 10     # Frames consecutivos para alarme

# --- Mapeamento de Classes MobileNet-SSD (PASCAL VOC) ---
# Usamos este modelo porque é leve, mas não tem "cell phone".
# Vamos procurar por "tvmonitor" (ID 20) como um substituto.
CLASSES_MOBILENET = {
    0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat",
    5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
    11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike",
    15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa",
    19: "train", 20: "tvmonitor" 
}
TARGET_CLASS_ID = 20 # ID de "tvmonitor"
TARGET_CLASS_NAME = "Monitor/Celular" 
# (MODO DEBUG) Cores para as caixas de depuração
DEBUG_COLORS = np.random.uniform(0, 255, size=(len(CLASSES_MOBILENET), 3))

# Índices Dlib para os olhos
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

class DriverMonitor:
    """
    Classe principal que encapsula toda a lógica de deteção de IA.
    """
    
    def __init__(self, frame_size):
        logging.info("A inicializar o DriverMonitor Core...")
        
        # Dimensões do frame
        self.frame_height = frame_size[0]
        self.frame_width = frame_size[1]

        # Modelos Dlib
        self.dlib_detector = None
        self.dlib_predictor = None
        
        # Modelos MobileNet-SSD
        self.mobilenet_net = None
        self.cellphone_min_conf = CELLPHONE_MIN_CONFIDENCE
        self.target_class_id = TARGET_CLASS_ID

        # Contadores de Alerta
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        self.cellphone_counter = 0
        
        # Inicializa todos os modelos
        self.initialize_dlib_models()
        self.initialize_mobilenet_model()

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
            raise e # Propaga o erro para o app.py

    def initialize_mobilenet_model(self):
        """Carrega o modelo MobileNet-SSD (Caffe)."""
        try:
            logging.info(f">>> Carregando modelo MobileNet-SSD (prototxt e caffemodel)...")
            self.mobilenet_net = cv2.dnn.readNetFromCaffe(MOBILENET_PROTOTXT_PATH, MOBILENET_MODEL_PATH)
            logging.info(">>> Modelo MobileNet-SSD carregado com sucesso.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelo MobileNet-SSD: {e}", exc_info=True)
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
        
        # Pontos de referência faciais 3D (modelo genérico)
        model_points = np.array([
                                (0.0, 0.0, 0.0),      # Ponta do nariz (30)
                                (0.0, -330.0, -65.0),  # Queixo (8)
                                (-225.0, 170.0, -135.0), # Canto do olho esquerdo (36)
                                (225.0, 170.0, -135.0),  # Canto do olho direito (45)
                                (-150.0, -150.0, -125.0), # Canto da boca esquerdo (48)
                                (150.0, -150.0, -125.0)  # Canto da boca direito (54)
                            ])
        
        # Parâmetros da câmara (assumidos)
        focal_length = self.frame_width
        center = (self.frame_width / 2, self.frame_height / 2)
        camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = "double"
                                )
        
        dist_coeffs = np.zeros((4,1)) # Assumindo sem distorção de lente

        # Pontos 2D correspondentes do Dlib
        image_points = np.array([
                                shape_np[30], # Ponta do nariz
                                shape_np[8],  # Queixo
                                shape_np[36], # Canto do olho esquerdo
                                shape_np[45], # Canto do olho direito
                                shape_np[48], # Canto da boca esquerdo
                                shape_np[54]  # Canto da boca direito
                            ], dtype="double")
        
        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            
            yaw = angles[1]   # Olhar para os lados
            pitch = angles[0] # Olhar para cima/baixo
            roll = angles[2]  # Inclinação
            
            return yaw, pitch, roll
        except Exception as e:
            logging.debug(f"Erro ao calcular pose da cabeça: {e}")
            return 0, 0, 0 # Retorna 0 se falhar

    def _shape_to_np(self, shape):
        """Converte o objeto de landmarks do Dlib para um array NumPy."""
        coords = np.zeros((shape.num_parts, 2), dtype="int")
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _detect_cell_phone(self, frame):
        """Executa a deteção de objetos com MobileNet-SSD."""
        try:
            # Cria um "blob" (imagem processada) para a rede neural
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            
            # Passa o blob pela rede
            self.mobilenet_net.setInput(blob)
            detections = self.mobilenet_net.forward()
            return detections
        except Exception as e:
            logging.warning(f"Falha ao executar a rede MobileNet-SSD: {e}")
            return None

    def process_frame(self, frame, gray):
        """
        Função principal que processa um único frame.
        Recebe o frame colorido (para MobileNet) e em escala de cinza (para Dlib).
        Retorna o frame com os desenhos e uma tupla de alarmes.
        """
        
        # --- 1. Deteção de Rosto e Landmarks ---
        rects = self.dlib_detector(gray, 0)
        
        alarm_drowsy = False
        alarm_distraction = False

        if len(rects) > 0:
            rect = rects[0] # Assume o primeiro rosto como o motorista
            shape = self.dlib_predictor(gray, rect)
            shape_np = self._shape_to_np(shape)

            # --- 1a. Verificação de Sonolência (EAR) ---
            leftEye = shape_np[lStart:lEnd]
            rightEye = shape_np[rStart:rEnd]
            leftEAR = self._eye_aspect_ratio(leftEye)
            rightEAR = self._eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Desenha os contornos dos olhos
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
                
            # Desenha infos de debug
            cv2.putText(frame, f"EAR: {ear:.2f}", (self.frame_width - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (self.frame_width - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # Se nenhuma face for detetada, reinicia os contadores
            self.drowsiness_counter = 0
            self.distraction_counter = 0

        # --- 3. Deteção de Celular (MobileNet-SSD) ---
        detections = self._detect_cell_phone(frame)
        alarm_cellphone = False
        
        if detections is not None:
            # Loop sobre todas as deteções
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # (MODO DEBUG) Verificamos se a confiança é maior que o nosso limite baixo
                if confidence > self.cellphone_min_conf:
                    idx = int(detections[0, 0, i, 1])
                    
                    # Pega o nome da classe e a cor
                    class_name = CLASSES_MOBILENET.get(idx, "desconhecido")
                    color = DEBUG_COLORS[idx]

                    # Calcula a caixa (bounding box)
                    box = detections[0, 0, i, 3:7] * np.array([self.frame_width, self.frame_height, self.frame_width, self.frame_height])
                    (startX, startY, endX, endY) = box.astype("int")

                    # (MODO DEBUG) Desenha a caixa e o rótulo de TUDO o que for detetado
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Agora, verificamos se este objeto é o nosso ALVO (tvmonitor)
                    if idx == self.target_class_id:
                        self.cellphone_counter += 1
                        if self.cellphone_counter >= CELLPHONE_CONSEC_FRAMES:
                            alarm_cellphone = True
                            logging.warning(f"DETEÇÃO DE {TARGET_CLASS_NAME} (Conf: {confidence:.2f})")
                    # (Não reinicia o contador se outro objeto for visto,
                    #  apenas se o telemóvel não for visto em nenhum loop)

            # Se o telemóvel não foi detetado NENHUMA vez neste frame
            if not alarm_cellphone:
                 # Esta lógica está um pouco simplista, mas por agora:
                 # Se não houve deteção do alvo, reinicia o contador
                 # (Uma lógica melhor seria verificar se alguma deteção foi o alvo)
                 
                 # Vamos simplificar: se o alvo não foi detetado, reinicia
                 found_target = any(int(detections[0, 0, i, 1]) == self.target_class_id and detections[0, 0, i, 2] > self.cellphone_min_conf for i in np.arange(0, detections.shape[2]))
                 if not found_target:
                     self.cellphone_counter = 0


        # --- 4. Desenha os Alertas Visuais Finais ---
        if alarm_drowsy:
             cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if alarm_distraction:
             cv2.putText(frame, "ALERTA: DISTRACAO!", (10, 60),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if alarm_cellphone:
             cv2.putText(frame, f"ALERTA: {TARGET_CLASS_NAME}!", (10, 90),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame, (alarm_drowsy, alarm_distraction, alarm_cellphone)

