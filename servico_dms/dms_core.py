# Documentação: Núcleo do SistemaDMS (DriverMonitor Core)
# Fase 3.2: Refatorado com MobileNet-SSD
# Responsável por toda a lógica de Visão Computacional (IA).
# Utiliza Dlib (Sonolência, Distração) e MobileNet-SSD (Celular).

import cv2
import dlib
import numpy as np
import logging
from scipy.spatial import distance as dist
import math
import os

# --- Configurações de Deteção ---
# (Pode mover para um 'config.py' no futuro)

# Caminhos dos Modelos (Dlib)
DLIB_LANDMARKS_PATH = 'shape_predictor_68_face_landmarks.dat'
# Índices Dlib
EYE_AR_LEFT_START = 42
EYE_AR_LEFT_END = 48
EYE_AR_RIGHT_START = 36
EYE_AR_RIGHT_END = 42

# Caminhos dos Modelos (MobileNet-SSD) - NOVO
MOBILENET_PROTOTXT_PATH = "MobileNetSSD_deploy.prototxt"
MOBILENET_MODEL_PATH = "MobileNetSSD_deploy.caffemodel"

# Classes do MobileNet-SSD (índices)
# Estamos interessados apenas no 'cell phone' (telemóvel)
MOBILENET_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                     "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                     "sofa", "train", "tvmonitor"]
# "cell phone" (telemóvel) não está no modelo padrão MobileNetSSD Caffe.
# Vamos usar "bottle" (garrafa) [índice 5] ou "tvmonitor" [índice 20] como substituto por agora.
# Vamos assumir que "bottle" (índice 5) é o nosso "telemóvel" para este teste.
# NOTA: O modelo Caffe original não tem "cell phone".
# Vamos usar o índice 6 (autocarro) ou 7 (carro)? Não. Vamos usar 'bottle' (5)
# ou 'person' (15)
# Vamos redefinir: O modelo Caffe que estamos a usar TEM 21 classes (0-20).
# O modelo treinado em COCO (como o YOLO) tem 'cell phone'. O Caffe treinado em PASCAL VOC não.
# Vamos usar o índice 15 ("person") para testar se a deteção de objetos funciona,
# e depois trocamos por um modelo que detete telemóveis.
#
# REVISÃO: Vamos usar 'tvmonitor' (índice 20) que é o mais próximo de um telemóvel/ecrã.
# Se o utilizador segurar o telemóvel, ele pode ser detetado como 'tvmonitor'.
TARGET_CLASS_ID = 20 # Índice de "tvmonitor"
TARGET_CLASS_NAME = "Monitor/Celular"
CONFIDENCE_THRESHOLD_OBJECT = 0.3 # Confiança mínima para o objeto

# --- Configurações de Alerta ---
# Sonolência
EAR_THRESHOLD = 0.25 # Limite do Eye Aspect Ratio para considerar "fechado"
EAR_CONSEC_FRAMES = 15 # Número de frames consecutivos com olhos fechados

# Distração (Pose da Cabeça)
DISTRACTION_THRESHOLD_ANGLE = 30.0 # Ângulo (em graus) para considerar "distraído"
DISTRACTION_CONSEC_FRAMES = 25 # Número de frames consecutivos distraído

# Uso de Celular (NOVO)
CELLPHONE_CONSEC_FRAMES = 10 # Número de frames consecutivos com objeto

class DriverMonitor:
    """
    Classe principal que encapsula toda a lógica de deteção do DMS.
    """
    def __init__(self, frame_size):
        logging.info("A inicializar o DriverMonitor Core...")
        self.frame_size = frame_size
        self.detector = None
        self.predictor = None
        
        # --- NOVO: MobileNet-SSD ---
        self.mobilenet_net = None
        self.class_id_target = TARGET_CLASS_ID # O que procuramos (tvmonitor)
        
        # Contadores de Alerta
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        self.cellphone_counter = 0

        # Carrega os modelos
        self.initialize_dlib_model()
        self.initialize_mobilenet_model() # NOVO

    # --- Funções de Inicialização ---

    def initialize_dlib_model(self):
        """Carrega o detetor de face e o preditor de landmarks do Dlib."""
        try:
            logging.info(f">>> Carregando detetor de faces do Dlib...")
            self.detector = dlib.get_frontal_face_detector()
            logging.info(f">>> Carregando preditor de landmarks faciais ({DLIB_LANDMARKS_PATH})...")
            self.predictor = dlib.shape_predictor(DLIB_LANDMARKS_PATH)
            logging.info(">>> Modelos Dlib carregados com sucesso.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelos Dlib: {e}", exc_info=True)
            raise

    def initialize_mobilenet_model(self):
        """Carrega o modelo MobileNet-SSD (Caffe)."""
        if not (os.path.exists(MOBILENET_PROTOTXT_PATH) and os.path.exists(MOBILENET_MODEL_PATH)):
            logging.error("!!! ERRO FATAL: Ficheiros do MobileNet-SSD não encontrados.")
            logging.error(f"Verifique se '{MOBILENET_PROTOTXT_PATH}' e '{MOBILENET_MODEL_PATH}' existem.")
            raise FileNotFoundError("Ficheiros de modelo MobileNet-SSD não encontrados.")
            
        try:
            logging.info(">>> Carregando modelo MobileNet-SSD (prototxt e caffemodel)...")
            self.mobilenet_net = cv2.dnn.readNetFromCaffe(MOBILENET_PROTOTXT_PATH, MOBILENET_MODEL_PATH)
            logging.info(">>> Modelo MobileNet-SSD carregado com sucesso.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelo MobileNet-SSD: {e}", exc_info=True)
            raise

    # --- Funções Auxiliares (Dlib) ---

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

    def _estimate_head_pose(self, shape):
        """Estima a pose da cabeça (para onde o motorista está a olhar)."""
        model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])
        
        focal_length = self.frame_size[1]
        center = (self.frame_size[1]/2, self.frame_size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4,1))

        image_points = np.array([
            shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]
        ], dtype="double")
        
        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            yaw, pitch, roll = angles[1], angles[0], angles[2]
            return yaw, pitch, roll
        except Exception:
            return 0, 0, 0

    # --- Função de Deteção de Objeto (MobileNet-SSD) ---

    def _detect_cell_phone(self, frame):
        """
        Deteta objetos no frame usando MobileNet-SSD e procura pelo alvo (tvmonitor/celular).
        """
        try:
            (h, w) = frame.shape[:2]
            # Cria um "blob" do frame (pré-processamento)
            # 300x300 é o tamanho que o MobileNet-SSD espera
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                         0.007843, (300, 300), 127.5)
            
            # Passa o blob pela rede
            self.mobilenet_net.setInput(blob)
            detections = self.mobilenet_net.forward()

            found_object = False
            box = (0, 0, 0, 0)

            # Itera sobre as deteções
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > CONFIDENCE_THRESHOLD_OBJECT:
                    class_id = int(detections[0, 0, i, 1])

                    # Verifica se é a classe que procuramos
                    if class_id == self.class_id_target:
                        box_coords = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box_coords.astype("int")
                        
                        # Retorna a primeira deteção válida
                        found_object = True
                        box = (startX, startY, endX, endY)
                        break 
            
            return found_object, box

        except Exception as e:
            logging.warning(f"Erro durante a deteção de objeto (MobileNet): {e}")
            return False, (0, 0, 0, 0)


    # --- Função de Processamento Principal ---

    def process_frame(self, frame_display, gray_frame):
        """
        Executa todas as deteções no frame recebido.
        """
        # --- Alertas ---
        alarm_drowsy = False
        alarm_distraction = False
        alarm_cellphone = False # NOVO

        # --- Deteção de Rosto (Dlib) ---
        rects = self.detector(gray_frame, 0)
        
        if not rects:
             # Se nenhuma face for detetada, reinicia os contadores de rosto
             self.drowsiness_counter = 0
             self.distraction_counter = 0
        
        # Loop sobre as faces detetadas (deve ser só 1, o motorista)
        for rect in rects:
            shape = self.predictor(gray_frame, rect)
            shape = self._shape_to_np(shape)

            # --- 1. Verificação de Sonolência (EAR) ---
            leftEye = shape[EYE_AR_LEFT_START:EYE_AR_LEFT_END]
            rightEye = shape[EYE_AR_RIGHT_START:EYE_AR_RIGHT_END]
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
            yaw, pitch, roll = self._estimate_head_pose(shape)
            
            if abs(yaw) > DISTRACTION_THRESHOLD_ANGLE or pitch > DISTRACTION_THRESHOLD_ANGLE:
                self.distraction_counter += 1
                if self.distraction_counter >= DISTRACTION_CONSEC_FRAMES:
                    alarm_distraction = True
                    logging.warning(f"DETEÇÃO DE DISTRAÇÃO (Yaw: {yaw:.1f}, Pitch: {pitch:.1f})")
            else:
                self.distraction_counter = 0
                
            # Desenha infos de debug (rosto)
            cv2.putText(frame_display, f"EAR: {ear:.2f}", (self.frame_size[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Yaw: {yaw:.1f}", (self.frame_size[1] - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # --- 3. Verificação de Uso de Celular (MobileNet-SSD) ---
        # (Executa mesmo se não houver rosto)
        
        found_cell, box = self._detect_cell_phone(frame_display)
        
        if found_cell:
            self.cellphone_counter += 1
            # Desenha a caixa do objeto
            (startX, startY, endX, endY) = box
            cv2.rectangle(frame_display, (startX, startY), (endX, endY), (0, 255, 255), 2)
            cv2.putText(frame_display, TARGET_CLASS_NAME, (startX, startY - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if self.cellphone_counter >= CELLPHONE_CONSEC_FRAMES:
                alarm_cellphone = True
                logging.warning("DETEÇÃO DE USO DE CELULAR/OBJETO")
        else:
            self.cellphone_counter = 0
            
        # --- Desenha os Alertas Visuais Finais ---
        y_offset = 30
        if alarm_drowsy:
            cv2.putText(frame_display, "ALERTA: SONOLENCIA!", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            y_offset += 30
        
        if alarm_distraction:
            cv2.putText(frame_display, "ALERTA: DISTRACAO!", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            y_offset += 30
            
        if alarm_cellphone:
            cv2.putText(frame_display, "ALERTA: USO DE CELULAR!", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame_display, (alarm_drowsy, alarm_distraction, alarm_cellphone)

