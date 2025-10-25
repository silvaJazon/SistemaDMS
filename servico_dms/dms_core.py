# Documentação: Núcleo de Processamento do SistemaDMS
# Responsável por:
# 1. Carregar os modelos de IA (Dlib, YOLOv4-tiny)
# 2. Processar os frames de vídeo para detetar (Sonolência, Distração, Celular)

import cv2
import time
import logging
import numpy as np
import dlib
from scipy.spatial import distance as dist
import math

# --- Constantes de Caminho dos ModelOS ---
DLIB_MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'
YOLO_CONFIG_PATH = 'yolov4-tiny.cfg'
YOLO_WEIGHTS_PATH = 'yolov4-tiny.weights'
YOLO_NAMES_PATH = 'coco.names'

# --- Constantes de Deteção ---

# 1. Sonolência (EAR)
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15

# 2. Distração (Pose da Cabeça)
DISTRACTION_THRESHOLD_ANGLE = 30.0
DISTRACTION_CONSEC_FRAMES = 25

# 3. Celular (YOLOv4-tiny)
# (OTIMIZAÇÃO) Baixamos a confiança para 0.25 para ser mais sensível
CELLPHONE_MIN_CONFIDENCE = 0.25
CELLPHONE_NMS_THRESHOLD = 0.3
CELLPHONE_CONSEC_FRAMES = 10 # Se o FPS for 10, isto é 1 segundo
TARGET_CLASS_NAME = "cell phone"

# (OTIMIZAÇÃO AVANÇADA) Só executamos o YOLO a cada N frames
YOLO_FRAME_SKIP = 8 

# Índices Dlib para os olhos
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

class DriverMonitor:
    """
    Classe principal que encapsula toda a lógica de deteção de IA.
    """
    
    def __init__(self, frame_size):
        logging.info("A inicializar o DriverMonitor Core...")
        
        self.frame_height = frame_size[0]
        self.frame_width = frame_size[1]

        # Modelos Dlib
        self.dlib_detector = None
        self.dlib_predictor = None
        
        # Modelos YOLOv4-tiny
        self.yolo_net = None
        self.yolo_coco_classes = []
        self.yolo_output_layers = []
        self.target_class_id = -1

        # Contadores de Alerta
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        self.cellphone_counter = 0
        
        # (OTIMIZAÇÃO AVANÇADA) Contadores para decoupling
        self.frame_counter = 0
        self.cellphone_detected_in_last_check = False # "Memória" da deteção do telemóvel

        
        self.initialize_dlib_models()
        self.initialize_yolo_model()

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

    def initialize_yolo_model(self):
        """Carrega o modelo YOLOv4-tiny (Darknet)."""
        try:
            logging.info(f">>> Carregando nomes de classes COCO ({YOLO_NAMES_PATH})...")
            with open(YOLO_NAMES_PATH, 'r') as f:
                self.yolo_coco_classes = [line.strip() for line in f.readlines()]
            
            try:
                self.target_class_id = self.yolo_coco_classes.index(TARGET_CLASS_NAME)
                logging.info(f">>> Classe alvo '{TARGET_CLASS_NAME}' encontrada com ID: {self.target_class_id}")
            except ValueError:
                logging.error(f"!!! ERRO FATAL: Classe '{TARGET_CLASS_NAME}' não encontrada em {YOLO_NAMES_PATH}")
                raise

            logging.info(f">>> Carregando modelo YOLOv4-tiny (config e pesos)...")
            self.yolo_net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)
            
            self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            layer_names = self.yolo_net.getLayerNames()
            self.yolo_output_layers = [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers().flatten()]

            logging.info(">>> Modelo YOLOv4-tiny carregado com sucesso.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelo YOLOv4-tiny: {e}", exc_info=True)
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

    def _detect_cell_phone(self, frame):
        """Executa a deteção de objetos com YOLOv4-tiny."""
        
        boxes = []
        confidences = []
        class_ids = []
        found_cellphone = False

        try:
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            
            self.yolo_net.setInput(blob)
            layer_outputs = self.yolo_net.forward(self.yolo_output_layers)

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if class_id == self.target_class_id and confidence > CELLPHONE_MIN_CONFIDENCE:
                        box = detection[0:4] * np.array([self.frame_width, self.frame_height, self.frame_width, self.frame_height])
                        (centerX, centerY, width, height) = box.astype("int")
                        
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, CELLPHONE_MIN_CONFIDENCE, CELLPHONE_NMS_THRESHOLD)
            
            if len(indices) > 0:
                found_cellphone = True
                for i in indices.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    
                    color = (255, 255, 0) # Ciano
                    label = f"{TARGET_CLASS_NAME}: {confidences[i]:.2f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return found_cellphone

        except Exception as e:
            logging.warning(f"Falha ao executar a rede YOLOv4-tiny: {e}")
            return False


    def process_frame(self, frame, gray):
        """
        Função principal que processa um único frame.
        """
        
        # --- 1. Deteção de Rosto e Landmarks (Dlib) ---
        # (OTIMIZAÇÃO) Dlib corre em todos os frames
        rects = self.dlib_detector(gray, 0)
        
        alarm_drowsy = False
        alarm_distraction = False
        alarm_cellphone = False # Começa como Falso

        if len(rects) > 0:
            rect = rects[0]
            shape = self.dlib_predictor(gray, rect)
            shape_np = self._shape_to_np(shape)

            # (OTIMIZAÇÃO) Incrementa o contador de frames aqui
            self.frame_counter += 1

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

            # --- 3. Deteção de Celular (YOLOv4-tiny) ---
            # (OTIMIZAÇÃO AVANÇADA) Só executamos o YOLO (lento) a cada N frames
            if self.frame_counter % YOLO_FRAME_SKIP == 0:
                # Corre a deteção pesada e atualiza a "memória"
                self.cellphone_detected_in_last_check = self._detect_cell_phone(frame)
            
            # A lógica do alarme usa o resultado da última verificação ("memória")
            if self.cellphone_detected_in_last_check:
                self.cellphone_counter += 1
                if self.cellphone_counter >= CELLPHONE_CONSEC_FRAMES:
                    alarm_cellphone = True
                    logging.warning(f"ALARME DE {TARGET_CLASS_NAME.upper()}!")
            else:
                self.cellphone_counter = 0

        else:
            # (OTIMIZAÇÃO) Se nenhum rosto for detetado, reinicia TODOS os contadores
            self.drowsiness_counter = 0
            self.distraction_counter = 0
            self.cellphone_counter = 0
            self.frame_counter = 0 # Reinicia o contador de frames
            self.cellphone_detected_in_last_check = False # Limpa a "memória"


        # --- 4. Desenha os Alertas Visuais Finais ---
        if alarm_drowsy:
             cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if alarm_distraction:
             cv2.putText(frame, "ALERTA: DISTRACAO!", (10, 60),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if alarm_cellphone:
             cv2.putText(frame, f"ALERTA: {TARGET_CLASS_NAME.upper()}!", (10, 90),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame, (alarm_drowsy, alarm_distraction, alarm_cellphone)

