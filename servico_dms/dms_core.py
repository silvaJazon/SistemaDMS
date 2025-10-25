# Documentação: Núcleo de Processamento do SistemaDMS (Fase 3)
# Contém a classe 'DriverMonitor' que encapsula toda a lógica de deteção:
# 1. Dlib: EAR (sonolência) e Pose da Cabeça (distração).
# 2. YOLOv3-tiny: Deteção de telemóvel.

import cv2
import dlib
import numpy as np
import logging
import sys
import os
from scipy.spatial import distance as dist
import math

# --- Configurações de Alerta (Dlib) ---
EAR_THRESHOLD = 0.25 # Limite do Eye Aspect Ratio para considerar "fechado"
EAR_CONSEC_FRAMES = 15 # Número de frames consecutivos com olhos fechados para disparar o alarme

DISTRACTION_THRESHOLD_ANGLE = 30.0 # Ângulo (em graus) para considerar "distraído"
DISTRACTION_CONSEC_FRAMES = 25 # Número de frames consecutivos distraído para disparar o alarme

# --- NOVO: Configurações de Alerta (YOLO) ---
CELLPHONE_CONFID_THRESHOLD = 0.5 # Confiança mínima para detetar um telemóvel
CELLPHONE_CONSEC_FRAMES = 10 # Frames consecutivos com telemóvel para disparar alarme

# --- Caminhos dos Modelos ---
MODEL_PATH = 'shape_predictor_68_face_landmarks.dat' # Modelo Dlib

# --- NOVO: Caminhos dos Modelos (YOLO) ---
YOLO_CONFIG_PATH = 'yolov3-tiny.cfg'
YOLO_WEIGHTS_PATH = 'yolov3-tiny.weights'
COCO_NAMES_PATH = 'coco.names'

# Índices dos landmarks do Dlib
EYE_AR_LEFT_START = 42
EYE_AR_LEFT_END = 48
EYE_AR_RIGHT_START = 36
EYE_AR_RIGHT_END = 42

class DriverMonitor:
    """
    Classe principal para o monitoramento do motorista.
    Engloba a inicialização dos modelos e o processamento de cada frame.
    """
    
    def __init__(self):
        logging.info("A inicializar o DriverMonitor Core...")
        
        # --- Dlib ---
        self.lStart = EYE_AR_LEFT_START
        self.lEnd = EYE_AR_LEFT_END
        self.rStart = EYE_AR_RIGHT_START
        self.rEnd = EYE_AR_RIGHT_END
        self.drowsiness_counter = 0
        self.distraction_counter = 0
        self.detector = None
        self.predictor = None
        
        # --- NOVO: YOLO ---
        self.cellphone_counter = 0
        self.yolo_net = None
        self.coco_classes = None
        self.output_layers = None
        self.yolo_input_size = (416, 416) # YOLOv3-tiny usa 416x416 ou 320x320

        # --- Inicializar Modelos ---
        self.initialize_dlib_model()
        self.initialize_yolo_model() # NOVO

    def initialize_dlib_model(self):
        """Carrega o detetor de face e o preditor de landmarks do Dlib."""
        try:
            logging.info(">>> Carregando detetor de faces do Dlib...")
            self.detector = dlib.get_frontal_face_detector()
            
            logging.info(f">>> Carregando preditor de landmarks faciais ({MODEL_PATH})...")
            if not os.path.exists(MODEL_PATH):
                logging.error(f"!!! ERRO FATAL: Modelo Dlib '{MODEL_PATH}' não encontrado.")
                raise FileNotFoundError
                
            self.predictor = dlib.shape_predictor(MODEL_PATH)
            logging.info(">>> Modelos Dlib carregados com sucesso.")
            
        except FileNotFoundError:
            logging.error("O ficheiro do modelo Dlib não foi encontrado. O Dockerfile descarregou-o?")
            sys.exit(1)
        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelos Dlib: {e}", exc_info=True)
            sys.exit(1)

    # NOVO: Função para carregar o modelo YOLO
    def initialize_yolo_model(self):
        """Carrega o modelo YOLOv3-tiny e os nomes das classes COCO."""
        try:
            logging.info(">>> Carregando nomes de classes COCO...")
            if not os.path.exists(COCO_NAMES_PATH):
                 logging.error(f"!!! ERRO FATAL: Ficheiro COCO '{COCO_NAMES_PATH}' não encontrado.")
                 raise FileNotFoundError
                 
            with open(COCO_NAMES_PATH, 'r') as f:
                self.coco_classes = [line.strip() for line in f.readlines()]
            
            logging.info(">>> Carregando modelo YOLOv3-tiny (config e pesos)...")
            if not os.path.exists(YOLO_CONFIG_PATH) or not os.path.exists(YOLO_WEIGHTS_PATH):
                logging.error(f"!!! ERRO FATAL: Ficheiros YOLO (.cfg ou .weights) não encontrados.")
                raise FileNotFoundError

            self.yolo_net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)
            
            # Define o backend para OpenCV (otimizado para CPU)
            self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Obtém os nomes das camadas de saída
            layer_names = self.yolo_net.getLayerNames()
            # Correção para diferentes versões do OpenCV
            try:
                self.output_layers = [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
            except TypeError:
                 self.output_layers = [layer_names[i[0] - 1] for i in self.yolo_net.getUnconnectedOutLayers()]

            logging.info(">>> Modelo YOLO carregado com sucesso.")

        except Exception as e:
            logging.error(f"!!! ERRO FATAL ao carregar modelo YOLO: {e}", exc_info=True)
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
        model_points = np.array([
                                (0.0, 0.0, 0.0),             # Ponta do nariz (30)
                                (0.0, -330.0, -65.0),        # Queixo (8)
                                (-225.0, 170.0, -135.0),     # Canto do olho esquerdo (36)
                                (225.0, 170.0, -135.0),      # Canto do olho direito (45)
                                (-150.0, -150.0, -125.0),    # Canto da boca esquerdo (48)
                                (150.0, -150.0, -125.0)     # Canto da boca direito (54)
                            ])
        focal_length = frame_size[1]
        center = (frame_size[1]/2, frame_size[0]/2)
        camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = "double"
                                )
        dist_coeffs = np.zeros((4,1))
        image_points = np.array([
                                shape_np[30], shape_np[8], shape_np[36],
                                shape_np[45], shape_np[48], shape_np[54]
                            ], dtype="double")
        
        try:
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            yaw, pitch, roll = angles[1], angles[0], angles[2]
            return yaw, pitch, roll
        except Exception as e:
            logging.debug(f"Erro ao calcular pose da cabeça: {e}")
            return 0, 0, 0

    # NOVO: Função para detetar telemóveis com YOLO
    def _detect_cell_phone(self, frame, frame_width, frame_height):
        """Executa a deteção de objetos (YOLO) no frame para encontrar telemóveis."""
        
        # Cria um "blob" (imagem de entrada) para o YOLO
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, self.yolo_input_size, swapRB=True, crop=False)
        self.yolo_net.setInput(blob)
        
        # Executa a passagem forward (deteção)
        try:
            layerOutputs = self.yolo_net.forward(self.output_layers)
        except Exception as e:
            logging.warning(f"Erro ao executar forward pass do YOLO: {e}")
            return False, None

        boxes = []
        confidences = []
        classIDs = []

        # Itera sobre todas as saídas da rede
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Filtra pela classe "cell phone" e pela confiança
                if self.coco_classes[classID] == "cell phone" and confidence > CELLPHONE_CONFID_THRESHOLD:
                    # Escala a bounding box de volta para o tamanho original do frame
                    box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    # Usa o centro (x, y) para derivar o canto superior esquerdo
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Aplica "Non-maxima suppression" para remover caixas sobrepostas
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CELLPHONE_CONFID_THRESHOLD, 0.3)

        if len(idxs) > 0:
            # Assume que a deteção mais forte é o telemóvel
            best_idx = idxs[0][0] if isinstance(idxs[0], list) else idxs[0]
            best_box = boxes[best_idx]
            return True, best_box # Encontrado
            
        return False, None # Não encontrado

    def process_frame(self, frame_display, gray):
        """
        Processa um único frame para deteção de sonolência, distração e telemóvel.
        Retorna o frame com as anotações e um dicionário de status.
        """
        
        (frame_height, frame_width) = frame_display.shape[:2]
        
        # --- Deteção de Rosto (Dlib) ---
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
            yaw, pitch, roll = self._estimate_head_pose(shape_np, (frame_height, frame_width))
            status_data['yaw'] = yaw
            status_data['pitch'] = pitch
            
            if abs(yaw) > DISTRACTION_THRESHOLD_ANGLE or pitch > DISTRACTION_THRESHOLD_ANGLE:
                self.distraction_counter += 1
                if self.distraction_counter >= DISTRACTION_CONSEC_FRAMES:
                    alarm_distraction = True
                    logging.warning(f"DETEÇÃO DE DISTRAÇÃO (Yaw: {yaw:.1f}, Pitch: {pitch:.1f})")
            else:
                self.distraction_counter = 0
                
            cv2.putText(frame_display, f"EAR: {ear:.2f}", (frame_width - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Yaw: {yaw:.1f}", (frame_width - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not rects:
            self.drowsiness_counter = 0
            self.distraction_counter = 0

        # --- NOVO: 3. Verificação de Uso de Telemóvel (YOLO) ---
        alarm_cellphone = False
        # Executa a deteção de telemóvel em todos os frames
        found_cellphone, box = self._detect_cell_phone(frame_display, frame_width, frame_height)

        if found_cellphone:
            self.cellphone_counter += 1
            
            # Desenha a caixa de deteção do telemóvel
            (x, y, w, h) = box
            cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 255), 2) # Ciano
            cv2.putText(frame_display, "Telemovel", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if self.cellphone_counter >= CELLPHONE_CONSEC_FRAMES:
                alarm_cellphone = True
                logging.warning("DETEÇÃO DE USO DE CELULAR")
        else:
            self.cellphone_counter = 0
        

        # --- Desenha os Alertas Visuais Finais ---
        if alarm_drowsy:
            cv2.putText(frame_display, "ALERTA: SONOLENCIA!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if alarm_distraction:
            cv2.putText(frame_display, "ALERTA: DISTRACAO!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # NOVO: Adiciona o alerta de telemóvel
        if alarm_cellphone:
            # Coloca este alerta abaixo dos outros
            y_pos = 90 if alarm_drowsy or alarm_distraction else 30
            if alarm_drowsy and alarm_distraction: y_pos = 90
            
            cv2.putText(frame_display, "ALERTA: USO DE CELULAR!", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


        # --- Retorno ---
        status_data['alarm_drowsy'] = alarm_drowsy
        status_data['alarm_distraction'] = alarm_distraction
        status_data['alarm_cellphone'] = alarm_cellphone # NOVO
        
        return frame_display, status_data

