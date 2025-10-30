# Documentação: Núcleo do SistemaDMS (Implementação MediaPipe + YOLOv8)
# (VERSÃO: Híbrido Otimizado - Deteta Mão (MP), depois Celular (YOLO) no recorte)

import cv2
import mediapipe as mp
import numpy as np
import logging
import math
import threading
from scipy.spatial import distance as dist
from datetime import datetime
import time
from collections import deque

from ultralytics import YOLO
from dms_base import BaseMonitor
from camera_thread import CameraThread

cv2.setUseOptimized(True)

# --- Índices MediaPipe (permanecem iguais) ---
MP_LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE_IDX = [362, 385, 387, 263, 380, 373]
MP_MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 87]

# (REMOVIDO) YOLO_IMG_SIZE/SCALE_X/Y - Vamos usar coordenadas dinâmicas


class MediaPipeMonitor(BaseMonitor):
    """
    Implementação Multithread:
    - Thread 1 (Principal): MediaPipe Face Mesh (EAR/MAR)
    - Thread 2 (Fundo): Híbrido Otimizado:
        1. MediaPipe Hands (Rápido)
        2. Se Mão encontrada -> YOLOv8 no recorte da Mão (Super Rápido)
    """

    def __init__(self, frame_size, stop_event: threading.Event, default_settings: dict = None):
        super().__init__(frame_size, stop_event, default_settings)
        logging.info("A inicializar o MediaPipeMonitor Core (Modo: Híbrido Otimizado MP-Mão + YOLO-Recorte)...")

        # --- 1. Inicializa o MediaPipe FaceMesh (Thread Principal) ---
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            logging.info(">>> Modelos MediaPipe FaceMesh carregados.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL MediaPipe (FaceMesh): {e}", exc_info=True)
            raise RuntimeError(f"Erro MediaPipe (FaceMesh): {e}")

        # --- 2. Inicializa o MediaPipe Hands (Thread Fundo) ---
        try:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5
            )
            logging.info(">>> Modelos MediaPipe Hands carregados.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL MediaPipe (Hands): {e}", exc_info=True)
            raise RuntimeError(f"Erro MediaPipe (Hands): {e}")

        # --- 3. Carregar Modelo YOLOv8 (Thread Fundo) ---
        try:
            model_file = 'yolov8s.pt' 
            logging.info(f">>> Carregando modelo YOLOv8 ('{model_file}')...")
            self.yolo_model = YOLO(model_file)
            logging.info(f">>> Modelo {model_file} carregado.")
            
            self.yolo_cellphone_class_id = -1
            if self.yolo_model.names:
                for class_id, name in self.yolo_model.names.items():
                    if name == 'cell phone':
                        self.yolo_cellphone_class_id = class_id
                        logging.info(f"Classe 'cell phone' encontrada no YOLO. ID: {class_id}")
                        break
            if self.yolo_cellphone_class_id == -1:
                 logging.warning("!!! Classe 'cell phone' não encontrada nos nomes do modelo YOLO.")
            
            logging.info(">>> Executando 'warm-up' (primeira inferência)...")
            try:
                # (ALTERADO) Warm-up agora só precisa do 'hands'
                dummy_frame_rgb = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                self.hands.process(dummy_frame_rgb)
                # O YOLO será "aquecido" na primeira vez que uma mão for detetada
                logging.info(">>> Warm-up (Hands) concluído.")
            except Exception as e:
                logging.warning(f"Falha no warm-up: {e}")
                 
        except Exception as e:
            logging.error(f"!!! ERRO FATAL YOLO: {e}", exc_info=True)
            raise RuntimeError(f"Erro YOLO: {e}")

        # --- 4. Contadores e Configurações ---
        self.lock = threading.Lock() 
        self.drowsiness_counter = 0
        self.yawn_counter = 0
        self.phone_counter = 0
        
        self.drowsy_alert_active = False
        self.yawn_alert_active = False
        self.phone_alert_active = False

        self.ear_threshold = self.default_settings.get('ear_threshold', 0.25)
        self.ear_frames = self.default_settings.get('ear_frames', 7)
        self.mar_threshold = self.default_settings.get('mar_threshold', 0.40)
        self.mar_frames = self.default_settings.get('mar_frames', 10)
        
        self.phone_detection_enabled = True
        self.phone_confidence = 0.20
        self.phone_frames = 5      

        # --- 5. Configuração do Thread YOLO ---
        self.cam_thread_ref: CameraThread = None
        self.phone_thread = None
        self.yolo_lock = threading.Lock() 
        self.last_yolo_boxes = [] # (ALTERADO) Agora guarda a BBox da MÃO
        self.phone_found_by_thread = False


    # --- Loop do Thread YOLO ---
    def _yolo_loop(self):
        """
        Loop (Thread Fundo) - Lógica Híbrida Otimizada:
        1. Procura Mão (MediaPipe Hands - Rápido)
        2. SE encontrar mão -> Recorta (Crop) a imagem
        3. Executa YOLO (Celular) SÓ no recorte (Super Rápido)
        """
        logging.info(">>> _yolo_loop (Thread Híbrido) iniciado.")
        
        if self.stop_event.wait(timeout=3.0): return

        while not self.stop_event.is_set():
            start_time_yolo = time.time()
            
            with self.lock:
                phone_enabled = self.phone_detection_enabled
                current_phone_confidence = self.phone_confidence
            
            if not phone_enabled or self.cam_thread_ref is None or self.yolo_cellphone_class_id == -1:
                with self.yolo_lock:
                    self.last_yolo_boxes = []
                    self.phone_found_by_thread = False
                logging.info("_yolo_loop: Deteção de celular DESATIVADA. A aguardar...")
                if self.stop_event.wait(timeout=2.0): break
                continue

            try:
                frame = self.cam_thread_ref.get_frame()
                if frame is None:
                    logging.warning("_yolo_loop: Não obteve frame. A tentar novamente em 2s.")
                    if self.stop_event.wait(timeout=2.0): break
                    continue
                
                logging.debug("_yolo_loop: A executar inferência (1. Mãos)...")
                
                # Converte para RGB (necessário para MediaPipe Hands)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # --- 1. Inferência MediaPipe Hands ---
                results_hands = self.hands.process(frame_rgb)
                
                phone_found_final = False
                current_boxes = [] # Caixa da MÃO

                if results_hands.multi_hand_landmarks:
                    logging.info("_yolo_loop: Mão(MP) encontrada. Verificando se há um celular (YOLO)...")
                    
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        # --- 2. Calcular a Bounding Box da Mão ---
                        h, w, _ = frame.shape
                        x_min, y_min = w, h
                        x_max, y_max = 0, 0
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            if x < x_min: x_min = x
                            if x > x_max: x_max = x
                            if y < y_min: y_min = y
                            if y > y_max: y_max = y
                        
                        # Adicionar uma "margem" (padding) de segurança
                        padding = 30
                        x_min = max(0, x_min - padding)
                        y_min = max(0, y_min - padding)
                        x_max = min(w, x_max + padding)
                        y_max = min(h, y_max + padding)

                        if x_min >= x_max or y_min >= y_max:
                            continue

                        # --- 3. Recortar (Crop) a imagem original ---
                        hand_crop = frame[y_min:y_max, x_min:x_max]

                        # --- 4. Executar YOLO *apenas* no recorte ---
                        if hand_crop.size == 0:
                            logging.warning("_yolo_loop: Recorte da mão resultou em imagem vazia.")
                            continue
                            
                        results_yolo = self.yolo_model(
                            hand_crop, verbose=False, classes=[self.yolo_cellphone_class_id],
                            conf=current_phone_confidence, imgsz=160, # Tamanho pequeno, pois a imagem já é pequena
                            augment=False, half=False
                        )

                        if results_yolo and results_yolo[0].boxes:
                            for box in results_yolo[0].boxes:
                                if int(box.cls) == self.yolo_cellphone_class_id:
                                    phone_found_final = True
                                    # A "caixa" que desenhamos é a da MÃO
                                    current_boxes.append([x_min, y_min, x_max, y_max]) 
                                    break # Encontrou telemóvel nesta mão
                    
                        if phone_found_final:
                             break # Sai do loop 'for hand_landmarks'
                
                # --- 5. Atualizar os resultados (thread-safe) ---
                with self.yolo_lock:
                    self.phone_found_by_thread = phone_found_final
                    self.last_yolo_boxes = current_boxes

                logging.info(f"_yolo_loop: Inferência concluída. Mão/Celular Híbrido: {phone_found_final}. Duração: {time.time() - start_time_yolo:.3f}s")

            except Exception as e:
                logging.error(f"_yolo_loop: Erro na inferência: {e}", exc_info=True)
                with self.yolo_lock:
                    self.phone_found_by_thread = False
                    self.last_yolo_boxes = []

            # (ALTERADO) Descansa 2 segundos. O processo agora é muito mais rápido (MP Hands + YOLO-crop)
            sleep_time = 2.0
            if self.stop_event.wait(timeout=sleep_time):
                break
        
        logging.info(">>> _yolo_loop (Thread) terminado.")

    def start_yolo_thread(self, cam_thread_ref: CameraThread):
        self.cam_thread_ref = cam_thread_ref
        self.phone_thread = threading.Thread(target=self._yolo_loop, name="PhoneDetectionThread")
        self.phone_thread.daemon = True
        self.phone_thread.start()

    # --- Funções de Cálculo (permanecem iguais) ---
    def _eye_aspect_ratio(self, eye_landmarks):
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5]); B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        return 0.3 if C < 1e-6 else (A + B) / (2.0 * C)
    def _mouth_aspect_ratio(self, mouth_landmarks):
        A = dist.euclidean(mouth_landmarks[1], mouth_landmarks[7]); B = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])
        C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])
        return 0.0 if C < 1e-6 else (A + B) / (2.0 * C)
    def _get_landmarks_from_result(self, landmarks, indices):
        coords = np.zeros((len(indices), 2), dtype="int")
        for i, idx in enumerate(indices):
            lm = landmarks[idx]
            coords[i] = (int(lm.x * self.frame_width), int(lm.y * self.frame_height))
        return coords


    def process_frame(self, frame, gray):
        logging.debug("DMSCore(MediaPipe): process_frame (MP Rápido) iniciado.")
        start_time_total = time.time()
        events_list = []
        status_data = {"ear": "-", "mar": "-", "yaw": "-", "pitch": "-", "roll": "-"}
        face_found_this_frame = False
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_mp = self.face_mesh.process(frame_rgb)
        except Exception as e:
            logging.error(f"DMSCore(MediaPipe): Erro .process(): {e}", exc_info=True)
            return frame, events_list, status_data

        if results_mp.multi_face_landmarks:
            face_landmarks = results_mp.multi_face_landmarks[0].landmark
            face_found_this_frame = True

            try:
                left_eye_pts = self._get_landmarks_from_result(face_landmarks, MP_LEFT_EYE_IDX)
                right_eye_pts = self._get_landmarks_from_result(face_landmarks, MP_RIGHT_EYE_IDX)
                mouth_pts = self._get_landmarks_from_result(face_landmarks, MP_MOUTH_IDX)
                ear_left = self._eye_aspect_ratio(left_eye_pts); ear_right = self._eye_aspect_ratio(right_eye_pts)
                ear = (ear_left + ear_right) / 2.0;
                mar = self._mouth_aspect_ratio(mouth_pts);
                status_data["ear"] = f"{ear:.2f}"; status_data["mar"] = f"{mar:.2f}"

                cv2.drawContours(frame, [cv2.convexHull(left_eye_pts)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye_pts)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(mouth_pts)], -1, (0, 255, 255), 1)

            except Exception as e:
                logging.error(f"DMSCore(MediaPipe): Erro ao processar landmarks: {e}", exc_info=True)
                face_found_this_frame = False
        
        local_boxes = []
        phone_found = False
        
        with self.lock:
            phone_enabled_locked = self.phone_detection_enabled
        
        if phone_enabled_locked:
            with self.yolo_lock:
                local_boxes = self.last_yolo_boxes # (ALTERADO) Isto agora é a BBox da mão
                phone_found = self.phone_found_by_thread
            
            # (ALTERADO) Desenha a caixa da MÃO (apenas se o telemóvel foi encontrado nela)
            for box_coords in local_boxes:
                x1, y1, x2, y2 = map(int, box_coords) 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, "Celular (na Mao)", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        
        logging.debug("DMSCore(MediaPipe): Lock alerta...")
        with self.lock:
            logging.debug("DMSCore(MediaPipe): Lock alerta OK.")

            if face_found_this_frame:
                # Sonolência
                if ear < self.ear_threshold:
                    self.drowsiness_counter += 1
                    logging.debug(f"DMSCore(MediaPipe): EAR baixo ({ear:.3f}<{self.ear_threshold}), cont={self.drowsiness_counter}/{self.ear_frames}")
                    if self.drowsiness_counter >= self.ear_frames and not self.drowsy_alert_active:
                        self.drowsy_alert_active = True
                        events_list.append({"type": "SONOLENCIA", "value": f"EAR: {ear:.2f}", "timestamp": datetime.now().isoformat() + "Z"})
                        logging.warning("DMSCore(MediaPipe): EVENTO SONOLENCIA.")
                else:
                    if self.drowsiness_counter > 0: logging.debug("DMSCore(MediaPipe): Sonolência reset.")
                    self.drowsiness_counter = 0; self.drowsy_alert_active = False

                # Bocejo
                if mar > self.mar_threshold:
                    self.yawn_counter += 1
                    logging.debug(f"DMSCore(MediaPipe): MAR alto ({mar:.3f}>{self.mar_threshold}), cont={self.yawn_counter}/{self.mar_frames}")
                    if self.yawn_counter >= self.mar_frames and not self.yawn_alert_active:
                        self.yawn_alert_active = True
                        events_list.append({"type": "BOCEJO", "value": f"MAR: {mar:.2f}", "timestamp": datetime.now().isoformat() + "Z"})
                        logging.warning("DMSCore(MediaPipe): EVENTO BOCEJO.")
                else:
                    if self.yawn_counter > 0: logging.debug("DMSCore(MediaPipe): Bocejo reset.")
                    self.yawn_counter = 0; self.yawn_alert_active = False
            
            else:
                logging.debug("DMSCore(MediaPipe): Nenhuma face encontrada.")
                self.drowsiness_counter = 0; self.drowsy_alert_active = False
                self.yawn_counter = 0; self.yawn_alert_active = False

            if phone_enabled_locked:
                if phone_found:
                    self.phone_counter += 1 
                    logging.debug(f"DMSCore(YOLO/Mao): Celular/Mão encontrado (lido do thread), cont={self.phone_counter}/{self.phone_frames}")
                    if self.phone_counter >= self.phone_frames and not self.phone_alert_active:
                        self.phone_alert_active = True
                        events_list.append({"type": "DISTRACAO", "value": "Celular na mao", "timestamp": datetime.now().isoformat() + "Z"})
                        logging.warning("DMSCore(YOLO/Mao): EVENTO DISTRACAO (CELULAR NA MAO).")
                else:
                    if self.phone_counter > 0: logging.debug("DMSCore(YOLO/Mao): Deteção celular/mão reset.")
                    self.phone_counter = 0; self.phone_alert_active = False

        logging.debug("DMSCore(MediaPipe): Lock alerta libertado.")

        if self.drowsy_alert_active:
            cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if self.yawn_alert_active:
            cv2.putText(frame, "ALERTA: BOCEJO!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if phone_enabled_locked and self.phone_alert_active:
             cv2.putText(frame, "ALERTA: CELULAR/MAO!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        total_time = time.time() - start_time_total
        logging.debug(f"DMSCore(MediaPipe): process_frame (MP Rápido) {total_time:.4f}s.")
        return frame, events_list, status_data

    def update_settings(self, settings):
        logging.debug(f"DMSCore(MediaPipe): Tentando atualizar conf: {settings}")
        
        with self.lock:
            try:
                self.ear_threshold = float(settings.get('ear_threshold', self.ear_threshold))
                self.ear_frames = int(settings.get('ear_frames', self.ear_frames))
                self.mar_threshold = float(settings.get('mar_threshold', self.mar_threshold))
                self.mar_frames = int(settings.get('mar_frames', self.mar_frames))
                
                self.phone_detection_enabled = bool(settings.get('phone_detection_enabled', self.phone_detection_enabled))
                self.phone_confidence = float(settings.get('phone_confidence', self.phone_confidence))
                self.phone_frames = int(settings.get('phone_frames', self.phone_frames))
                
                distraction_status = "ATIVADA" if self.phone_detection_enabled else "DESATIVADA"
                logging.info(f"Conf DMS Core(MediaPipe): EAR<{self.ear_threshold}({self.ear_frames}f), MAR>{self.mar_threshold}({self.mar_frames}f), Celular:{distraction_status} [Conf>{self.phone_confidence}({self.phone_frames}f)]")

                if not self.phone_detection_enabled:
                    self.phone_counter = 0
                    self.phone_alert_active = False
                    with self.yolo_lock:
                         self.phone_found_by_thread = False
                         self.last_yolo_boxes = []

                return True
            except (ValueError, TypeError) as e:
                logging.error(f"Erro conf MediaPipe (valor inválido?): {e}"); return False
            except Exception as e:
                logging.error(f"Erro inesperado conf MediaPipe: {e}", exc_info=True); return False

    def get_settings(self):
        logging.debug("DMSCore(MediaPipe): get_settings.")
        with self.lock:
            return {"ear_threshold": self.ear_threshold, "ear_frames": self.ear_frames,
                    "mar_threshold": self.mar_threshold, "mar_frames": self.mar_frames,
                    "phone_detection_enabled": self.phone_detection_enabled,
                    "phone_confidence": self.phone_confidence,
                    "phone_frames": self.phone_frames,
                   }