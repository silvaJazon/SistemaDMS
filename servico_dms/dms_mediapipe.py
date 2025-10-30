# Documentação: Núcleo do SistemaDMS (Implementação MediaPipe + YOLOv8)
# Focada em Sonolência (EAR), Bocejo (MAR) e Deteção de Celular (YOLO)

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

# (NOVO) Importa YOLO
from ultralytics import YOLO

from dms_base import BaseMonitor

cv2.setUseOptimized(True)

# (NOVO) Stride para Deteção de Celular (executa o modelo a cada N frames)
PHONE_DETECTION_STRIDE = 5 # Executa YOLO a cada 5 frames

# --- Constantes de Índices do MediaPipe Face Mesh (478 landmarks) ---
MP_LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE_IDX = [362, 385, 387, 263, 380, 373]
MP_MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 87]

class MediaPipeMonitor(BaseMonitor):
    """
    Implementação do BaseMonitor:
    - MediaPipe Face Mesh (EAR/MAR)
    - YOLOv8 (Deteção de Celular)
    """

    def __init__(self, frame_size, default_settings: dict = None):
        super().__init__(frame_size, default_settings)
        logging.info("A inicializar o MediaPipeMonitor Core (Modo: MediaPipe EAR/MAR + YOLO Celular)...")

        # --- 1. Inicializa o MediaPipe FaceMesh ---
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logging.info(">>> Modelos MediaPipe FaceMesh carregados.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL MediaPipe: {e}", exc_info=True)
            raise RuntimeError(f"Erro MediaPipe: {e}")

        # --- 2. (NOVO) Carregar Modelo YOLOv8 ---
        try:
            logging.info(">>> Carregando modelo YOLOv8n ('yolov8n.pt')...")
            self.yolo_model = YOLO('yolov8n.pt')
            logging.info(">>> Modelo YOLOv8n carregado.")
            
            self.yolo_cellphone_class_id = -1
            if self.yolo_model.names:
                for class_id, name in self.yolo_model.names.items():
                    if name == 'cell phone':
                        self.yolo_cellphone_class_id = class_id
                        logging.info(f"Classe 'cell phone' encontrada no YOLO. ID: {class_id}")
                        break
            if self.yolo_cellphone_class_id == -1:
                 logging.warning("!!! Classe 'cell phone' não encontrada nos nomes do modelo YOLO.")
                 
        except Exception as e:
            logging.error(f"!!! ERRO FATAL YOLO: {e}", exc_info=True)
            raise RuntimeError(f"Erro YOLO: {e}")


        # --- 3. Contadores e Configurações ---
        self.lock = threading.Lock()
        self.drowsiness_counter = 0
        self.yawn_counter = 0
        self.phone_counter = 0 # (NOVO)
        self.frame_counter = 0 # (NOVO)
        
        self.drowsy_alert_active = False
        self.yawn_alert_active = False
        self.phone_alert_active = False # (NOVO)

        # Configurações (Padrão)
        self.ear_threshold = self.default_settings.get('ear_threshold', 0.25)
        self.ear_frames = self.default_settings.get('ear_frames', 7)
        self.mar_threshold = self.default_settings.get('mar_threshold', 0.40)
        self.mar_frames = self.default_settings.get('mar_frames', 10)
        
        # (ALTERADO) Configurações de Deteção de Celular (agora ativas)
        self.phone_detection_enabled = False
        self.phone_confidence = 0.50
        self.phone_frames = 20

    # ... (Funções _eye_aspect_ratio, _mouth_aspect_ratio, _get_landmarks_from_result permanecem iguais) ...
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
        """Analisa um frame (MediaPipe EAR/MAR + YOLO Celular)."""
        logging.debug("DMSCore(MediaPipe): process_frame (MP + YOLO) iniciado.")
        start_time_total = time.time()
        events_list = []
        status_data = {"ear": "-", "mar": "-", "yaw": "-", "pitch": "-", "roll": "-"}
        face_found_this_frame = False
        
        self.frame_counter += 1 # (NOVO) Incrementa contador de frames

        # --- (NOVO) Lógica de Deteção de Celular (YOLOv8) ---
        phone_found_this_stride = False
        phone_enabled_locked = False # Lê o estado dentro do lock
        
        with self.lock:
            phone_enabled_locked = self.phone_detection_enabled

        if phone_enabled_locked and (self.frame_counter % PHONE_DETECTION_STRIDE == 0) and self.yolo_cellphone_class_id != -1:
            logging.debug("DMSCore(YOLO): Executando deteção de celular (YOLO)...")
            start_time_yolo = time.time()
            
            try:
                results = self.yolo_model(frame, verbose=False, classes=[self.yolo_cellphone_class_id], conf=self.phone_confidence)
                
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        if int(box.cls) == self.yolo_cellphone_class_id:
                            phone_found_this_stride = True
                            logging.debug(f"DMSCore(YOLO): Celular encontrado! Conf: {box.conf.item():.2f}")
                            
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cv2.putText(frame, f"Celular {box.conf.item():.2f}", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            break
                            
            except Exception as e:
                logging.error(f"DMSCore(YOLO): Erro durante inferência YOLO: {e}", exc_info=True)
            logging.debug(f"DMSCore(YOLO): Deteção YOLO {time.time()-start_time_yolo:.4f}s. Encontrado={phone_found_this_stride}")
        
        elif not phone_enabled_locked: # Se desativado, reseta
             with self.lock:
                if self.phone_counter > 0 or self.phone_alert_active:
                    self.phone_counter = 0; self.phone_alert_active = False
                    logging.debug("DMSCore(YOLO): Deteção celular desativada, resetando contador/flag.")
        
        
        # --- Lógica de Deteção Facial (MediaPipe) ---
        # MediaPipe espera frames RGB
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_mp = self.face_mesh.process(frame_rgb)
        except cv2.error as e:
            logging.error(f"DMSCore(MediaPipe): Erro ao converter BGR->RGB: {e}")
            return frame, events_list, status_data
        except Exception as e:
            logging.error(f"DMSCore(MediaPipe): Erro .process(): {e}", exc_info=True)
            return frame, events_list, status_data

        logging.debug(f"DMSCore(MediaPipe): Processamento MP {time.time() - start_time_total:.4f}s.")

        if results_mp.multi_face_landmarks:
            face_landmarks = results_mp.multi_face_landmarks[0].landmark
            face_found_this_frame = True

            try:
                left_eye_pts = self._get_landmarks_from_result(face_landmarks, MP_LEFT_EYE_IDX)
                right_eye_pts = self._get_landmarks_from_result(face_landmarks, MP_RIGHT_EYE_IDX)
                mouth_pts = self._get_landmarks_from_result(face_landmarks, MP_MOUTH_IDX)
                
                ear_left = self._eye_aspect_ratio(left_eye_pts); ear_right = self._eye_aspect_ratio(right_eye_pts)
                ear = (ear_left + ear_right) / 2.0; logging.debug(f"DMSCore(MediaPipe): EAR={ear:.3f}")
                mar = self._mouth_aspect_ratio(mouth_pts); logging.debug(f"DMSCore(MediaPipe): MAR={mar:.3f}")

                status_data["ear"] = f"{ear:.2f}"; status_data["mar"] = f"{mar:.2f}"

                cv2.drawContours(frame, [cv2.convexHull(left_eye_pts)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye_pts)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(mouth_pts)], -1, (0, 255, 255), 1)

            except Exception as e:
                logging.error(f"DMSCore(MediaPipe): Erro ao processar landmarks: {e}", exc_info=True)
                face_found_this_frame = False

        
        # --- Lógica de Alerta (Combinada) ---
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
                    self.drowsiness_counter = 0
                    self.drowsy_alert_active = False

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
                    self.yawn_counter = 0
                    self.yawn_alert_active = False
            
            else: # Nenhuma face encontrada
                logging.debug("DMSCore(MediaPipe): Nenhuma face encontrada.")
                # Reseta contadores faciais se a cara for perdida
                self.drowsiness_counter = 0; self.drowsy_alert_active = False
                self.yawn_counter = 0; self.yawn_alert_active = False

            # (NOVO) Lógica de Alerta (Celular) - (executa mesmo sem rosto)
            if phone_enabled_locked and (self.frame_counter % PHONE_DETECTION_STRIDE == 0):
                if phone_found_this_stride:
                    self.phone_counter += PHONE_DETECTION_STRIDE # Incrementa o "stride"
                    logging.debug(f"DMSCore(YOLO): Celular encontrado, cont={self.phone_counter}/{self.phone_frames}")
                    if self.phone_counter >= self.phone_frames and not self.phone_alert_active:
                        self.phone_alert_active = True
                        events_list.append({"type": "DISTRACAO", "value": "Celular detectado", "timestamp": datetime.now().isoformat() + "Z"})
                        logging.warning("DMSCore(YOLO): EVENTO DISTRACAO (CELULAR).")
                else:
                    if self.phone_counter > 0: logging.debug("DMSCore(YOLO): Deteção celular reset.")
                    self.phone_counter = 0; self.phone_alert_active = False

        logging.debug("DMSCore(MediaPipe): Lock alerta libertado.")

        # Desenha alertas
        if self.drowsy_alert_active:
            cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if self.yawn_alert_active:
            cv2.putText(frame, "ALERTA: BOCEJO!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # (NOVO) Alerta de Celular
        if phone_enabled_locked and self.phone_alert_active:
             cv2.putText(frame, "ALERTA: CELULAR!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)


        total_time = time.time() - start_time_total
        logging.debug(f"DMSCore(MediaPipe): process_frame (MP + YOLO) {total_time:.4f}s.")
        return frame, events_list, status_data

    def update_settings(self, settings):
        """Atualiza configurações."""
        logging.debug(f"DMSCore(MediaPipe): Tentando atualizar conf: {settings}")
        with self.lock:
            try:
                self.ear_threshold = float(settings.get('ear_threshold', self.ear_threshold))
                self.ear_frames = int(settings.get('ear_frames', self.ear_frames))
                self.mar_threshold = float(settings.get('mar_threshold', self.mar_threshold))
                self.mar_frames = int(settings.get('mar_frames', '10')) # Corrigido para str
                
                # (ALTERADO) Ativa a leitura dos settings de celular
                self.phone_detection_enabled = bool(settings.get('phone_detection_enabled', self.phone_detection_enabled))
                self.phone_confidence = float(settings.get('phone_confidence', self.phone_confidence))
                self.phone_frames = int(settings.get('phone_frames', self.phone_frames))
                
                # (REMOVIDO) O aviso "Não implementada"
                
                distraction_status = "ATIVADA" if self.phone_detection_enabled else "DESATIVADA"
                logging.info(f"Conf DMS Core(MediaPipe): EAR<{self.ear_threshold}({self.ear_frames}f), MAR>{self.mar_threshold}({self.mar_frames}f), Celular:{distraction_status} [Conf>{self.phone_confidence}({self.phone_frames}f)]")

                # (NOVO) Se a deteção foi desativada, reseta
                if not self.phone_detection_enabled:
                    self.phone_counter = 0
                    self.phone_alert_active = False

                return True
            except (ValueError, TypeError) as e:
                logging.error(f"Erro conf MediaPipe (valor inválido?): {e}"); return False
            except Exception as e:
                logging.error(f"Erro inesperado conf MediaPipe: {e}", exc_info=True); return False

    def get_settings(self):
        """Obtém configurações."""
        logging.debug("DMSCore(MediaPipe): get_settings.")
        with self.lock:
            # (SEM ALTERAÇÃO) Esta função já estava correta
            return {"ear_threshold": self.ear_threshold, "ear_frames": self.ear_frames,
                    "mar_threshold": self.mar_threshold, "mar_frames": self.mar_frames,
                    
                    "phone_detection_enabled": self.phone_detection_enabled,
                    "phone_confidence": self.phone_confidence,
                    "phone_frames": self.phone_frames,
                   }