# Documentação: Núcleo do SistemaDMS (Implementação Dlib + YOLOv8)
# (VERSÃO: Híbrido + Deteção de Celular (YOLOv8))

import cv2
import dlib
import numpy as np
import logging
import math
import threading
from scipy.spatial import distance as dist
from datetime import datetime
import time
import sys
from collections import deque

# (NOVO) Importa YOLO
from ultralytics import YOLO

# (NOVO) Importa a classe base
from dms_base import BaseMonitor

cv2.setUseOptimized(True)

MOUTH_AR_START = 60
MOUTH_AR_END = 68
FRAMES_FOR_REDETECTION = 10
TRACKER_CONFIDENCE_THRESHOLD = 5.5
# (REMOVIDO) ANGLE_SMOOTHING_FRAMES

# (NOVO) Stride para Deteção de Celular (executa o modelo a cada N frames)
PHONE_DETECTION_STRIDE = 5 # Executa YOLO a cada 5 frames

class DlibMonitor(BaseMonitor):
    """
    Classe principal para a deteção de sonolência e (NOVO) telemóvel.
    (VERSÃO: Híbrido Dlib + Deteção de Celular YOLOv8)
    """
    EYE_AR_LEFT_START = 42; EYE_AR_LEFT_END = 48
    EYE_AR_RIGHT_START = 36; EYE_AR_RIGHT_END = 42

    # (REMOVIDO) Lógica de Pose 3D (HEAD_MODEL_POINTS, HEAD_IMAGE_POINTS_IDX)

    def __init__(self, frame_size, default_settings: dict = None):
        super().__init__(frame_size, default_settings)
        
        logging.info("A inicializar o DlibMonitor Core (Modo: Híbrido Dlib + Deteção Celular YOLOv8)...")

        # --- Carregar Modelos Dlib ---
        self.detector = None
        self.predictor = None
        self.initialize_dlib()

        # --- (NOVO) Carregar Modelo YOLOv8 ---
        try:
            logging.info(">>> Carregando modelo YOLOv8n ('yolov8n.pt')...")
            self.yolo_model = YOLO('yolov8n.pt')
            logging.info(">>> Modelo YOLOv8n carregado.")
            
            # (NOVO) Encontra o ID da classe "cell phone" no modelo
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

        self.lock = threading.Lock()
        self.drowsiness_counter = 0
        self.yawn_counter = 0
        self.phone_counter = 0 # (RENOMEADO) de distraction_counter
        self.frame_counter = 0 # (NOVO) Para o stride
        
        self.drowsy_alert_active = False
        self.yawn_alert_active = False
        self.phone_alert_active = False # (RENOMEADO) de distraction_alert_active

        self.face_tracker = None; self.tracking_active = False; self.tracked_rect = None; self.frame_since_detection = 0
        # (REMOVIDO) Histórico de Ângulos (yaw_history, pitch_history)

        # Configurações (Padrão)
        self.ear_threshold = self.default_settings.get('ear_threshold', 0.25)
        self.ear_frames = self.default_settings.get('ear_frames', 7)
        self.mar_threshold = self.default_settings.get('mar_threshold', 0.40)
        self.mar_frames = self.default_settings.get('mar_frames', 10)
        
        # (ALTERADO) Configurações de Deteção de Celular
        self.phone_detection_enabled = False # (RENOMEADO)
        self.phone_confidence = 0.50 # (NOVO) Confiança mínima
        self.phone_frames = 20 # (RENOMEADO) N frames para alerta (antigo distraction_frames)

    def initialize_dlib(self):
        """Carrega modelos Dlib."""
        try:
            logging.info(">>> Carregando detetor faces (Dlib)...")
            self.detector = dlib.get_frontal_face_detector()
            model_path = 'shape_predictor_68_face_landmarks.dat'
            logging.info(f">>> Carregando preditor landmarks ({model_path})...")
            self.predictor = dlib.shape_predictor(model_path)
            logging.info(">>> Modelos Dlib carregados.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL Dlib ({model_path}): {e}", exc_info=True)
            raise RuntimeError(f"Erro Dlib: {e}")

    # ... (Funções _shape_to_np, _eye_aspect_ratio, _mouth_aspect_ratio permanecem iguais) ...
    def _shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(shape.num_parts): coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    def _eye_aspect_ratio(self, eye):
        A=dist.euclidean(eye[1],eye[5]); B=dist.euclidean(eye[2],eye[4]); C=dist.euclidean(eye[0],eye[3])
        return 0.3 if C < 1e-6 else (A + B) / (2.0 * C)
    def _mouth_aspect_ratio(self, mouth):
        A=dist.euclidean(mouth[1],mouth[7]); B=dist.euclidean(mouth[3],mouth[5]); C=dist.euclidean(mouth[0],mouth[4])
        return 0.0 if C < 1e-6 else (A + B) / (2.0 * C)

    # (REMOVIDO) def _estimate_head_pose(self, shape_np): ...

    def _reset_counters_and_cooldowns(self):
        """Reinicia contadores e cooldowns."""
        logging.debug("DMSCore(Dlib): Reset counters/cooldowns.")
        self.drowsiness_counter=0; self.yawn_counter=0; self.phone_counter = 0
        self.drowsy_alert_active=False; self.yawn_alert_active=False; self.phone_alert_active = False
        # (REMOVIDO) Reset de histórico de ângulos

    def process_frame(self, frame, gray):
        """Analisa um frame (Híbrido Dlib + Deteção Celular YOLOv8)."""
        logging.debug("DMSCore(Dlib): process_frame (HÍBRIDO + YOLO CELULAR) iniciado.")
        start_time_total = time.time()
        events_list = []
        # (ALTERADO) Status data não tem pose
        status_data = {"ear": "-", "mar": "-", "yaw": "-", "pitch": "-", "roll": "-"}
        face_found_this_frame = False
        gray_processed = cv2.equalizeHist(gray)
        current_rect = None
        
        self.frame_counter += 1 # Incrementa contador de frames

        # --- Lógica de Deteção/Tracking de Rosto (Dlib) ---
        logging.debug("DMSCore(Dlib): Lock tracking...")
        with self.lock:
            # (ALTERADO) Lê estado de ativação da deteção de celular
            phone_enabled = self.phone_detection_enabled

            logging.debug("DMSCore(Dlib): Lock tracking OK.")
            needs_detection = (not self.tracking_active) or (self.frame_since_detection >= FRAMES_FOR_REDETECTION)
            logging.debug(f"DMSCore(Dlib): Track={self.tracking_active}, Frames={self.frame_since_detection}, Detect={needs_detection}")
            if needs_detection:
                logging.debug("DMSCore(Dlib): Detecção completa...")
                start_time_detect = time.time()
                rects = self.detector(gray_processed, 0)
                logging.debug(f"DMSCore(Dlib): Deteção {len(rects)} faces {time.time()-start_time_detect:.4f}s.")
                if rects:
                    current_rect = max(rects, key=lambda r: r.width()*r.height())
                    if self.face_tracker is None: self.face_tracker = dlib.correlation_tracker()
                    logging.debug("DMSCore(Dlib): Iniciando tracker...")
                    self.face_tracker.start_track(frame, current_rect)
                    self.tracking_active=True; self.tracked_rect=current_rect; self.frame_since_detection=0; face_found_this_frame=True
                else:
                    logging.debug("DMSCore(Dlib): Nenhuma face detetada.")
                    if self.tracking_active: self._reset_counters_and_cooldowns()
                    self.tracking_active=False; self.tracked_rect=None
            elif self.tracking_active:
                logging.debug("DMSCore(Dlib): Tracking...")
                start_time_track = time.time()
                confidence = self.face_tracker.update(frame)
                logging.debug(f"DMSCore(Dlib): Track conf={confidence:.2f} {time.time()-start_time_track:.4f}s.")
                if confidence > TRACKER_CONFIDENCE_THRESHOLD:
                    self.tracked_rect = self.face_tracker.get_position()
                    current_rect = dlib.rectangle(int(self.tracked_rect.left()), int(self.tracked_rect.top()),
                                                   int(self.tracked_rect.right()), int(self.tracked_rect.bottom()))
                    self.frame_since_detection += 1; face_found_this_frame = True
                else:
                    logging.debug(f"DMSCore(Dlib): Tracker perdeu face (conf {confidence:.2f}).")
                    self._reset_counters_and_cooldowns(); self.tracking_active=False; self.tracked_rect=None
        logging.debug("DMSCore(Dlib): Lock tracking libertado.")

        # --- (NOVO) Lógica de Deteção de Celular (YOLOv8) ---
        phone_found_this_stride = False
        if phone_enabled and (self.frame_counter % PHONE_DETECTION_STRIDE == 0) and self.yolo_cellphone_class_id != -1:
            logging.debug("DMSCore(YOLO): Executando deteção de celular (YOLO)...")
            start_time_yolo = time.time()
            
            # Executa a inferência YOLO. verbose=False desliga os logs do YOLO.
            try:
                # Usamos o 'frame' original (BGR), que é o que o YOLO espera
                results = self.yolo_model(frame, verbose=False, classes=[self.yolo_cellphone_class_id], conf=self.phone_confidence)
                
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        # Verifica se a classe é 'cell phone' (apesar do filtro, é uma boa prática)
                        # e se a confiança é suficiente
                        if int(box.cls) == self.yolo_cellphone_class_id:
                            phone_found_this_stride = True
                            logging.debug(f"DMSCore(YOLO): Celular encontrado! Conf: {box.conf.item():.2f}")
                            
                            # (NOVO) Desenha a caixa no frame
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cv2.putText(frame, f"Celular {box.conf.item():.2f}", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            
                            # Encontrou um, não precisa de procurar mais neste frame
                            break 
                            
            except Exception as e:
                logging.error(f"DMSCore(YOLO): Erro durante inferência YOLO: {e}", exc_info=True)

            logging.debug(f"DMSCore(YOLO): Deteção YOLO {time.time()-start_time_yolo:.4f}s. Encontrado={phone_found_this_stride}")
        
        elif not phone_enabled: # Se desativado, reseta
             with self.lock:
                if self.phone_counter > 0 or self.phone_alert_active:
                    self.phone_counter = 0; self.phone_alert_active = False
                    logging.debug("DMSCore(YOLO): Deteção celular desativada, resetando contador/flag.")

        # --- Processamento dos Landmarks (Rosto) ---
        if face_found_this_frame and current_rect is not None:
            logging.debug("DMSCore(Dlib): Prevendo landmarks...")
            start_time_predict = time.time()
            shape = self.predictor(gray, current_rect)
            shape_np = self._shape_to_np(shape)
            logging.debug(f"DMSCore(Dlib): Landmarks {time.time()-start_time_predict:.4f}s.")

            # Cálculos EAR, MAR
            leftEye=shape_np[self.EYE_AR_LEFT_START:self.EYE_AR_LEFT_END]; rightEye=shape_np[self.EYE_AR_RIGHT_START:self.EYE_AR_RIGHT_END]
            ear = (self._eye_aspect_ratio(leftEye) + self._eye_aspect_ratio(rightEye)) / 2.0; logging.debug(f"DMSCore(Dlib): EAR={ear:.3f}")
            mouth = shape_np[MOUTH_AR_START:MOUTH_AR_END]; mar = self._mouth_aspect_ratio(mouth); logging.debug(f"DMSCore(Dlib): MAR={mar:.3f}")
            
            # (REMOVIDO) Cálculo de Pose

            # Desenho (Rosto)
            try:
                cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1); cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 255), 1)
                pt1=(int(current_rect.left()), int(current_rect.top())); pt2=(int(current_rect.right()), int(current_rect.bottom()))
                cv2.rectangle(frame, pt1, pt2, (255, 255, 0), 2)
            except Exception as e: logging.warning(f"DMSCore(Dlib): Erro desenho: {e}")

            # --- Lógica de Alerta (Sonolência, Bocejo) ---
            logging.debug("DMSCore(Dlib): Lock alerta...")
            with self.lock:
                logging.debug("DMSCore(Dlib): Lock alerta OK.")
                # Sonolência (igual)
                if ear < self.ear_threshold:
                    self.drowsiness_counter += 1; logging.debug(f"DMSCore(Dlib): EAR baixo ({ear:.3f}<{self.ear_threshold}), cont={self.drowsiness_counter}/{self.ear_frames}")
                    if self.drowsiness_counter >= self.ear_frames and not self.drowsy_alert_active:
                        self.drowsy_alert_active=True; events_list.append({"type":"SONOLENCIA", "value":f"EAR: {ear:.2f}", "timestamp":datetime.now().isoformat()+"Z"}); logging.warning("DMSCore(Dlib): EVENTO SONOLENCIA.")
                else:
                    if self.drowsiness_counter > 0: logging.debug("DMSCore(Dlib): Sonolência reset.")
                    self.drowsiness_counter=0; self.drowsy_alert_active=False
                # Bocejo (igual)
                if mar > self.mar_threshold:
                    self.yawn_counter += 1; logging.debug(f"DMSCore(Dlib): MAR alto ({mar:.3f}>{self.mar_threshold}), cont={self.yawn_counter}/{self.mar_frames}")
                    if self.yawn_counter >= self.mar_frames and not self.yawn_alert_active:
                        self.yawn_alert_active=True; events_list.append({"type":"BOCEJO", "value":f"MAR: {mar:.2f}", "timestamp":datetime.now().isoformat()+"Z"}); logging.warning("DMSCore(Dlib): EVENTO BOCEJO.")
                else:
                    if self.yawn_counter > 0: logging.debug("DMSCore(Dlib): Bocejo reset.")
                    self.yawn_counter=0; self.yawn_alert_active=False

                # --- (ALTERADO) Lógica de Alerta (Celular) ---
                # Esta lógica é atualizada apenas nos frames em que o YOLO corre
                if phone_enabled and (self.frame_counter % PHONE_DETECTION_STRIDE == 0):
                    if phone_found_this_stride:
                        self.phone_counter += PHONE_DETECTION_STRIDE # Incrementa o "stride"
                        logging.debug(f"DMSCore(YOLO): Celular encontrado, cont={self.phone_counter}/{self.phone_frames}")
                        if self.phone_counter >= self.phone_frames and not self.phone_alert_active:
                            self.phone_alert_active = True
                            # (ALTERADO) Tipo de evento para "DISTRACAO" (ou "CELULAR")
                            events_list.append({"type": "DISTRACAO", "value": "Celular detectado", "timestamp": datetime.now().isoformat() + "Z"})
                            logging.warning("DMSCore(YOLO): EVENTO DISTRACAO (CELULAR).")
                    else:
                        if self.phone_counter > 0: logging.debug("DMSCore(YOLO): Deteção celular reset.")
                        self.phone_counter = 0; self.phone_alert_active = False
                
            logging.debug("DMSCore(Dlib): Lock alerta libertado.")

            # (ALTERADO) Status data
            status_data = {"ear": f"{ear:.2f}", "mar": f"{mar:.2f}",
                           "yaw": "-", "pitch": "-", "roll": "-"} # Pose removida
                           
            # Desenha alertas
            if self.drowsy_alert_active: cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.yawn_alert_active: cv2.putText(frame, "ALERTA: BOCEJO!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # (ALTERADO) Alerta de Celular
            if phone_enabled and self.phone_alert_active:
                 cv2.putText(frame, "ALERTA: CELULAR!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        else: # Se nenhuma face foi encontrada
             logging.debug("DMSCore(Dlib): Nenhuma face encontrada/rastreada.")
             # Reset já feito dentro do lock de tracking ou no início do loop se rects vazio

        total_time = time.time() - start_time_total
        logging.debug(f"DMSCore(Dlib): process_frame (HÍBRIDO + YOLO CELULAR) {total_time:.4f}s.")
        return frame, events_list, status_data

    def update_settings(self, settings):
        """Atualiza configurações."""
        logging.debug(f"DMSCore(Dlib): Tentando atualizar conf: {settings}")
        with self.lock:
            try:
                self.ear_threshold = float(settings.get('ear_threshold', self.ear_threshold))
                self.ear_frames = int(settings.get('ear_frames', self.ear_frames))
                self.mar_threshold = float(settings.get('mar_threshold', self.mar_threshold))
                self.mar_frames = int(settings.get('mar_frames', self.mar_frames))
                
                # (ALTERADO) Atualiza settings de celular
                self.phone_detection_enabled = bool(settings.get('phone_detection_enabled', self.phone_detection_enabled))
                self.phone_confidence = float(settings.get('phone_confidence', self.phone_confidence))
                self.phone_frames = int(settings.get('phone_frames', self.phone_frames))
                
                # (REMOVIDO) Settings de distração/pose

                # Log atualizado com estado de ativação
                distraction_status = "ATIVADA" if self.phone_detection_enabled else "DESATIVADA"
                logging.info(f"Conf DMS Core(Dlib): EAR<{self.ear_threshold}({self.ear_frames}f), MAR>{self.mar_threshold}({self.mar_frames}f), Celular:{distraction_status} [Conf>{self.phone_confidence}({self.phone_frames}f)]")

                # (NOVO) Se a deteção foi desativada, reseta
                if not self.phone_detection_enabled:
                    self.phone_counter = 0
                    self.phone_alert_active = False

                return True
            except (ValueError, TypeError) as e: logging.error(f"Erro conf Dlib (valor inválido?): {e}"); return False
            except Exception as e: logging.error(f"Erro inesperado conf Dlib: {e}", exc_info=True); return False

    def get_settings(self):
        """Obtém configurações."""
        logging.debug("DMSCore(Dlib): get_settings.")
        with self.lock:
            # (ALTERADO) Retorna settings de celular
            return {"ear_threshold": self.ear_threshold, "ear_frames": self.ear_frames,
                    "mar_threshold": self.mar_threshold, "mar_frames": self.mar_frames,
                    
                    "phone_detection_enabled": self.phone_detection_enabled,
                    "phone_confidence": self.phone_confidence,
                    "phone_frames": self.phone_frames,
                   }