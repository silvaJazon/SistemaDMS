# Documentação: Núcleo do SistemaDMS (Implementação MediaPipe + YOLOv8)
# (ROADMAP 2.0: Convertido para Multiprocessing)
# (CORRIGIDO: Removido 'task_done' e 'half=True' para prevenir crashes)

import cv2
import mediapipe as mp
import numpy as np
import logging
import threading
import multiprocessing # (CORRIGIDO) Sem alias 'mp'
from scipy.spatial import distance as dist
from datetime import datetime
import time
import queue # Fila de threads (para comunicação interna)

from ultralytics import YOLO
from dms_base import BaseMonitor

cv2.setUseOptimized(True)

# --- Índices MediaPipe (permanecem iguais) ---
MP_LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE_IDX = [362, 385, 387, 263, 380, 373]
MP_MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 87]

NULL_LANDMARKS = np.zeros((6, 2), dtype="int")
NULL_LANDMARKS_MOUTH = np.zeros((8, 2), dtype="int")

# --- Nível de Log para o Processo Filho ---
def setup_logging(log_level):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - DMS(Worker) - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --- O PROCESSO DE CV ---

class _CVWorkerProcess(multiprocessing.Process):
    """
    Este processo corre num núcleo de CPU separado.
    Faz TODO o trabalho pesado de CV.
    """
    def __init__(self, frame_size, default_settings, stop_event,
                 input_queue, output_queue,
                 settings_in_queue, settings_out_queue,
                 log_level):
        super().__init__(name="CVWorkerProcess")
        self.daemon = True
        
        self.frame_height, self.frame_width = frame_size
        self.log_level = log_level
        
        self.stop_event = stop_event
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.settings_in_queue = settings_in_queue
        self.settings_out_queue = settings_out_queue
        
        self.face_mesh = None
        self.hands = None
        self.yolo_model = None
        self.yolo_cellphone_class_id = -1
        
        self.settings = default_settings.copy()
        
        self.drowsiness_counter = 0
        self.yawn_counter = 0
        self.phone_detected_time = None
        self.drowsy_alert_active = False
        self.yawn_alert_active = False
        self.phone_alert_active = False
        self._phone_event_sent = False

    # --- Funções de Cálculo (internas ao worker) ---
    def _eye_aspect_ratio(self, eye_landmarks):
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        return 0.3 if C < 1e-6 else (A + B) / (2.0 * C)

    def _mouth_aspect_ratio(self, mouth_landmarks):
        A = dist.euclidean(mouth_landmarks[1], mouth_landmarks[7])
        B = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])
        C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])
        return 0.0 if C < 1e-6 else (A + B) / (2.0 * C)

    def _get_landmarks_from_result(self, landmarks, indices):
        coords = np.zeros((len(indices), 2), dtype="int")
        for i, idx in enumerate(indices):
            lm = landmarks[idx]
            coords[i] = (int(lm.x * self.frame_width), int(lm.y * self.frame_height))
        return coords

    def _load_models(self):
        """Carrega os modelos de CV (chamado dentro do 'run')."""
        try:
            logging.info("A carregar MediaPipe (FaceMesh, Hands)...")
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
            )
            
            model_file = "models/yolov8n.pt" # Modelo NANO
            logging.info(f"A carregar YOLO ('{model_file}')...")
            self.yolo_model = YOLO(model_file)
            
            if self.yolo_model.names:
                for class_id, name in self.yolo_model.names.items():
                    if name == "cell phone":
                        self.yolo_cellphone_class_id = class_id
                        break
            logging.info(f"Classe 'cell phone' ID: {self.yolo_cellphone_class_id}")

            logging.info("A executar 'warm-up'...")
            dummy_frame_rgb = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            self.face_mesh.process(dummy_frame_rgb)
            self.hands.process(dummy_frame_rgb)
            dummy_crop_rgb = np.zeros((320, 320, 3), dtype=np.uint8)
            # (CORRIGIDO) 'half=False' para CPU
            self.yolo_model(dummy_crop_rgb, verbose=False, imgsz=320, half=False) 
            logging.info("Modelos de CV carregados e aquecidos.")
            return True
        except Exception as e:
            logging.error(f"!!! ERRO FATAL AO CARREGAR MODELOS: {e}", exc_info=True)
            return False

    def _check_for_settings_update(self):
        """Verifica se há novos settings vindos do app.py (não-bloqueante)."""
        try:
            # 1. Há um pedido para enviar os settings atuais?
            if not self.settings_out_queue.empty():
                _ = self.settings_out_queue.get_nowait() # Limpa o pedido
                self.settings_out_queue.put(self.settings.copy())
                logging.debug("Settings atuais enviados para o processo principal.")

            # 2. Há um pedido para atualizar os settings?
            if not self.settings_in_queue.empty():
                new_settings = self.settings_in_queue.get_nowait()
                self.settings.update(new_settings)
                logging.info(f"Settings atualizados: {self.settings}")
                
                if not self.settings.get("phone_detection_enabled", True):
                    self.phone_alert_active = False
                    self.phone_detected_time = None

        except queue.Empty:
            pass # Normal
        except Exception as e:
            logging.warning(f"Erro ao verificar settings: {e}")

    def run(self):
        """O loop principal do processo de CV."""
        setup_logging(self.log_level)
        
        if not self._load_models():
            self.stop_event.set()
            return

        frame_counter_skip = 0
        YOLO_FRAME_SKIP = 3 # O mesmo throttle

        while not self.stop_event.is_set():
            
            self._check_for_settings_update()
            
            frame_bgr = None
            try:
                frame_bgr = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                logging.debug("Fila de input vazia, a aguardar...")
                continue
            
            start_time_total = time.time()

            local_left_eye = NULL_LANDMARKS.copy()
            local_right_eye = NULL_LANDMARKS.copy()
            local_mouth = NULL_LANDMARKS_MOUTH.copy()
            local_boxes = []
            local_ear, local_mar = 0.5, 0.0
            new_events_list = []
            face_found = False
            phone_found_this_loop = False
            
            phone_enabled = self.settings.get("phone_detection_enabled", True)
            current_phone_confidence = self.settings.get("phone_confidence", 0.20)
            ear_thresh = self.settings.get("ear_threshold", 0.25)
            ear_frames_thresh = self.settings.get("ear_frames", 7)
            mar_thresh = self.settings.get("mar_threshold", 0.40)
            mar_frames_thresh = self.settings.get("mar_frames", 10)
            phone_seconds_thresh = self.settings.get("phone_frames", 5)

            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # 4.1. Inferência FaceMesh (EAR/MAR)
                results_mp = self.face_mesh.process(frame_rgb)
                if results_mp.multi_face_landmarks:
                    face_landmarks = results_mp.multi_face_landmarks[0].landmark
                    face_found = True
                    local_left_eye = self._get_landmarks_from_result(face_landmarks, MP_LEFT_EYE_IDX)
                    local_right_eye = self._get_landmarks_from_result(face_landmarks, MP_RIGHT_EYE_IDX)
                    local_mouth = self._get_landmarks_from_result(face_landmarks, MP_MOUTH_IDX)
                    local_ear = (self._eye_aspect_ratio(local_left_eye) + self._eye_aspect_ratio(local_right_eye)) / 2.0
                    local_mar = self._mouth_aspect_ratio(local_mouth)

                # 4.2. Inferência Híbrida (Mãos + YOLO) (COM THROTTLE)
                frame_counter_skip += 1
                if frame_counter_skip % YOLO_FRAME_SKIP == 0 and phone_enabled and self.yolo_cellphone_class_id != -1:
                    logging.debug("A executar inferência YOLO/Hands...")
                    results_hands = self.hands.process(frame_rgb)
                    if results_hands.multi_hand_landmarks:
                        for hand_landmarks in results_hands.multi_hand_landmarks:
                            h, w, _ = frame_bgr.shape
                            x_min, y_min = w, h
                            x_max, y_max = 0, 0
                            for lm in hand_landmarks.landmark:
                                x, y = int(lm.x * w), int(lm.y * h)
                                x_min, x_max = min(x_min, x), max(x_max, x)
                                y_min, y_max = min(y_min, y), max(y_max, y)

                            padding = 60
                            x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                            x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

                            if x_min >= x_max or y_min >= y_max: continue
                            hand_crop = frame_bgr[y_min:y_max, x_min:x_max]
                            if hand_crop.size == 0: continue

                            results_yolo = self.yolo_model(
                                hand_crop, verbose=False,
                                classes=[self.yolo_cellphone_class_id],
                                conf=current_phone_confidence,
                                imgsz=320, augment=False, 
                                half=False # (CORRIGIDO) half=False para CPU
                            )
                            if results_yolo and results_yolo[0].boxes:
                                for box in results_yolo[0].boxes:
                                    if int(box.cls) == self.yolo_cellphone_class_id:
                                        phone_found_this_loop = True
                                        local_boxes.append([int(c) for c in box.xyxy[0]])
                                        break
                            if phone_found_this_loop: break
                
                # --- 5. Lógica de Alerta ---
                current_time = time.time()
                
                if face_found:
                    if local_ear < ear_thresh:
                        self.drowsiness_counter += 1
                        if (self.drowsiness_counter >= ear_frames_thresh and not self.drowsy_alert_active):
                            self.drowsy_alert_active = True
                            new_events_list.append({
                                "type": "SONOLENCIA", "value": f"EAR: {local_ear:.2f}",
                                "timestamp": datetime.now().isoformat() + "Z",
                            })
                            logging.warning("EVENTO SONOLENCIA.")
                    else:
                        self.drowsiness_counter = 0
                        self.drowsy_alert_active = False
                    
                    if local_mar > mar_thresh:
                        self.yawn_counter += 1
                        if (self.yawn_counter >= mar_frames_thresh and not self.yawn_alert_active):
                            self.yawn_alert_active = True
                            new_events_list.append({
                                "type": "BOCEJO", "value": f"MAR: {local_mar:.2f}",
                                "timestamp": datetime.now().isoformat() + "Z",
                            })
                            logging.warning("EVENTO BOCEJO.")
                    else:
                        self.yawn_counter = 0
                        self.yawn_alert_active = False
                else:
                    self.drowsiness_counter = 0
                    self.drowsy_alert_active = False
                    self.yawn_counter = 0
                    self.yawn_alert_active = False

                if phone_enabled:
                    if phone_found_this_loop:
                        if self.phone_detected_time is None:
                            self.phone_detected_time = current_time
                        elapsed = current_time - self.phone_detected_time
                        if elapsed >= phone_seconds_thresh and not self.phone_alert_active:
                            self.phone_alert_active = True
                            logging.warning("EVENTO DISTRACAO (CELULAR NA MAO) ATIVADO.")
                    else:
                        self.phone_detected_time = None
                        self.phone_alert_active = False
                else:
                    self.phone_detected_time = None
                    self.phone_alert_active = False
                
                if self.phone_alert_active and not self._phone_event_sent:
                    self._phone_event_sent = True 
                    new_events_list.append({
                        "type": "DISTRACAO", "value": "Celular na mao",
                        "timestamp": datetime.now().isoformat() + "Z",
                    })
                    logging.warning("Gerando EVENTO DISTRACAO.")
                elif not self.phone_alert_active and self._phone_event_sent:
                    self._phone_event_sent = False
                
                # --- 6. Preparar dados de saída ---
                status_data = {"ear": f"{local_ear:.2f}", "mar": f"{local_mar:.2f}", "yaw": "-", "pitch": "-", "roll": "-"}
                annotations = {
                    "l_eye": local_left_eye, "r_eye": local_right_eye,
                    "mouth": local_mouth, "boxes": local_boxes,
                    "d_alert": self.drowsy_alert_active,
                    "y_alert": self.yawn_alert_active,
                    "p_alert": self.phone_alert_active,
                    "p_enabled": phone_enabled
                }
                
                # --- 7. Enviar resultados para o processo principal ---
                try:
                    # (CORRIGIDO) Envolve a limpeza da fila num try-except
                    try:
                        while not self.output_queue.empty():
                            self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                    
                    self.output_queue.put_nowait((status_data, new_events_list, annotations))
                except queue.Full:
                    logging.warning("Fila de output cheia, app.py está lento?")
                    pass

                logging.info(f"Inferência (Face+Hands+YOLO) completa. Duração: {time.time() - start_time_total:.3f}s")
                
            except Exception as e:
                logging.error(f"Erro na inferência: {e}", exc_info=True)
                self.phone_detected_time = None
            
            # (CORRIGIDO) Removida a chamada 'self.input_queue.task_done()'
            
        logging.info(">>> _cv_worker_loop (Single Worker Thread) terminado.")


# --- A CLASSE DE CONTROLO (usada por app.py) ---

class MediaPipeMonitor(BaseMonitor):
    """
    Esta classe corre no processo principal (app.py) e gere
    o processo _CVWorkerProcess.
    """
    
    def __init__(self, frame_size, stop_event: multiprocessing.Event, default_settings: dict):
        self.frame_height, self.frame_width = frame_size
        self.stop_event = stop_event
        self.default_settings = default_settings
        self.log_level = logging.getLogger().level

        self.input_queue = multiprocessing.Queue(maxsize=1)
        self.output_queue = multiprocessing.Queue(maxsize=1)
        self.settings_in_queue = multiprocessing.Queue(maxsize=1)
        self.settings_out_queue = multiprocessing.Queue(maxsize=1)
        
        self.worker_process = None
        
        self.last_annotations = {
            "l_eye": NULL_LANDMARKS.copy(), "r_eye": NULL_LANDMARKS.copy(),
            "mouth": NULL_LANDMARKS_MOUTH.copy(), "boxes": [],
            "d_alert": False, "y_alert": False, "p_alert": False, "p_enabled": True
        }
        self.last_sent_settings = default_settings.copy()

    def start_process(self):
        """Inicia o processo de CV."""
        logging.info("A iniciar o processo _CVWorkerProcess...")
        self.worker_process = _CVWorkerProcess(
            frame_size=(self.frame_height, self.frame_width),
            default_settings=self.default_settings,
            stop_event=self.stop_event,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            settings_in_queue=self.settings_in_queue,
            settings_out_queue=self.settings_out_queue,
            log_level=self.log_level
        )
        self.worker_process.start()
        return self.worker_process

    def is_alive(self):
        """Verifica se o processo de CV está a correr."""
        return self.worker_process is not None and self.worker_process.is_alive()

    def stop(self):
        """Sinaliza ao processo de CV para parar."""
        logging.info("A enviar sinal de paragem para o MediaPipeMonitor...")
        self.stop_event.set()
        try:
            while not self.input_queue.empty(): self.input_queue.get_nowait()
        except Exception: pass
        try:
            while not self.output_queue.empty(): self.output_queue.get_nowait()
        except Exception: pass
        

    # --- Funções de Interface (chamadas por app.py) ---

    def process_frame(self, frame: np.ndarray):
        """(LEVE) Apenas enfileira o frame para o processo de CV."""
        try:
            while not self.input_queue.empty():
                self.input_queue.get_nowait()
            self.input_queue.put_nowait(frame)
            logging.debug("Frame enviado para o processo de CV.")
        except queue.Full:
            logging.warning("Fila de input de CV cheia, frame descartado.")
        except Exception as e:
            logging.error(f"Erro ao enfileirar frame: {e}")
        
        pass 
    
    def get_results(self):
        """(LEVE) Pega os últimos resultados do processo de CV (não-bloqueante)."""
        try:
            results = self.output_queue.get_nowait() 
            if results:
                self.last_annotations = results[2] # Guarda o dict 'annotations'
            return results
        except queue.Empty:
            return None # Sem novos resultados
        except Exception as e:
            logging.warning(f"Erro ao ler fila de output: {e}")
            return None

    def draw_annotations(self, frame, annotations=None):
        """(LEVE) Desenha as *últimas* anotações conhecidas no frame atual."""
        if annotations is None:
            annotations = self.last_annotations
        
        try:
            if annotations["l_eye"].any():
                cv2.drawContours(frame, [cv2.convexHull(annotations["l_eye"])], -1, (0, 255, 0), 1)
            if annotations["r_eye"].any():
                cv2.drawContours(frame, [cv2.convexHull(annotations["r_eye"])], -1, (0, 255, 0), 1)
            if annotations["mouth"].any():
                cv2.drawContours(frame, [cv2.convexHull(annotations["mouth"])], -1, (0, 255, 255), 1)

            if annotations["p_enabled"]:
                for box_coords in annotations["boxes"]:
                    x1, y1, x2, y2 = map(int, box_coords)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(
                        frame, "Celular (na Mao)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
                    )
            
            if annotations["d_alert"]:
                cv2.putText(frame, "ALERTA: SONOLENCIA!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if annotations["y_alert"]:
                cv2.putText(frame, "ALERTA: BOCEJO!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if annotations["p_enabled"] and annotations["p_alert"]:
                cv2.putText(frame, "ALERTA: CELULAR/MAO!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        except Exception as e:
            logging.error(f"Erro ao desenhar anotações: {e}", exc_info=True)
            
        return frame


    def update_settings(self, settings: dict) -> bool:
        """(LEVE) Envia novos settings para o processo de CV."""
        try:
            while not self.settings_in_queue.empty():
                self.settings_in_queue.get_nowait()
            self.settings_in_queue.put_nowait(settings)
            self.last_sent_settings = settings.copy()
            logging.debug("Novos settings enviados para o processo de CV.")
            return True
        except queue.Full:
            logging.warning("Fila de settings cheia.")
            return False
        except Exception as e:
            logging.error(f"Erro ao enviar settings: {e}")
            return False
            
    def get_last_sent_settings(self):
        """(LEVE) Retorna o último dict de settings enviado."""
        return self.last_sent_settings

    def get_settings(self) -> dict:
        """(LEVE) Pede os settings atuais ao processo de CV."""
        try:
            while not self.settings_out_queue.empty():
                self.settings_out_queue.get_nowait()
            self.settings_out_queue.put_nowait("GET")
            
            try:
                settings = self.settings_out_queue.get(timeout=0.5) # Espera 500ms
                return settings
            except queue.Empty:
                logging.warning("Timeout ao pedir settings ao processo de CV.")
                return None
        except Exception as e:
            logging.error(f"Erro ao pedir settings: {e}")
            return None