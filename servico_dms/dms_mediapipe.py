# Documentação: Núcleo do SistemaDMS
# (Versão com estabilização de EAR (Grace Period))
# (Versão com Calibração Automática de EAR)
# (CORREÇÃO 1.1: Lógica de Latch de Alerta)

import cv2
import mediapipe as mp
import numpy as np
import logging
import threading
from scipy.spatial import distance as dist
from datetime import datetime
import time

from ultralytics import YOLO
from dms_base import BaseMonitor
from camera_thread import CameraThread

cv2.setUseOptimized(True)

# --- Índices MediaPipe ---
MP_LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE_IDX = [362, 385, 387, 263, 380, 373]
MP_MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 87]


class MediaPipeMonitor(BaseMonitor):
    """
    Implementação Multithread:
    - Thread 1 (Principal): MediaPipe Face Mesh (EAR/MAR)
    - Thread 2 (Fundo): Híbrido Otimizado:
        1. MediaPipe Hands (Rápido)
        2. Se Mão encontrada -> YOLOvxxo recorte da Mão (Rápido)
    """

    def __init__(
        self, frame_size, stop_event: threading.Event, default_settings: dict = None
    ):
        super().__init__(frame_size, stop_event, default_settings)
        logging.info(
            "A inicializar o MediaPipeMonitor Core "
            "(Modo: Híbrido Otimizado MP-Mão + YOLO-Recorte)..."
        )

        # --- 1. Inicializa o MediaPipe FaceMesh (Thread Principal) ---
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logging.info(">>> Modelos MediaPipe FaceMesh carregados.")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL MediaPipe (FaceMesh): {e}", exc_info=True)
            raise RuntimeError(f"Erro MediaPipe (FaceMesh): {e}")

        # --- 2. Inicializa o MediaPipe Hands (Thread Fundo) ---
        try:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
            )
            logging.info(">>> Modelos MediaPipe Hands carregados (max_num_hands=2).")
        except Exception as e:
            logging.error(f"!!! ERRO FATAL MediaPipe (Hands): {e}", exc_info=True)
            raise RuntimeError(f"Erro MediaPipe (Hands): {e}")

        # --- 3. Carregar Modelo YOLOv8 (Thread Fundo) ---
        try:
            model_file = "models/yolov8n.pt"
            logging.info(f">>> Carregando modelo YOLOv8 ('{model_file}')...")
            self.yolo_model = YOLO(model_file)
            logging.info(f">>> Modelo {model_file} carregado.")

            self.yolo_cellphone_class_id = -1
            if self.yolo_model.names:
                for class_id, name in self.yolo_model.names.items():
                    if name == "cell phone":
                        self.yolo_cellphone_class_id = class_id
                        logging.info(
                            f"Classe 'cell phone' encontrada no YOLO. ID: {class_id}"
                        )
                        break
            if self.yolo_cellphone_class_id == -1:
                logging.warning(
                    "!!! Classe 'cell phone' não encontrada nos nomes do modelo YOLO."
                )

            logging.info(">>> Executando 'warm-up' (primeira inferência)...")
            try:
                dummy_frame_rgb = np.zeros(
                    (self.frame_height, self.frame_width, 3), dtype=np.uint8
                )
                self.hands.process(dummy_frame_rgb)
                logging.info(">>> Warm-up (Hands) concluído.")

                dummy_crop_rgb = np.zeros((320, 320, 3), dtype=np.uint8)
                self.yolo_model(dummy_crop_rgb, verbose=False, imgsz=320)
                logging.info(">>> Warm-up (YOLO-Recorte) concluído.")

            except Exception as e:
                logging.warning(f"Falha no warm-up: {e}")

        except Exception as e:
            logging.error(f"!!! ERRO FATAL YOLO: {e}", exc_info=True)
            raise RuntimeError(f"Erro YOLO: {e}")

        # --- 4. Contadores e Configurações ---
        self.lock = threading.RLock() # (Usar RLock é ligeiramente mais seguro)
        
        self.drowsiness_counter = 0
        self.yawn_counter = 0
        
        self.drowsiness_reset_counter = 0
        self.yawn_reset_counter = 0 # NOVO: Contador de reset para Bocejo

        # --- NOVAS VARIÁVEIS DE CALIBRAÇÃO ---
        self.calibration_state = self.default_settings.get("calibration_state", "IDLE")
        self.calibration_samples = []
        self.CALIBRATION_FRAMES_TARGET = 100 
        # -------------------------------------

        self.drowsy_alert_active = False
        self.yawn_alert_active = False
        self.phone_alert_active = False

        self.ear_threshold = self.default_settings.get("ear_threshold", 0.30)
        self.ear_frames = self.default_settings.get("ear_frames", 2)
        self.ear_reset_frames = self.default_settings.get("ear_reset_frames", 5) 
        self.ear_calibration_factor = self.default_settings.get("ear_calibration_factor", 0.80)
        
        self.mar_threshold = self.default_settings.get("mar_threshold", 0.40)
        self.mar_frames = self.default_settings.get("mar_frames", 2)
        # NOVO: Frames de reset para bocejo (lógica de Grace Period)
        self.mar_reset_frames = self.default_settings.get("mar_reset_frames", 5) 

        self.phone_detection_enabled = self.default_settings.get(
            "phone_detection_enabled", True
        )
        self.phone_confidence = self.default_settings.get("phone_confidence", 0.30)
        self.phone_frames = self.default_settings.get(
            "phone_frames", 1
        )  # (Interpretado como SEGUNDOS)

        # --- 5. Configuração do Thread YOLO ---
        self.cam_thread_ref: CameraThread = None
        self.phone_thread = None
        self.yolo_lock = threading.Lock()
        self.last_yolo_boxes = []
        self.phone_detected_time = None

    # --- Loop do Thread YOLO ---
    def _yolo_loop(self):
        logging.info(">>> _yolo_loop (Thread Híbrido) iniciado.")

        if self.stop_event.wait(timeout=3.0):
            return

        while not self.stop_event.is_set():
            start_time_yolo = time.time()

            with self.lock:
                phone_enabled = self.phone_detection_enabled
                current_phone_confidence = self.phone_confidence

            if (
                not phone_enabled
                or self.cam_thread_ref is None
                or self.yolo_cellphone_class_id == -1
            ):
                with self.yolo_lock:
                    self.last_yolo_boxes = []
                    self.phone_detected_time = None 
                logging.info("_yolo_loop: Deteção de celular DESATIVADA. A aguardar...")
                if self.stop_event.wait(timeout=2.0):
                    break
                continue

            try:
                frame = self.cam_thread_ref.get_frame()
                if frame is None:
                    logging.warning(
                        "_yolo_loop: Não obteve frame. A tentar novamente em 2s."
                    )
                    if self.stop_event.wait(timeout=2.0):
                        break
                    continue

                logging.debug("_yolo_loop: A executar inferência (1. Mãos)...")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # --- 1. Inferência MediaPipe Hands ---
                results_hands = self.hands.process(frame_rgb)

                phone_found_this_loop = False
                current_boxes = []

                if results_hands.multi_hand_landmarks:
                    logging.debug(
                        f"_yolo_loop: Mãos(MP) encontradas "
                        f"({len(results_hands.multi_hand_landmarks)}). "
                        "Verificando se há um celular (YOLO)..."
                    )

                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        # --- 2. Calcular a Bounding Box da Mão ---
                        h, w, _ = frame.shape
                        x_min, y_min = w, h
                        x_max, y_max = 0, 0
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            if x < x_min:
                                x_min = x
                            if x > x_max:
                                x_max = x
                            if y < y_min:
                                y_min = y
                            if y > y_max:
                                y_max = y

                        padding = 60
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
                            logging.warning(
                                "_yolo_loop: Recorte da mão resultou em imagem vazia."
                            )
                            continue

                        results_yolo = self.yolo_model(
                            hand_crop,
                            verbose=False,
                            classes=[self.yolo_cellphone_class_id],
                            conf=current_phone_confidence,
                            imgsz=320,
                            augment=False,
                            half=False,
                        )

                        if results_yolo and results_yolo[0].boxes:
                            for box in results_yolo[0].boxes:
                                if int(box.cls) == self.yolo_cellphone_class_id:
                                    phone_found_this_loop = True
                                    box_coords_global = [
                                        int(box.xyxy[0][0] + x_min),
                                        int(box.xyxy[0][1] + y_min),
                                        int(box.xyxy[0][2] + x_min),
                                        int(box.xyxy[0][3] + y_min),
                                    ]
                                    current_boxes.append(box_coords_global)
                                    break
                        if phone_found_this_loop:
                            break

                # --- 5. Atualizar os resultados (thread-safe) ---
                with self.yolo_lock:
                    self.last_yolo_boxes = current_boxes if phone_found_this_loop else []

                    # Lógica de tempo
                    current_time = time.time()
                    if phone_found_this_loop:
                        if self.phone_detected_time is None:
                            # Inicia o cronômetro na primeira detecção
                            self.phone_detected_time = current_time
                            logging.debug("_yolo_loop: Detecção de celular INICIADA.")
                    else:
                        # Se não encontrou, reseta o cronômetro
                        if self.phone_detected_time is not None:
                            logging.debug("_yolo_loop: Detecção de celular INTERROMPIDA.")
                        self.phone_detected_time = None

                logging.debug(
                    f"_yolo_loop: Inferência concluída. "
                    f"Mão/Celular Híbrido: {phone_found_this_loop}. "
                    f"Duração: {time.time() - start_time_yolo:.3f}s"
                )

            except Exception as e:
                logging.error(f"_yolo_loop: Erro na inferência: {e}", exc_info=True)
                with self.yolo_lock:
                    self.phone_detected_time = None
                    self.last_yolo_boxes = []
            
    
            TARGET_YOLO_CYCLE_TIME = 1.0 # Alvo de 1 ciclo por segundo
            
            processing_time = time.time() - start_time_yolo
            wait_time = TARGET_YOLO_CYCLE_TIME - processing_time
            
            logging.debug(
                f"_yolo_loop: Inferência concluída. "
                f"Híbrido: {phone_found_this_loop}. "
                f"Duração: {processing_time:.3f}s. Espera: {max(0, wait_time):.3f}s"
            )

            if wait_time > 0:
                if self.stop_event.wait(timeout=wait_time):
                    break
            else:
                # O loop está lento, não espera
                if self.stop_event.wait(timeout=0.01):
                    break
            # ------------------------------------

        logging.info(">>> _yolo_loop (Thread) terminado.")

    def start_yolo_thread(self, cam_thread_ref: CameraThread):
        self.cam_thread_ref = cam_thread_ref
        self.phone_thread = threading.Thread(
            target=self._yolo_loop, name="PhoneDetectionThread"
        )
        self.phone_thread.daemon = True
        self.phone_thread.start()

    # --- Funções de Cálculo ---
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

    def _finish_calibration(self):
        """
        Função interna para calcular e aplicar o limiar EAR após 
        a recolha de amostras.
        (Assume que self.lock JÁ ESTÁ ADQUIRIDO pela 'process_frame')
        """
        
        if len(self.calibration_samples) < self.CALIBRATION_FRAMES_TARGET / 2:
            logging.warning("Calibração falhou: poucas amostras. Voltando a IDLE.")
            self.calibration_state = "IDLE"
            self.calibration_samples = []
            return

        sorted_samples = sorted(self.calibration_samples)
        percentile_index = int(len(sorted_samples) * 0.20)
        
        valid_samples = sorted_samples[percentile_index:]
        
        if not valid_samples:
            logging.warning("Calibração falhou: não há amostras válidas (talvez só piscadas?).")
            self.calibration_state = "IDLE"
            self.calibration_samples = []
            return
            
        baseline_ear_open = np.mean(valid_samples)
        
        new_threshold = baseline_ear_open * self.ear_calibration_factor

        logging.info(f"CALIBRAÇÃO CONCLUÍDA:")
        logging.info(f"  > EAR Base (Aberto): {baseline_ear_open:.4f}")
        logging.info(f"  > Fator Aplicado: {self.ear_calibration_factor * 100}%")
        
        self.ear_threshold = new_threshold
        self.calibration_state = "DONE"
        self.calibration_samples = [] 
        
        logging.info(f"  > NOVO Limiar EAR (ear_threshold) definido para: {self.ear_threshold:.4f}")


    def process_frame(self, frame, frame_rgb):
        logging.debug("DMSCore(MediaPipe): process_frame (MP Rápido) iniciado.")
        start_time_total = time.time()
        events_list = []
        status_data = {"ear": "-", "mar": "-", "yaw": "-", "pitch": "-", "roll": "-"}
        face_found_this_frame = False

        try:
            results_mp = self.face_mesh.process(frame_rgb) 
        except Exception as e:
            logging.error(f"DMSCore(MediaPipe): Erro .process(): {e}", exc_info=True)
            return frame, events_list, status_data, False # Retorna False em caso de erro

        if results_mp.multi_face_landmarks:
            face_landmarks = results_mp.multi_face_landmarks[0].landmark
            face_found_this_frame = True

            try:
                left_eye_pts = self._get_landmarks_from_result(
                    face_landmarks, MP_LEFT_EYE_IDX
                )
                right_eye_pts = self._get_landmarks_from_result(
                    face_landmarks, MP_RIGHT_EYE_IDX
                )
                mouth_pts = self._get_landmarks_from_result(
                    face_landmarks, MP_MOUTH_IDX
                )
                ear_left = self._eye_aspect_ratio(left_eye_pts)
                ear_right = self._eye_aspect_ratio(right_eye_pts)
                ear = (ear_left + ear_right) / 2.0
                mar = self._mouth_aspect_ratio(mouth_pts)
                status_data["ear"] = f"{ear:.2f}"
                status_data["mar"] = f"{mar:.2f}"

                cv2.drawContours(frame, [cv2.convexHull(left_eye_pts)], -1, (0, 255, 0), 1)
                cv2.drawContours(
                    frame, [cv2.convexHull(right_eye_pts)], -1, (0, 255, 0), 1
                )
                cv2.drawContours(
                    frame, [cv2.convexHull(mouth_pts)], -1, (0, 255, 255), 1
                )

            except Exception as e:
                logging.error(
                    f"DMSCore(MediaPipe): Erro ao processar landmarks: {e}",
                    exc_info=True,
                )
                face_found_this_frame = False

        local_boxes = []
        current_phone_detected_time = None
        with self.lock:
            phone_enabled_locked = self.phone_detection_enabled

        if phone_enabled_locked:
            with self.yolo_lock:
                local_boxes = self.last_yolo_boxes
                current_phone_detected_time = self.phone_detected_time

            for box_coords in local_boxes:
                x1, y1, x2, y2 = map(int, box_coords)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(
                    frame,
                    "Celular (na Mao)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )

        logging.debug("DMSCore(MediaPipe): Lock alerta...")
        with self.lock:
            logging.debug("DMSCore(MediaPipe): Lock alerta OK.")

            current_calib_state = self.calibration_state

            if face_found_this_frame:
                
                # --- LÓGICA DE CALIBRAÇÃO ---
                if current_calib_state == "CALIBRATING":
                    self.calibration_samples.append(ear)
                    
                    progresso = len(self.calibration_samples)
                    texto_calib = f"CALIBRANDO... {progresso}/{self.CALIBRATION_FRAMES_TARGET}"
                    cv2.putText(frame, texto_calib, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, "MANTENHA OS OLHOS ABERTOS", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    self.drowsiness_counter = 0
                    self.drowsy_alert_active = False
                    self.drowsiness_reset_counter = 0

                    if len(self.calibration_samples) >= self.CALIBRATION_FRAMES_TARGET:
                        self._finish_calibration() 
                
                elif current_calib_state == "IDLE":
                    cv2.putText(frame, "CALIBRACAO NECESSARIA", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(frame, "Pressione 'Calibrar' na interface", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    
                    self.drowsiness_counter = 0
                    self.drowsy_alert_active = False
                    self.drowsiness_reset_counter = 0

                elif current_calib_state == "DONE":
                    # --- Lógica de Sonolência (EAR) ---
                    if ear < self.ear_threshold: 
                        self.drowsiness_counter += 1
                        self.drowsiness_reset_counter = 0 
                        
                        logging.debug(
                            f"DMSCore(MediaPipe): EAR baixo ({ear:.3f}<{self.ear_threshold:.3f}), "
                            f"cont={self.drowsiness_counter}/{self.ear_frames}"
                        )
                        
                        if (
                            self.drowsiness_counter >= self.ear_frames
                            and not self.drowsy_alert_active
                        ):
                            self.drowsy_alert_active = True
                            events_list.append(
                                {
                                    "type": "SONOLENCIA",
                                    "value": f"EAR: {ear:.2f}",
                                    "timestamp": datetime.now().isoformat() + "Z",
                                }
                            )
                            logging.warning("DMSCore(MediaPipe): EVENTO SONOLENCIA.")
                    
                    else:
                        # Olhos abertos (EAR >= threshold)
                        self.drowsiness_reset_counter += 1
                        
                        logging.debug(
                            f"DMSCore(MediaPipe): EAR OK ({ear:.3f}). "
                            f"Contagem para reset: {self.drowsiness_reset_counter}/{self.ear_reset_frames}"
                        )

                        if self.drowsiness_reset_counter >= self.ear_reset_frames:
                            if self.drowsiness_counter > 0 or self.drowsy_alert_active:
                                logging.debug("DMSCore(MediaPipe): Sonolência RESETADA (Grace period).")
                            
                            self.drowsiness_counter = 0
                            self.drowsy_alert_active = False
                
                # --- Lógica de Bocejo (MAR) com Grace Period ---
                if mar > self.mar_threshold:
                    self.yawn_counter += 1
                    self.yawn_reset_counter = 0 # Reseta contador 'boca fechada'
                    
                    logging.debug(
                        f"DMSCore(MediaPipe): MAR alto ({mar:.3f}>{self.mar_threshold}), "
                        f"cont={self.yawn_counter}/{self.mar_frames}"
                    )
                    if (
                        self.yawn_counter >= self.mar_frames
                        and not self.yawn_alert_active
                    ):
                        self.yawn_alert_active = True
                        events_list.append(
                            {
                                "type": "BOCEJO",
                                "value": f"MAR: {mar:.2f}",
                                "timestamp": datetime.now().isoformat() + "Z",
                            }
                        )
                        logging.warning("DMSCore(MediaPipe): EVENTO BOCEJO.")
                else:
                    # Boca fechada (MAR <= threshold)
                    self.yawn_reset_counter += 1
                    
                    logging.debug(
                        f"DMSCore(MediaPipe): MAR OK ({mar:.3f}). "
                        f"Contagem reset bocejo: {self.yawn_reset_counter}/{self.mar_reset_frames}"
                    )
                    
                    # Apenas reseta o alerta se a boca estiver fechada
                    # por 'mar_reset_frames' consecutivos.
                    if self.yawn_reset_counter >= self.mar_reset_frames:
                        if self.yawn_counter > 0 or self.yawn_alert_active:
                             logging.debug("DMSCore(MediaPipe): Bocejo RESETADO (Grace period).")
                        
                        self.yawn_counter = 0
                        self.yawn_alert_active = False
                # --- Fim da Lógica de Bocejo ---

            else:
                # --- Lógica de Nenhuma Face Encontrada ---
                logging.debug("DMSCore(MediaPipe): Nenhuma face encontrada.")
                # Se nenhuma face for encontrada, reinicia os contadores de frames,
                # mas mantém as travas de alerta (drowsy_alert_active) ATIVADAS.
                # O alerta só será resetado quando o motorista provar que está
                # acordado (olhos abertos) ou sem bocejar (boca fechada) 
                # pelo 'reset_frames' período.
                
                self.drowsiness_counter = 0
                self.drowsiness_reset_counter = 0 
                
                self.yawn_counter = 0
                self.yawn_reset_counter = 0 # (Adicionado)

                # Se a calibração estava a decorrer e a face foi perdida,
                # para e volta a IDLE para recomeçar
                if self.calibration_state == "CALIBRATING":
                    logging.warning("Calibração interrompida (face perdida). Voltando a IDLE.")
                    self.calibration_state = "IDLE"
                    self.calibration_samples = []
            
            # --- Lógica de Celular (Distração) ---
            if phone_enabled_locked:
                if current_phone_detected_time is not None:
                    elapsed = time.time() - current_phone_detected_time
                    phone_alert_seconds = self.phone_frames

                    logging.debug(
                        f"DMSCore(YOLO): Celular detectado por {elapsed:.1f}s "
                        f"(Alvo: {phone_alert_seconds}s)"
                    )

                    if elapsed >= phone_alert_seconds and not self.phone_alert_active:
                        self.phone_alert_active = True
                        events_list.append(
                            {
                                "type": "DISTRACAO",
                                "value": "Celular na mao",
                                "timestamp": datetime.now().isoformat() + "Z",
                            }
                        )
                        logging.warning("DMSCore(YOLO/Mao): EVENTO DISTRACAO (CELULAR NA MAO).")
                else:
                    if self.phone_alert_active:
                        logging.debug("DMSCore(YOLO/Mao): Deteção celular reset.")
                    self.phone_alert_active = False

        logging.debug("DMSCore(MediaPipe): Lock alerta libertado.")

        # --- Desenhar Alertas na Tela ---
        if self.drowsy_alert_active:
            cv2.putText(
                frame,
                "ALERTA: SONOLENCIA!",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        if self.yawn_alert_active:
            y_pos = 90 if self.calibration_state != "DONE" else 60
            cv2.putText(
                frame,
                "ALERTA: BOCEJO!",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        if phone_enabled_locked and self.phone_alert_active:
            y_pos = 120 if self.calibration_state != "DONE" else 90
            cv2.putText(
                frame,
                "ALERTA: CELULAR/MAO!",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )

        total_time = time.time() - start_time_total
        logging.debug(f"DMSCore(MediaPipe): process_frame (MP Rápido) {total_time:.4f}s.")
        
        # --- MODIFICAÇÃO PRINCIPAL ---
        # Retorna o status da deteção de rosto
        return frame, events_list, status_data, face_found_this_frame

    def update_settings(self, settings):
        logging.debug(f"DMSCore(MediaPipe): Tentando atualizar conf: {settings}")

        with self.lock:
            try:
                if settings.get("start_calibration"):
                    logging.info("Comando 'start_calibration' recebido. Iniciando calibração...")
                    self.calibration_state = "CALIBRATING"
                    self.calibration_samples = []
                    settings.pop("start_calibration", None)
                
                self.ear_threshold = float(
                    settings.get("ear_threshold", self.ear_threshold)
                )
                self.ear_frames = int(settings.get("ear_frames", self.ear_frames))
    
                self.ear_reset_frames = int(
                    settings.get("ear_reset_frames", self.ear_reset_frames)
                )

                self.ear_calibration_factor = float(
                    settings.get("ear_calibration_factor", self.ear_calibration_factor)
                )

                self.mar_threshold = float(
                    settings.get("mar_threshold", self.mar_threshold)
                )
                self.mar_frames = int(settings.get("mar_frames", self.mar_frames))
                
                self.mar_reset_frames = int(
                    settings.get("mar_reset_frames", self.mar_reset_frames)
                )

                self.phone_detection_enabled = bool(
                    settings.get("phone_detection_enabled", self.phone_detection_enabled)
                )
                self.phone_confidence = float(
                    settings.get("phone_confidence", self.phone_confidence)
                )
                self.phone_frames = int(
                    settings.get("phone_frames", self.phone_frames)
                ) 

                distraction_status = (
                    "ATIVADA" if self.phone_detection_enabled else "DESATIVADA"
                )
                logging.info(
                    f"Conf DMS Core(MediaPipe): EAR<{self.ear_threshold:.4f}({self.ear_frames}f), "
                    f"EAR_Reset>{self.ear_reset_frames}f, "
                    f"EAR_Fator>{self.ear_calibration_factor}, "
                    f"MAR>{self.mar_threshold}({self.mar_frames}f), "
                    f"MAR_Reset>{self.mar_reset_frames}f, " # Log atualizado
                    f"Celular:{distraction_status} "
                    f"[Conf>{self.phone_confidence}({self.phone_frames}s)]"
                )

                if not self.phone_detection_enabled:
                    self.phone_alert_active = False
                    with self.yolo_lock:
                        self.phone_detected_time = None
                        self.last_yolo_boxes = []

                return True
            except (ValueError, TypeError) as e:
                logging.error(f"Erro conf MediaPipe (valor inválido?): {e}")
                return False
            except Exception as e:
                logging.error(f"Erro inesperado conf MediaPipe: {e}", exc_info=True)
                return False

    def get_settings(self):
        logging.debug("DMSCore(MediaPipe): get_settings.")
        with self.lock:
            return {
                "ear_threshold": self.ear_threshold,
                "ear_frames": self.ear_frames,
                "ear_reset_frames": self.ear_reset_frames,
                "ear_calibration_factor": self.ear_calibration_factor, 
                "calibration_state": self.calibration_state,       
                "mar_threshold": self.mar_threshold,
                "mar_frames": self.mar_frames,
                "mar_reset_frames": self.mar_reset_frames, # Adicionado
                "phone_detection_enabled": self.phone_detection_enabled,
                "phone_confidence": self.phone_confidence,
                "phone_frames": self.phone_frames,
            }