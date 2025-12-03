import cv2
import mediapipe as mp
import numpy as np
import time
import os
import logging
from dms_base import BaseMonitor
from score_manager import ScoreManager


class MediaPipeMonitor(BaseMonitor):
    def __init__(self, frame_size, stop_event, default_settings=None):
        super().__init__(frame_size, stop_event, default_settings)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.LIPS = [61, 291, 39, 181, 0, 17]

        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])

        # Adicionamos 'fps' às métricas iniciais
        self.latest_metrics = {
            "ear": 1.0, "mar": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "fps": 0
        }

        self.score_manager = ScoreManager()
        self.last_process_time = time.time()
        self.process_width = 320
        self.prev_frame_time = 0
        self.current_fps = 0

        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_YELLOW = (0, 255, 255)

    def process_frame(self, frame: np.ndarray, frame_rgb: np.ndarray):
        img_h, img_w, _ = frame.shape
        current_time = time.time()

        if self.prev_frame_time > 0:
            fps_calc = 1.0 / (current_time - self.prev_frame_time)
            self.current_fps = 0.9 * self.current_fps + 0.1 * fps_calc
        self.prev_frame_time = current_time

        dt = current_time - self.last_process_time
        self.last_process_time = current_time
        if dt > 1.0: dt = 0.03
        if dt <= 0: dt = 0.001

        frame_small = cv2.resize(frame_rgb, (self.process_width, int(self.process_width * img_h / img_w)))
        results = self.face_mesh.process(frame_small)

        if not results.multi_face_landmarks:
            score, events = self.score_manager.update(self._get_empty_metrics(), dt)
            cv2.putText(frame, "ROSTO NAO DETECTADO", (50, img_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.COLOR_RED, 2)
            self._draw_hud(frame, score, 0.0, 0.0)

            # Atualiza métricas com FPS mesmo sem rosto
            self.latest_metrics["fps"] = int(self.current_fps)
            return frame, events, self.latest_metrics, False

        face_landmarks = results.multi_face_landmarks[0]
        landmarks_np = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])

        left_ear = self._calculate_ear(landmarks_np, self.LEFT_EYE)
        right_ear = self._calculate_ear(landmarks_np, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = self._calculate_mar(landmarks_np, self.LIPS)
        pitch, yaw, roll = self._calculate_head_pose(face_landmarks, img_w, img_h)

        self.latest_metrics = {
            "ear": round(avg_ear, 3), "mar": round(mar, 3),
            "pitch": round(pitch, 1), "yaw": round(yaw, 1), "roll": round(roll, 1),
            "fps": int(self.current_fps)  # Passamos o FPS aqui para a API ler
        }

        score, events = self.score_manager.update(self.latest_metrics, dt)

        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )

        self._draw_hud(frame, score, avg_ear, mar, pitch, roll)

        return frame, events, self.latest_metrics, True

    def _get_empty_metrics(self):
        return {"ear": 1.0, "mar": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "fps": 0}

    def _draw_hud(self, frame, score, ear, mar, pitch=0, roll=0):
        color_score = self.COLOR_GREEN
        if score > 50: color_score = self.COLOR_YELLOW
        if score > 80: color_score = self.COLOR_RED

        bar_width = 200
        filled_width = int((score / 100.0) * bar_width)
        cv2.rectangle(frame, (10, 10), (10 + bar_width, 40), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 10), (10 + filled_width, 40), color_score, -1)
        cv2.putText(frame, f"FADIGA: {int(score)}%", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"EAR:{ear:.2f} MAR:{mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # REMOVIDO: O FPS desenhado no vídeo. Agora vai para o HTML.

    def update_settings(self, settings: dict) -> bool:
        if self.default_settings: self.default_settings.update(settings)
        if self.score_manager: self.score_manager.update_settings(settings)
        return True

    def get_settings(self) -> dict:
        return self.default_settings

    def _calculate_ear(self, landmarks, indices):
        p1 = landmarks[indices[0]];
        p2 = landmarks[indices[1]]
        p3 = landmarks[indices[2]];
        p4 = landmarks[indices[3]]
        p5 = landmarks[indices[4]];
        p6 = landmarks[indices[5]]
        v1 = np.linalg.norm(p2 - p6);
        v2 = np.linalg.norm(p3 - p5)
        h = np.linalg.norm(p1 - p4)
        return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

    def _calculate_mar(self, landmarks, indices):
        p1 = landmarks[indices[0]];
        p2 = landmarks[indices[1]]
        p3 = landmarks[indices[2]];
        p4 = landmarks[indices[3]]
        p5 = landmarks[indices[4]];
        p6 = landmarks[indices[5]]
        v1 = np.linalg.norm(p3 - p4);
        v2 = np.linalg.norm(p5 - p6)
        h = np.linalg.norm(p1 - p2)
        return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

    def _calculate_head_pose(self, face_landmarks, img_w, img_h):
        image_points = np.array([
            (face_landmarks.landmark[1].x * img_w, face_landmarks.landmark[1].y * img_h),
            (face_landmarks.landmark[152].x * img_w, face_landmarks.landmark[152].y * img_h),
            (face_landmarks.landmark[33].x * img_w, face_landmarks.landmark[33].y * img_h),
            (face_landmarks.landmark[263].x * img_w, face_landmarks.landmark[263].y * img_h),
            (face_landmarks.landmark[61].x * img_w, face_landmarks.landmark[61].y * img_h),
            (face_landmarks.landmark[291].x * img_w, face_landmarks.landmark[291].y * img_h)
        ], dtype="double")
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                                 dtype="double")
        dist_coeffs = np.zeros((4, 1))
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            proj_matrix = np.hstack((rotation_matrix, translation_vector))
            euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
            pitch, yaw, roll = [element.item() for element in euler_angles]
            return np.clip(pitch, -90, 90), np.clip(yaw, -90, 90), np.clip(roll, -90, 90)
        except Exception:
            return 0, 0, 0

    @property
    def mp_drawing_styles(self):
        return mp.solutions.drawing_styles