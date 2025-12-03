import time
import logging


class ScoreManager:
    def __init__(self):
        # --- Configuração de Limiares ---
        self.EAR_THRESH = 0.25
        self.EAR_TIME_THRESH = 0.2
        self.MAR_THRESH = 0.50
        self.MAR_TIME_THRESH = 1.0
        self.PITCH_DOWN_THRESH = -20
        self.HEAD_TIME_THRESH = 2.0
        self.ROLL_THRESH = 20

        # --- Estado Interno ---
        self.t_eyes_closed = 0.0
        self.t_mouth_open = 0.0
        self.t_head_bad_pose = 0.0

        # --- Pontuação ---
        self.sleep_score = 0.0
        self.score_decay_rate = 10.0
        self.SCORE_EYES = 40.0
        self.SCORE_HEAD = 30.0
        self.SCORE_YAWN = 10.0

        # --- CONTROLO DE SPAM (COOLDOWNS) ---
        self.last_alert_time = 0
        self.last_event_times = {}
        self.ALERT_COOLDOWN = 10.0

    def update_settings(self, settings: dict):
        FPS_ASSUMED = 30.0
        if "ear_threshold" in settings: self.EAR_THRESH = float(settings["ear_threshold"])
        if "ear_frames" in settings:
            val = float(settings["ear_frames"])
            self.EAR_TIME_THRESH = val / FPS_ASSUMED if val > 0 else 0.1
        if "mar_threshold" in settings: self.MAR_THRESH = float(settings["mar_threshold"])
        if "mar_frames" in settings:
            val = float(settings["mar_frames"])
            self.MAR_TIME_THRESH = val / FPS_ASSUMED if val > 0 else 0.1

    def update(self, metrics, dt):
        events = []
        current_time = time.time()

        ear = metrics.get("ear", 1.0)
        mar = metrics.get("mar", 0.0)
        pitch = metrics.get("pitch", 0.0)
        roll = metrics.get("roll", 0.0)

        # 1. Olhos
        if ear < self.EAR_THRESH:
            self.t_eyes_closed += dt
            if self.t_eyes_closed >= self.EAR_TIME_THRESH:
                self.sleep_score += self.SCORE_EYES * dt
                if self._check_cooldown("microsleep", current_time, 10.0):
                    if self.t_eyes_closed < (self.EAR_TIME_THRESH + 0.5):
                        # ADICIONADO: score
                        events.append({
                            "type": "MICROSLEEP_START",
                            "message": "Micro-sono detectado",
                            "severity": 2,
                            "score": self.sleep_score
                        })
        else:
            self.t_eyes_closed = 0.0

        # 2. Boca
        if mar > self.MAR_THRESH:
            self.t_mouth_open += dt
            if self.t_mouth_open >= self.MAR_TIME_THRESH:
                self.sleep_score += self.SCORE_YAWN * dt
                if self._check_cooldown("yawn", current_time, 15.0):
                    # ADICIONADO: score
                    events.append({
                        "type": "YAWN",
                        "message": "Bocejo detectado",
                        "severity": 1,
                        "score": self.sleep_score
                    })
        else:
            self.t_mouth_open = 0.0

        # 3. Cabeça
        bad_pose = (pitch < self.PITCH_DOWN_THRESH) or (abs(roll) > self.ROLL_THRESH)
        if bad_pose:
            self.t_head_bad_pose += dt
            if self.t_head_bad_pose >= self.HEAD_TIME_THRESH:
                self.sleep_score += self.SCORE_HEAD * dt
                if self._check_cooldown("head", current_time, 10.0):
                    # ADICIONADO: score
                    events.append({
                        "type": "HEAD_DROP",
                        "message": "Posição de cabeça perigosa",
                        "severity": 2,
                        "score": self.sleep_score
                    })
        else:
            self.t_head_bad_pose = 0.0

        # 4. Decaimento
        no_fatigue_signs = (self.t_eyes_closed < self.EAR_TIME_THRESH) and \
                           (self.t_mouth_open < self.MAR_TIME_THRESH) and \
                           (self.t_head_bad_pose < self.HEAD_TIME_THRESH)

        if no_fatigue_signs:
            self.sleep_score -= self.score_decay_rate * dt

        self.sleep_score = max(0.0, min(100.0, self.sleep_score))

        # 5. Alertas de Nível (Estes já tinham score, mantemos)
        if (current_time - self.last_alert_time) > self.ALERT_COOLDOWN:
            if self.sleep_score >= 80:
                events.append(
                    {"type": "DANGER", "message": "PERIGO: FADIGA EXTREMA", "severity": 3, "score": self.sleep_score})
                self.last_alert_time = current_time
            elif self.sleep_score >= 50:
                events.append({"type": "WARNING", "message": "Atenção: Sinais de fadiga", "severity": 2,
                               "score": self.sleep_score})
                self.last_alert_time = current_time

        return self.sleep_score, events

    def _check_cooldown(self, key, current_time, duration):
        last = self.last_event_times.get(key, 0)
        if (current_time - last) > duration:
            self.last_event_times[key] = current_time
            return True
        return False