import time
import logging


class ScoreManager:
    def __init__(self):
        # --- Configuração de Limiares (Padrões Iniciais) ---
        # Estes valores serão atualizados automaticamente pelo site
        self.EAR_THRESH = 0.25  # Olho considerado fechado
        self.EAR_TIME_THRESH = 0.2  # Tempo mín (s) para considerar fechado

        self.MAR_THRESH = 0.50  # Boca considerada aberta (Bocejo)
        self.MAR_TIME_THRESH = 1.0  # Tempo mín (s) para validar bocejo

        self.PITCH_DOWN_THRESH = -20  # Cabeça baixa (graus)
        self.HEAD_TIME_THRESH = 2.0  # Tempo para considerar cabeça caída

        self.ROLL_THRESH = 20  # Inclinação lateral (graus)

        # --- Estado Interno ---
        self.t_eyes_closed = 0.0
        self.t_mouth_open = 0.0
        self.t_head_bad_pose = 0.0

        # --- Pontuação de Sonolência (0 a 100) ---
        self.sleep_score = 0.0
        self.score_decay_rate = 10.0  # Recuperação (% por segundo)

        # Pesos de Penalidade
        self.SCORE_EYES = 40.0
        self.SCORE_HEAD = 30.0
        self.SCORE_YAWN = 10.0

        # --- CONTROLO DE SPAM (COOLDOWNS) ---
        self.last_alert_time = 0
        self.last_event_times = {}
        self.ALERT_COOLDOWN = 2.0

    def update_settings(self, settings: dict):
        """
        Recebe as configurações da API e atualiza os limiares.
        Converte 'frames' (do slider) para 'segundos' (usado na lógica), assumindo 30 FPS.
        """
        FPS_ASSUMED = 30.0

        if "ear_threshold" in settings:
            self.EAR_THRESH = float(settings["ear_threshold"])

        if "ear_frames" in settings:
            # Ex: 6 frames / 30fps = 0.2 segundos
            self.EAR_TIME_THRESH = float(settings["ear_frames"]) / FPS_ASSUMED

        if "mar_threshold" in settings:
            self.MAR_THRESH = float(settings["mar_threshold"])

        if "mar_frames" in settings:
            self.MAR_TIME_THRESH = float(settings["mar_frames"]) / FPS_ASSUMED

    def update(self, metrics, dt):
        """
        Atualiza o estado do condutor.
        """
        events = []
        current_time = time.time()

        # Extrair métricas
        ear = metrics.get("ear", 1.0)
        mar = metrics.get("mar", 0.0)
        pitch = metrics.get("pitch", 0.0)
        roll = metrics.get("roll", 0.0)
        # phone_detected removido (não usado)

        # --- 1. Análise de Olhos (Fadiga) ---
        if ear < self.EAR_THRESH:
            self.t_eyes_closed += dt
            if self.t_eyes_closed >= self.EAR_TIME_THRESH:
                self.sleep_score += self.SCORE_EYES * dt
                if self._check_cooldown("microsleep", current_time, 5.0):
                    if self.t_eyes_closed < (self.EAR_TIME_THRESH + 0.5):
                        events.append({"type": "MICROSLEEP_START", "message": "Micro-sono detectado", "severity": 2})
        else:
            self.t_eyes_closed = 0.0

        # --- 2. Análise de Boca (Fadiga) ---
        if mar > self.MAR_THRESH:
            self.t_mouth_open += dt
            if self.t_mouth_open >= self.MAR_TIME_THRESH:
                self.sleep_score += self.SCORE_YAWN * dt
                if self._check_cooldown("yawn", current_time, 10.0):
                    events.append({"type": "YAWN", "message": "Bocejo detectado", "severity": 1})
        else:
            self.t_mouth_open = 0.0

        # --- 3. Análise de Cabeça (Fadiga) ---
        bad_pose = (pitch < self.PITCH_DOWN_THRESH) or (abs(roll) > self.ROLL_THRESH)
        if bad_pose:
            self.t_head_bad_pose += dt
            if self.t_head_bad_pose >= self.HEAD_TIME_THRESH:
                self.sleep_score += self.SCORE_HEAD * dt
                if self._check_cooldown("head", current_time, 5.0):
                    events.append({"type": "HEAD_DROP", "message": "Posição de cabeça perigosa", "severity": 2})
        else:
            self.t_head_bad_pose = 0.0

        # --- 4. Decaimento (Recuperação) ---
        no_fatigue_signs = (self.t_eyes_closed < self.EAR_TIME_THRESH) and \
                           (self.t_mouth_open < self.MAR_TIME_THRESH) and \
                           (self.t_head_bad_pose < self.HEAD_TIME_THRESH)

        if no_fatigue_signs:
            self.sleep_score -= self.score_decay_rate * dt

        self.sleep_score = max(0.0, min(100.0, self.sleep_score))

        # --- 5. Gerar Alertas de Nível de Fadiga ---
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