import time

class ScoreManager:
    def __init__(self):
        # --- Configuração de Limiares ---
        self.EAR_THRESH = 0.18        # Olho considerado fechado
        self.EAR_TIME_THRESH = 0.3    # Tempo mín (s) para considerar fechado
        
        self.MAR_THRESH = 0.60        # Boca considerada aberta (Bocejo)
        self.MAR_TIME_THRESH = 1.5    # Tempo mín (s) para validar bocejo
        
        self.PITCH_DOWN_THRESH = -20  # Cabeça baixa (graus)
        self.HEAD_TIME_THRESH = 2.0   # Tempo para considerar cabeça caída
        
        self.ROLL_THRESH = 20         # Inclinação lateral (graus)

        # --- Estado Interno ---
        self.t_eyes_closed = 0.0
        self.t_mouth_open = 0.0
        self.t_head_bad_pose = 0.0
        
        # --- Pontuação de Sonolência (0 a 100) ---
        self.sleep_score = 0.0
        self.score_decay_rate = 5.0   # Recuperação por segundo
        
        # Pesos de Penalidade (SÓ PARA FADIGA)
        self.SCORE_EYES = 40.0
        self.SCORE_HEAD = 30.0
        self.SCORE_YAWN = 10.0
        # self.SCORE_PHONE removido pois agora é tratado separadamente

        # --- CONTROLO DE SPAM (COOLDOWNS) ---
        self.last_alert_time = 0      
        self.last_event_times = {}    
        self.ALERT_COOLDOWN = 2.0     

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
        phone_detected = metrics.get("phone_detected", False)

        # --- 1. Distração (Celular) - LÓGICA SEPARADA ---
        # Não afeta o sleep_score, gera alerta imediato de Distração
        if phone_detected:
            # Cooldown de 3 segundos para não spammar "Celular detectado"
            if self._check_cooldown("phone", current_time, 3.0):
                 events.append({
                     "type": "DISTRACTION", 
                     "message": "ATENCAO: Uso de celular detectado!", 
                     "severity": 3
                 })

        # --- 2. Análise de Olhos (Fadiga) ---
        if ear < self.EAR_THRESH:
            self.t_eyes_closed += dt
            if self.t_eyes_closed >= self.EAR_TIME_THRESH:
                self.sleep_score += self.SCORE_EYES * dt
                if self._check_cooldown("microsleep", current_time, 5.0):
                     if self.t_eyes_closed < (self.EAR_TIME_THRESH + 0.2):
                        events.append({"type": "MICROSLEEP_START", "message": "Micro-sono detectado", "severity": 2})
        else:
            self.t_eyes_closed = 0.0

        # --- 3. Análise de Boca (Fadiga) ---
        if mar > self.MAR_THRESH:
            self.t_mouth_open += dt
            if self.t_mouth_open >= self.MAR_TIME_THRESH:
                self.sleep_score += self.SCORE_YAWN * dt
                if self._check_cooldown("yawn", current_time, 10.0):
                    events.append({"type": "YAWN", "message": "Bocejo detectado", "severity": 1})
        else:
            self.t_mouth_open = 0.0

        # --- 4. Análise de Cabeça (Fadiga) ---
        bad_pose = (pitch < self.PITCH_DOWN_THRESH) or (abs(roll) > self.ROLL_THRESH)
        if bad_pose:
            self.t_head_bad_pose += dt
            if self.t_head_bad_pose >= self.HEAD_TIME_THRESH:
                self.sleep_score += self.SCORE_HEAD * dt
                if self._check_cooldown("head", current_time, 5.0):
                    events.append({"type": "HEAD_DROP", "message": "Posição de cabeça perigosa", "severity": 2})
        else:
            self.t_head_bad_pose = 0.0

        # --- 5. Decaimento (Recuperação da Fadiga) ---
        # Se não houver sinais de FADIGA, recupera.
        # (Nota: Pode recuperar fadiga mesmo usando celular, pois são coisas distintas, 
        # mas se quiser que o celular PAUSE a recuperação, adicione "and not phone_detected" abaixo)
        no_fatigue_signs = (self.t_eyes_closed < self.EAR_TIME_THRESH) and \
                           (self.t_mouth_open < self.MAR_TIME_THRESH) and \
                           (self.t_head_bad_pose < self.HEAD_TIME_THRESH)
        
        if no_fatigue_signs:
            self.sleep_score -= self.score_decay_rate * dt

        self.sleep_score = max(0.0, min(100.0, self.sleep_score))

        # --- 6. Gerar Alertas de Fadiga ---
        if (current_time - self.last_alert_time) > self.ALERT_COOLDOWN:
            if self.sleep_score >= 80:
                events.append({"type": "DANGER", "message": "PERIGO: FADIGA EXTREMA", "severity": 3, "score": self.sleep_score})
                self.last_alert_time = current_time
            elif self.sleep_score >= 50:
                 events.append({"type": "WARNING", "message": "Atenção: Sinais de fadiga", "severity": 2, "score": self.sleep_score})
                 self.last_alert_time = current_time

        return self.sleep_score, events

    def _check_cooldown(self, key, current_time, duration):
        last = self.last_event_times.get(key, 0)
        if (current_time - last) > duration:
            self.last_event_times[key] = current_time
            return True
        return False