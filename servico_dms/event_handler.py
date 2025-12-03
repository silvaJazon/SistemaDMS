# Documentação: Gestor de Eventos com Filtro de Pontuação
# Apenas salva eventos se score >= 80.

import threading
import queue
import sqlite3
import os
import time
import cv2
import json
import logging
from datetime import datetime, timedelta


class EventHandler(threading.Thread):
    def __init__(self, queue, stop_event):
        threading.Thread.__init__(self, name="EventHandler")
        self.queue = queue
        self.stop_event = stop_event
        self.db_path = "/app/alerts/dms_alerts.db"
        self.save_path = "/app/alerts"
        self.camera_thread_ref = None

        self._init_db()

    def set_camera_thread(self, cam_thread):
        self.camera_thread_ref = cam_thread

    def _init_db(self):
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "videos"), exist_ok=True)

        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS alerts
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp TEXT,
                          type TEXT,
                          message TEXT,
                          severity INTEGER,
                          score REAL,
                          image_path TEXT,
                          video_path TEXT, 
                          synced INTEGER DEFAULT 0,
                          synced_time TEXT)''')
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"EventHandler: Erro init DB: {e}")

    def run(self):
        logging.info("EventHandler: Iniciado.")
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=1.0)
                self._process_event(item)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"EventHandler: Erro no loop: {e}")

    def _process_event(self, item):
        event_data = item.get("event_data")
        frame = item.get("frame")

        if not event_data: return

        # --- FILTRO DE IMPORTÂNCIA ---
        score = event_data.get("score", 0)
        if score < 80: return
        # -----------------------------

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename_base = f"{int(time.time())}_{event_data['type']}"

        # 1. Imagem
        img_filename = f"{filename_base}.jpg"
        img_full_path = os.path.join(self.save_path, "images", img_filename)

        if frame is not None:
            try:
                cv2.imwrite(img_full_path, frame)
            except:
                img_filename = ""
        else:
            img_filename = ""

        # 2. Vídeo (MUDANÇA PARA MP4 H.264)
        vid_filename = ""
        severity = event_data.get("severity", 0)

        if severity >= 2 and self.camera_thread_ref:
            logging.info(f"Gravando vídeo (H.264) para: {event_data['type']} (Score: {score})")
            # Volta para .mp4
            vid_filename = f"{filename_base}.mp4"
            vid_full_path = os.path.join(self.save_path, "videos", vid_filename)

            threading.Thread(target=self._save_video_clip,
                             args=(vid_full_path,)).start()

        # 3. DB
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                "INSERT INTO alerts (timestamp, type, message, severity, score, image_path, video_path, synced) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (timestamp, event_data['type'], event_data['message'], severity,
                 score, img_filename, vid_filename))
            conn.commit()
            conn.close()
            logging.info(f"Evento Crítico salvo DB: {event_data['type']} (Score: {int(score)})")
        except Exception as e:
            logging.error(f"Erro SQLite: {e}")

    def _save_video_clip(self, filepath):
        if not self.camera_thread_ref: return
        frames = self.camera_thread_ref.get_recent_frames()
        if not frames: return

        try:
            height, width, _ = frames[0].shape

            # --- MUDANÇA: 'avc1' (H.264) é o padrão ouro e costuma ser limpo nos logs ---
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            fps = 20.0

            out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            for f in frames:
                out.write(f)
            out.release()
            logging.info(f"Vídeo MP4 salvo: {os.path.basename(filepath)}")
        except Exception as e:
            logging.error(f"Erro ao gravar vídeo: {e}")

    # --- Métodos MQTT ---
    def get_pending_alerts(self, limit=10):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM alerts WHERE synced=0 ORDER BY id ASC LIMIT ?", (limit,))
            rows = c.fetchall()
            conn.close()
            alerts = []
            for row in rows:
                al = dict(row)
                alerts.append({
                    "id": al["id"],
                    "timestamp": al["timestamp"],
                    "event_type": al["type"],
                    "details": {
                        "message": al["message"], "score": al["score"],
                        "severity": al["severity"], "image": al["image_path"],
                        "video": al["video_path"]
                    }
                })
            return alerts
        except:
            return []

    def mark_alert_as_sent(self, db_id, sent_time):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("UPDATE alerts SET synced=1, synced_time=? WHERE id=?", (sent_time, db_id))
            conn.commit()
            conn.close()
        except:
            pass

    def cleanup_sent_alerts(self, days_to_keep=7):
        if days_to_keep <= 0: return 0, 0
        deleted = 0;
        failed = 0
        cutoff = (datetime.now() - timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT id, image_path, video_path FROM alerts WHERE synced=1 AND timestamp < ?", (cutoff,))
            rows = c.fetchall()
            for row in rows:
                try:
                    if row["image_path"]:
                        p = os.path.join(self.save_path, "images", row["image_path"])
                        if os.path.exists(p): os.remove(p)
                    if row["video_path"]:
                        p = os.path.join(self.save_path, "videos", row["video_path"])
                        if os.path.exists(p): os.remove(p)
                    c.execute("DELETE FROM alerts WHERE id=?", (row["id"],))
                    deleted += 1
                except:
                    failed += 1
            conn.commit();
            conn.close()
        except:
            pass
        return deleted, failed

    def get_alerts(self, limit=50):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
            rows = c.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except:
            return []