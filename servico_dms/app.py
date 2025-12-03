# Documentação: Aplicação Principal Flask (API Completa com Status MQTT e FPS)

import cv2
import time
import os
import numpy as np
import threading
import logging
from flask import Flask, Response, render_template, jsonify, request, send_from_directory
import queue
import json
import signal

from camera_thread import CameraThread
from dms_base import BaseMonitor
from dms_mediapipe import MediaPipeMonitor
from event_handler import EventHandler
from mqtt_uploader import MQTTUploader

try:
    from waitress import serve
    from waitress.channel import ClientDisconnected

    HAS_WAITRESS = True
except ImportError:
    HAS_WAITRESS = False
    ClientDisconnected = Exception

cv2.setUseOptimized(True)

# --- Config Logging ---
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - DMS - %(levelname)s - %(message)s")

CONFIG_DIR = "/app/config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "settings.json")
PROFILES_FILE = os.path.join(CONFIG_DIR, "profiles.json")


# --- Gestor de Perfis ---
class ProfileManager:
    def __init__(self):
        self.file = PROFILES_FILE
        if not os.path.exists(self.file): self._save({})

    def _load(self):
        try:
            with open(self.file, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save(self, data):
        try:
            with open(self.file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logging.error(f"Erro salvar perfis: {e}")

    def get_profiles(self):
        return self._load()

    def save_profile(self, name, settings):
        data = self._load()
        data[name] = settings
        self._save(data)

    def delete_profile(self, name):
        data = self._load()
        if name in data: del data[name]; self._save(data)


profile_manager = ProfileManager()


def load_config():
    if not os.path.exists(CONFIG_FILE): return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except:
        return {}


def save_config(settings_dict):
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(settings_dict, f, indent=4)
    except Exception as e:
        logging.error(f"Erro salvar config: {e}")


config_from_file = load_config()

# --- Config Globais ---
VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "0")
FRAME_WIDTH_DISPLAY = 640
FRAME_HEIGHT_DISPLAY = 480
JPEG_QUALITY = 60
TARGET_FPS = 30
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
EVENT_QUEUE_MAX_SIZE = 100
INITIAL_ROTATION = int(os.environ.get("ROTATE_FRAME", config_from_file.get("rotation", "0")))
DETECTION_BACKEND = "MEDIAPIPE"
ENABLE_VIDEO_STREAM = os.environ.get("ENABLE_VIDEO_STREAM", "0").lower() in ("true", "1", "t", "y")


# Auto Brightness
class AutoBrightnessManager:
    def __init__(self, cam_thread_ref):
        self.cam_thread = cam_thread_ref
        self.enabled = False
        self.consecutive_failures = 0
        self.FAILURE_THRESHOLD = 50
        self.BRIGHTNESS_STEP = 5.0
        self.BRIGHTNESS_MIN = 0.0
        self.BRIGHTNESS_MAX = 60.0
        self.current_brightness = 17.0
        self.search_direction = +1
        if self.cam_thread and not self.cam_thread.is_rtsp:
            try:
                self.current_brightness = self.cam_thread.get_brightness()
            except:
                pass

    def set_enabled(self, enabled):
        if self.cam_thread.is_rtsp: self.enabled = False; return
        self.enabled = enabled
        if enabled: self.current_brightness = self.cam_thread.get_brightness()

    def is_enabled(self):
        return self.enabled

    def update_status(self, face_found):
        if not self.enabled: return
        if face_found: self.consecutive_failures = 0; return
        self.consecutive_failures += 1
        if self.consecutive_failures < self.FAILURE_THRESHOLD: return
        self.consecutive_failures = 0
        self.current_brightness += (self.search_direction * self.BRIGHTNESS_STEP)
        if self.current_brightness >= self.BRIGHTNESS_MAX:
            self.current_brightness = self.BRIGHTNESS_MAX;
            self.search_direction = -1
        elif self.current_brightness <= self.BRIGHTNESS_MIN:
            self.current_brightness = self.BRIGHTNESS_MIN;
            self.search_direction = +1
        try:
            self.cam_thread.update_brightness(self.current_brightness)
        except:
            pass


# --- Globais ---
output_frame_display = None
output_frame_lock = threading.Lock()
status_data_global = {}
status_data_lock = threading.Lock()
stop_event = threading.Event()
cam_thread = None
dms_monitor = None
brightness_manager = None
mqtt_thread = None
event_handler = None
event_queue = None

app = Flask(__name__)


def create_placeholder_frame(text="Aguardando..."):
    frame = np.zeros((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY, 3), dtype=np.uint8)
    cv2.putText(frame, text, (30, FRAME_HEIGHT_DISPLAY // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def detection_loop(cam, monitor, queue_ref):
    global output_frame_display, status_data_global
    while not stop_event.is_set():
        start = time.time()
        frame = cam.get_frame()
        if frame is None: time.sleep(0.1); continue

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed, events, status, face_found = monitor.process_frame(frame.copy(), rgb)
            if brightness_manager: brightness_manager.update_status(face_found)

            with output_frame_lock:
                output_frame_display = processed.copy()
            with status_data_lock:
                status_data_global = status.copy()

            if events:
                for ev in events:
                    try:
                        queue_ref.put({"event_data": ev, "frame": frame.copy()}, block=False)
                    except:
                        pass
        except:
            pass

        elapsed = time.time() - start
        wait = TARGET_FRAME_TIME - elapsed
        if wait > 0: time.sleep(wait)


def generate_video_stream():
    global output_frame_display
    ph = create_placeholder_frame()
    while not stop_event.is_set():
        with output_frame_lock:
            frame = output_frame_display.copy() if output_frame_display is not None else ph.copy()
        try:
            ret, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ret: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        except (GeneratorExit, ClientDisconnected):
            break
        except Exception:
            break
        time.sleep(1 / 15)


@app.route("/")
def index():
    return render_template("index.html", source_desc="Cam", width=FRAME_WIDTH_DISPLAY, height=FRAME_HEIGHT_DISPLAY,
                           active_backend=DETECTION_BACKEND, video_stream_enabled=ENABLE_VIDEO_STREAM)


@app.route("/alerts")
def alerts_page(): return render_template("alerts.html")


@app.route("/video_feed")
def video_feed():
    if not ENABLE_VIDEO_STREAM:
        ret, jpg = cv2.imencode(".jpg", create_placeholder_frame("Stream OFF"), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        return Response(jpg.tobytes(), mimetype="image/jpeg")
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


# --- API ---
@app.route("/api/profiles", methods=["GET"])
def get_profiles(): return jsonify(profile_manager.get_profiles())


@app.route("/api/profiles/<name>", methods=["POST"])
def save_profile_route(name):
    profile_manager.save_profile(name, request.json)
    return jsonify({"success": True})


@app.route("/api/profiles/<name>", methods=["DELETE"])
def delete_profile_route(name):
    profile_manager.delete_profile(name)
    return jsonify({"success": True})


@app.route("/api/alerts/<alert_id>", methods=["DELETE"])
def delete_alert(alert_id):
    if request.headers.get("X-Admin-Password") != "Admin@1999": return jsonify({"error": "Senha incorreta"}), 403
    success = event_handler.delete_all_alerts() if alert_id == "all" else event_handler.delete_alert(alert_id)
    return jsonify({"success": True}) if success else (jsonify({"error": "Falha"}), 500)


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "GET":
        s = dms_monitor.get_settings()
        s["brightness"] = cam_thread.get_brightness()
        s["rotation"] = cam_thread.get_rotation()
        s["auto_brightness"] = brightness_manager.is_enabled()

        # MQTT Info
        with mqtt_thread.config_lock:
            for k in ["mqtt_enabled", "mqtt_broker", "mqtt_port", "mqtt_device_id", "mqtt_fleet_id", "mqtt_username",
                      "mqtt_password", "mqtt_retention_days"]:
                s[k] = mqtt_thread.config.get(k)

        # --- NOVO: Status Conexão MQTT ---
        s["mqtt_connected"] = mqtt_thread.is_connected()
        # ---------------------------------

        with status_data_lock:
            s["status"] = status_data_global.copy()
        s["queue_depth"] = event_queue.qsize()
        return jsonify(s)
    elif request.method == "POST":
        ns = request.json
        if "auto_brightness" in ns: brightness_manager.set_enabled(ns["auto_brightness"])
        dms_monitor.update_settings(ns)
        if "brightness" in ns: cam_thread.update_brightness(ns["brightness"])
        if "rotation" in ns: cam_thread.update_rotation(ns["rotation"])

        mqtt_keys = ["mqtt_enabled", "mqtt_broker", "mqtt_port", "mqtt_device_id", "mqtt_fleet_id", "mqtt_username",
                     "mqtt_password", "mqtt_retention_days"]
        mq_upd = {k: ns[k] for k in mqtt_keys if k in ns}
        if mq_upd: mqtt_thread.update_config(mq_upd)

        final = dms_monitor.get_settings()
        final["brightness"] = cam_thread.get_brightness()
        final["rotation"] = cam_thread.get_rotation()
        final["auto_brightness"] = brightness_manager.is_enabled()
        with mqtt_thread.config_lock:
            for k in mqtt_keys: final[k] = mqtt_thread.config.get(k)
        save_config(final)
        return jsonify({"success": True})


@app.route("/api/alerts", methods=["GET"])
def api_alerts_list(): return jsonify(event_handler.get_alerts())


@app.route("/alerts/images/<path:p>")
def serve_img(p): return send_from_directory(os.path.join(event_handler.save_path, "images"), p)


@app.route("/alerts/videos/<path:p>")
def serve_vid(p):
    m = 'video/webm' if p.endswith('.webm') else 'video/mp4'
    return send_from_directory(os.path.join(event_handler.save_path, "videos"), p, mimetype=m)


# --- Main ---
def shutdown(s, f): stop_event.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown);
    signal.signal(signal.SIGTERM, shutdown)

    event_queue = queue.Queue(maxsize=EVENT_QUEUE_MAX_SIZE)
    event_handler = EventHandler(event_queue, stop_event);
    event_handler.start()

    mq_cfg = {
        "mqtt_enabled": config_from_file.get("mqtt_enabled", False),
        "mqtt_broker": config_from_file.get("mqtt_broker", "broker.hivemq.com"),
        "mqtt_port": config_from_file.get("mqtt_port", 1883),
        "mqtt_device_id": config_from_file.get("mqtt_device_id", "dms_default"),
        "mqtt_fleet_id": config_from_file.get("mqtt_fleet_id", "default_fleet"),
        "mqtt_username": config_from_file.get("mqtt_username", ""),
        "mqtt_password": config_from_file.get("mqtt_password", ""),
        "mqtt_retention_days": config_from_file.get("mqtt_retention_days", 10),
    }
    mqtt_thread = MQTTUploader(event_handler, stop_event, mq_cfg);
    mqtt_thread.start()

    dms_monitor = MediaPipeMonitor((FRAME_HEIGHT_DISPLAY, FRAME_WIDTH_DISPLAY), stop_event, {})
    cam_thread = CameraThread(VIDEO_SOURCE, FRAME_WIDTH_DISPLAY, FRAME_HEIGHT_DISPLAY, INITIAL_ROTATION, stop_event)
    event_handler.set_camera_thread(cam_thread)
    cam_thread.start()

    t0 = time.time()
    while cam_thread.get_frame() is None:
        if time.time() - t0 > 15: break
        time.sleep(0.1)

    brightness_manager = AutoBrightnessManager(cam_thread)
    if config_from_file.get("auto_brightness"): brightness_manager.set_enabled(True)

    detection_thread = threading.Thread(target=detection_loop, args=(cam_thread, dms_monitor, event_queue))
    detection_thread.start()

    if HAS_WAITRESS:
        serve(app, host="0.0.0.0", port=5000, threads=8)
    else:
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)

    stop_event.set()