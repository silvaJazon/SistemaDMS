# Documentação: Gestor de Upload MQTT (Versão Blindada contra erros de Config)

import threading
import time
import json
import logging
import socket
from datetime import datetime
import paho.mqtt.client as mqtt
from event_handler import EventHandler


class MQTTUploader(threading.Thread):
    def __init__(self, event_handler_ref: EventHandler, stop_event: threading.Event, initial_config: dict):
        threading.Thread.__init__(self, name="MQTTUploaderThread")
        self.daemon = True
        self.stop_event = stop_event
        self.event_handler = event_handler_ref
        self.config = initial_config
        self.config_lock = threading.Lock()
        self.client = None
        self.pending_publish = {}
        self.pending_publish_lock = threading.Lock()
        self.last_internet_check = 0
        self.has_internet = False
        logging.info("MQTT Uploader inicializado.")

    def update_config(self, new_settings: dict):
        logging.info("MQTT Uploader: Recebendo atualização de configuração.")
        force_reconnect = False
        with self.config_lock:
            # Chaves críticas que exigem reconexão
            critical_keys = ["mqtt_broker", "mqtt_port", "mqtt_username", "mqtt_password", "mqtt_device_id",
                             "mqtt_enabled"]
            for key in critical_keys:
                if key in new_settings and self.config.get(key) != new_settings[key]:
                    force_reconnect = True
                    break
            self.config.update(new_settings)

        if force_reconnect and self.client:
            logging.info("MQTT Uploader: Configuração mudou. A reconectar...")
            if self.client.is_connected(): self.client.disconnect()

    def _check_internet(self, force=False) -> bool:
        now = time.time()
        if not force and (now - self.last_internet_check < 60): return self.has_internet
        self.last_internet_check = now
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            self.has_internet = True
            return True
        except:
            self.has_internet = False
            return False

    # --- Callbacks ---
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            # CORREÇÃO: Usa .get() para evitar KeyError se a config falhar
            broker = self.config.get('mqtt_broker', 'desconhecido')
            logging.info(f"MQTT Uploader: Conectado ao broker {broker}.")
        else:
            logging.error(f"MQTT Uploader: Falha ao conectar, código: {rc}")

    def on_disconnect(self, client, userdata, rc):
        if rc != 0: logging.warning(f"MQTT Uploader: Desconectado inesperadamente (rc: {rc}).")

    def on_publish(self, client, userdata, mid):
        with self.pending_publish_lock:
            if mid not in self.pending_publish: return
            alert_db_id = self.pending_publish.pop(mid)
        try:
            self.event_handler.mark_alert_as_sent(alert_db_id, datetime.now().isoformat())
            logging.debug(f"MQTT ACK recebido para ID {alert_db_id}")
        except:
            pass

    # --- Lógica ---
    def _connect_client(self):
        if self.client and self.client.is_connected(): return
        try:
            with self.config_lock:
                broker = self.config.get("mqtt_broker", "broker.hivemq.com")
                # CORREÇÃO: Garante que porta é int
                try:
                    port = int(self.config.get("mqtt_port", 1883))
                except:
                    port = 1883

                device_id = self.config.get("mqtt_device_id", "dms_default")
                username = self.config.get("mqtt_username")
                password = self.config.get("mqtt_password")

            client_id = f"dms-{device_id}-{int(time.time())}"
            self.client = mqtt.Client(client_id=client_id)
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_publish = self.on_publish

            if username: self.client.username_pw_set(username, password)
            if port == 8883: self.client.tls_set()

            self.client.connect_async(broker, port, 60)
            self.client.loop_start()
            logging.info(f"MQTT Uploader: A conectar a {broker}:{port}...")
        except Exception as e:
            logging.error(f"MQTT Uploader: Erro conexão: {e}")
            self.client = None

    def _process_pending_alerts(self):
        if not self.client or not self.client.is_connected(): return
        try:
            pending = self.event_handler.get_pending_alerts(limit=5)
            if not pending: return

            with self.config_lock:
                fleet = self.config.get("mqtt_fleet_id", "default")
                device = self.config.get("mqtt_device_id", "dms")

            topic = f"dms/alerts/{fleet}/{device}"

            for alert in pending:
                payload = json.dumps({
                    "device_id": device, "fleet_id": fleet,
                    "timestamp": alert.get("timestamp"), "type": alert.get("event_type"),
                    "details": alert.get("details"), "id": alert.get("id")
                })
                info = self.client.publish(topic, payload, qos=1)
                if info.rc == mqtt.MQTT_ERR_SUCCESS:
                    with self.pending_publish_lock:
                        self.pending_publish[info.mid] = alert.get("id")
                else:
                    break
        except Exception as e:
            logging.error(f"MQTT Erro envio: {e}")

    def _process_cleanup(self):
        try:
            with self.config_lock:
                # CORREÇÃO: Trata 'None' ou strings inválidas
                val = self.config.get("mqtt_retention_days", 10)
                try:
                    days = int(val) if val is not None else 10
                except:
                    days = 10

            if days > 0:
                self.event_handler.cleanup_sent_alerts(days)
        except Exception as e:
            logging.error(f"MQTT Erro limpeza: {e}")

    def run(self):
        self.stop_event.wait(5.0)
        while not self.stop_event.is_set():
            try:
                with self.config_lock:
                    enabled = self.config.get("mqtt_enabled", False)

                if not enabled:
                    if self.client:
                        self.client.loop_stop();
                        self.client.disconnect();
                        self.client = None
                    self.stop_event.wait(10);
                    continue

                if self._check_internet():
                    if not self.client: self._connect_client()
                    if self.client and self.client.is_connected():
                        self._process_pending_alerts()
                        self._process_cleanup()
                else:
                    logging.debug("MQTT: Sem internet.")
            except Exception as e:
                logging.error(f"MQTT Loop erro: {e}")

            self.stop_event.wait(10.0)

        if self.client: self.client.loop_stop(); self.client.disconnect()