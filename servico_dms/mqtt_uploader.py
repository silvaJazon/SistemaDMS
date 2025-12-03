# Documentação: Gestor de Upload MQTT (Reação Instantânea)

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
        self.wake_event = threading.Event()  # <--- NOVO: Para acordar a thread
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
        """Atualiza config e força reconexão imediata."""
        logging.info("MQTT Uploader: Configuração recebida.")
        should_wake = False

        with self.config_lock:
            # Verifica se algo crítico mudou
            critical_keys = ["mqtt_broker", "mqtt_port", "mqtt_username", "mqtt_password", "mqtt_device_id",
                             "mqtt_enabled"]
            for key in critical_keys:
                if key in new_settings and self.config.get(key) != new_settings[key]:
                    should_wake = True
                    break
            self.config.update(new_settings)

        if should_wake:
            logging.info("MQTT Uploader: Mudança crítica detetada. A reiniciar conexão...")
            if self.client:
                # Desconecta logo o antigo para limpar o estado
                try:
                    self.client.loop_stop()
                    self.client.disconnect()
                except:
                    pass
                self.client = None

            # ACORDA A THREAD IMEDIATAMENTE
            self.wake_event.set()

    def is_connected(self):
        return self.client is not None and self.client.is_connected()

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
            broker = self.config.get('mqtt_broker', 'desconhecido')
            logging.info(f"MQTT Uploader: SUCESSO - Conectado a {broker}")
        else:
            logging.error(f"MQTT Uploader: Falha conexão (RC={rc})")

    def on_disconnect(self, client, userdata, rc):
        if rc != 0: logging.warning(f"MQTT Uploader: Desconectado (RC={rc})")

    def on_publish(self, client, userdata, mid):
        with self.pending_publish_lock:
            if mid in self.pending_publish:
                db_id = self.pending_publish.pop(mid)
                try:
                    self.event_handler.mark_alert_as_sent(db_id, datetime.now().isoformat())
                except:
                    pass

    # --- Conexão ---
    def _connect_client(self):
        # Se já existe e está conectado, ignora
        if self.client and self.client.is_connected(): return

        try:
            with self.config_lock:
                broker = self.config.get("mqtt_broker", "broker.hivemq.com")
                try:
                    port = int(self.config.get("mqtt_port", 1883))
                except:
                    port = 1883
                device_id = self.config.get("mqtt_device_id", "dms_default")
                username = self.config.get("mqtt_username")
                password = self.config.get("mqtt_password")

            # Cria novo cliente limpo
            client_id = f"dms-{device_id}-{int(time.time())}"
            self.client = mqtt.Client(client_id=client_id)
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_publish = self.on_publish

            if username: self.client.username_pw_set(username, password)
            if port == 8883: self.client.tls_set()

            logging.info(f"MQTT Uploader: A conectar a {broker}:{port}...")
            self.client.connect_async(broker, port, 60)
            self.client.loop_start()

        except Exception as e:
            logging.error(f"MQTT Uploader: Erro ao iniciar cliente: {e}")
            self.client = None

    def _process_pending_alerts(self):
        if not self.is_connected(): return
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
                    with self.pending_publish_lock: self.pending_publish[info.mid] = alert.get("id")
        except Exception as e:
            logging.error(f"MQTT Erro envio: {e}")

    def _process_cleanup(self):
        try:
            with self.config_lock:
                val = self.config.get("mqtt_retention_days", 10)
                try:
                    days = int(val) if val is not None else 10
                except:
                    days = 10
            if days > 0: self.event_handler.cleanup_sent_alerts(days)
        except:
            pass

    def run(self):
        # Pequena pausa inicial para o sistema arrancar
        self.stop_event.wait(5.0)

        while not self.stop_event.is_set():
            # 1. Verifica se foi "acordado" por uma mudança de config
            if self.wake_event.is_set():
                logging.info("MQTT Uploader: Acordado! A aplicar novas configs...")
                self.wake_event.clear()  # Reseta o alarme
                # Continua imediatamente para tentar conectar

            try:
                with self.config_lock:
                    enabled = self.config.get("mqtt_enabled", False)

                if not enabled:
                    if self.client:
                        self.client.loop_stop();
                        self.client.disconnect();
                        self.client = None
                    # Se desativado, dorme até ser acordado ou passar 10s
                    self.wake_event.wait(10.0)
                    continue

                # Verifica net e conecta
                if self._check_internet():
                    if not self.client: self._connect_client()

                    if self.is_connected():
                        self._process_pending_alerts()
                        self._process_cleanup()
                else:
                    logging.debug("MQTT: Sem internet.")

            except Exception as e:
                logging.error(f"MQTT Loop erro: {e}")

            # 2. Dorme, mas fica atento ao botão "Salvar" (wake_event)
            # O wait retorna True se for acordado pelo evento, False se for timeout
            self.wake_event.wait(5.0)

        if self.client: self.client.loop_stop(); self.client.disconnect()