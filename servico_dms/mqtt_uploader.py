# servico_dms/mqtt_uploader.py
# Documentação: Gestor de Upload MQTT
# (Atualizado para suportar conexões SSL/TLS na porta 8883)

import threading
import time
import json
import logging
import socket
from datetime import datetime
import paho.mqtt.client as mqtt

# Importamos o EventHandler para ter acesso aos seus métodos de DB
from event_handler import EventHandler

class MQTTUploader(threading.Thread):
    """
    Uma thread que monitoriza a base de dados SQLite em busca de alertas
    pendentes e os envia para um broker MQTT.
    Também gere a limpeza de alertas antigos que já foram enviados.
    """

    def __init__(
        self,
        event_handler_ref: EventHandler,
        stop_event: threading.Event,
        initial_config: dict,
    ):
        threading.Thread.__init__(self, name="MQTTUploaderThread")
        self.daemon = True
        self.stop_event = stop_event
        self.event_handler = event_handler_ref  # Referência para o gestor de DB
        self.config = initial_config
        self.config_lock = threading.Lock()
        self.client = None
        
        self.pending_publish = {}
        self.pending_publish_lock = threading.Lock()
        
        self.last_internet_check = 0
        self.has_internet = False
        
        logging.info("MQTT Uploader inicializado.")

    def update_config(self, new_settings: dict):
        """Atualiza a configuração MQTT (chamado via API)."""
        logging.info("MQTT Uploader: Recebendo atualização de configuração.")
        force_reconnect = False
        with self.config_lock:
            critical_keys = [
                "mqtt_broker", "mqtt_port", "mqtt_username",
                "mqtt_password", "mqtt_device_id", "mqtt_enabled"
            ]
            for key in critical_keys:
                if key in new_settings and self.config.get(key) != new_settings[key]:
                    force_reconnect = True
                    break
            
            self.config.update(new_settings)

        if force_reconnect and self.client:
            logging.info("MQTT Uploader: Configuração crítica mudou. A forçar reconexão.")
            if self.client.is_connected():
                self.client.disconnect()

    def _check_internet(self, force=False) -> bool:
        """Verifica a conectividade com a internet (cache 60s)."""
        now = time.time()
        if not force and (now - self.last_internet_check < 60):
            return self.has_internet

        self.last_internet_check = now
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            logging.debug("MQTT Uploader: Verificação de internet OK.")
            self.has_internet = True
            return True
        except (OSError, socket.timeout):
            logging.warning("MQTT Uploader: Sem conectividade com a internet.")
            self.has_internet = False
            return False

    # --- Callbacks do Cliente MQTT ---

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info(f"MQTT Uploader: Conectado ao broker {self.config['mqtt_broker']}.")
        else:
            # RC 4 = Bad username/password, RC 5 = Not authorized
            logging.error(f"MQTT Uploader: Falha ao conectar, código: {rc}")

    def on_disconnect(self, client, userdata, rc):
        logging.warning(f"MQTT Uploader: Desconectado do broker (rc: {rc}).")

    def on_publish(self, client, userdata, mid):
        """
        Callback chamado quando o broker confirma a receção (ACK) da msg (QoS 1).
        """
        with self.pending_publish_lock:
            if mid not in self.pending_publish:
                logging.warning(f"MQTT Uploader: Recebido ACK para MID desconhecido: {mid}")
                return
            
            alert_db_id = self.pending_publish.pop(mid)
        
        try:
            sent_time = datetime.now().isoformat()
            self.event_handler.mark_alert_as_sent(alert_db_id, sent_time)
            logging.info(f"*** ALERTA ENVIADO (MQTT) *** ID DB: {alert_db_id}")
        except Exception as e:
            logging.error(f"MQTT Uploader: Falha ao marcar alerta {alert_db_id} como enviado no DB: {e}")

    # --- Lógica Principal da Thread ---

    def _connect_client(self):
        """Cria e conecta o cliente MQTT."""
        if self.client and self.client.is_connected():
            return
            
        try:
            with self.config_lock:
                broker = self.config.get("mqtt_broker", "broker.hivemq.com")
                port = int(self.config.get("mqtt_port", 1883))
                device_id = self.config.get("mqtt_device_id", "dms_default")
                username = self.config.get("mqtt_username")
                password = self.config.get("mqtt_password")

            client_id = f"dms-{device_id}-{int(time.time())}"
            self.client = mqtt.Client(client_id=client_id)
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_publish = self.on_publish

            if username:
                self.client.username_pw_set(username, password)
            
            # --- INÍCIO DA CORREÇÃO ---
            if port == 8883:
                logging.info("MQTT Uploader: A usar conexão segura (TLS) na porta 8883.")
                self.client.tls_set()
            # --- FIM DA CORREÇÃO ---
            
            self.client.connect_async(broker, port, 60)
            self.client.loop_start()
            logging.info(f"MQTT Uploader: A tentar conectar a {broker}:{port}...")
            
        except Exception as e:
            logging.error(f"MQTT Uploader: Erro ao iniciar conexão: {e}", exc_info=True)
            if self.client:
                self.client.loop_stop()
            self.client = None

    def _process_pending_alerts(self):
        """Busca alertas pendentes no DB e tenta publicá-los."""
        if not self.client or not self.client.is_connected():
            logging.debug("MQTT Uploader: Cliente não conectado, a saltar envio.")
            return

        try:
            pending_alerts = self.event_handler.get_pending_alerts(limit=10)
            if not pending_alerts:
                logging.debug("MQTT Uploader: Nenhum alerta pendente para enviar.")
                return

            logging.info(f"MQTT Uploader: {len(pending_alerts)} alertas pendentes para enviar.")

            with self.config_lock:
                fleet_id = self.config.get("mqtt_fleet_id", "default_fleet")
                device_id = self.config.get("mqtt_device_id", "dms_default")
            
            topic = f"dms/alerts/{fleet_id}/{device_id}"

            for alert in pending_alerts:
                try:
                    payload = {
                        "device_id": device_id,
                        "fleet_id": fleet_id,
                        "local_timestamp": alert.get("timestamp"),
                        "event_type": alert.get("event_type"),
                        "details": alert.get("details"),
                        "local_db_id": alert.get("id")
                    }
                    payload_json = json.dumps(payload)

                    msg_info = self.client.publish(topic, payload_json, qos=1)

                    if msg_info.rc == mqtt.MQTT_ERR_SUCCESS:
                        with self.pending_publish_lock:
                            self.pending_publish[msg_info.mid] = alert.get("id")
                        logging.debug(f"MQTT Uploader: Publicado alerta ID {alert.get('id')}, MID {msg_info.mid}")
                    else:
                        logging.warning(f"MQTT Uploader: Falha ao publicar ID {alert.get('id')}. RC: {msg_info.rc}")
                        if msg_info.rc == mqtt.MQTT_ERR_QUEUE_SIZE:
                            logging.error("MQTT Uploader: Fila Paho cheia. A aguardar.")
                            break 
                
                except Exception as e:
                    logging.error(f"MQTT Uploader: Erro ao processar alerta ID {alert.get('id')}: {e}")

        except Exception as e:
            logging.error(f"MQTT Uploader: Erro fatal em _process_pending_alerts: {e}", exc_info=True)

    def _process_cleanup(self):
        """Limpa alertas locais antigos que já foram enviados."""
        try:
            with self.config_lock:
                days_old = int(self.config.get("mqtt_retention_days", 10))
            
            if days_old <= 0:
                logging.debug("MQTT Uploader: Limpeza de alertas desativada (dias <= 0).")
                return

            logging.debug(f"MQTT Uploader: A executar limpeza de alertas enviados > {days_old} dias.")
            deleted_count, failed_count = self.event_handler.cleanup_sent_alerts(days_old)
            
            if deleted_count > 0 or failed_count > 0:
                logging.info(
                    f"MQTT Uploader: Limpeza concluída. "
                    f"{deleted_count} alertas apagados, {failed_count} falhas."
                )
        except Exception as e:
            logging.error(f"MQTT Uploader: Erro no processo de limpeza: {e}", exc_info=True)


    def run(self):
        """Loop principal da thread."""
        logging.info("Thread MQTT Uploader iniciada.")
        
        self.stop_event.wait(10.0) 

        while not self.stop_event.is_set():
            try:
                with self.config_lock:
                    is_enabled = self.config.get("mqtt_enabled", False)
                
                if not is_enabled:
                    logging.debug("MQTT Uploader: Módulo desativado na configuração.")
                    if self.client and self.client.is_connected():
                        self.client.disconnect()
                    if self.client:
                        self.client.loop_stop()
                        self.client = None
                    self.stop_event.wait(30.0)
                    continue

                if not self._check_internet():
                    logging.debug("MQTT Uploader: Sem internet, a aguardar.")
                    self.stop_event.wait(60.0)
                    continue
                
                if not self.client:
                    self._connect_client()
                    
                if self.client and self.client.is_connected():
                    self._process_pending_alerts()
                    self._process_cleanup()
                
            except Exception as e:
                logging.error(f"MQTT Uploader: Erro no loop principal: {e}", exc_info=True)

            self.stop_event.wait(30.0) # Intervalo do loop principal

        # --- Encerramento ---
        logging.info("Thread MQTT Uploader: Sinal de paragem recebido.")
        if self.client:
            self.client.disconnect()
            self.client.loop_stop()
        logging.info("Thread MQTT Uploader terminada.")