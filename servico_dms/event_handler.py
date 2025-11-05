# Documentação: Gestor de Eventos (Central de Alertas - SQLite)
# (Atualizado para suportar fila de envio MQTT e limpeza)

import threading
import queue
import logging
import os
import cv2
from datetime import datetime
import sqlite3
import time  # Para o retry

class EventHandler(threading.Thread):
    """
    Processa eventos de alerta numa thread separada para guardar
    informações (SQLite) e imagens (JPG) sem bloquear a thread principal.
    Gere as ligações SQLite de forma segura para threads.
    """

    def __init__(
        self, queue, stop_event, save_path="/app/alerts", db_name="alerts.db"
    ):
        threading.Thread.__init__(self, name="EventHandlerThread")
        self.daemon = True
        self.queue = queue
        self.stop_event = stop_event
        self.save_path = save_path
        self.image_save_path = os.path.join(self.save_path, "images")
        self.db_path = os.path.join(self.save_path, db_name)

        os.makedirs(self.image_save_path, exist_ok=True)

        logging.info(
            f"Gestor de Eventos inicializado. Base de dados: {self.db_path}, "
            f"Imagens em: {self.image_save_path}"
        )
        self._init_db()

    def _get_db_connection(self):
        """Cria e retorna uma nova ligação à base de dados."""
        try:
            conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            return conn
        except sqlite3.Error as e:
            logging.error(f"!!! Erro fatal ao conectar a SQLite: {e}", exc_info=True)
            raise

    def _init_db(self):
        """Inicializa a base de dados SQLite e cria/atualiza a tabela."""
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            # 1. Cria a tabela principal (com a nova coluna, se for nova)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details TEXT,
                    image_file TEXT,
                    mqtt_sent_timestamp TEXT DEFAULT NULL 
                )
            """
            )
            # 2. Cria o índice da coluna original
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts (timestamp)
            """
            )
            
            # --- INÍCIO DA CORREÇÃO ---
            
            # 3. Tenta adicionar a coluna (Migração para DBs existentes)
            #    Isto deve vir ANTES de tentar criar um índice nessa coluna.
            try:
                cursor.execute("ALTER TABLE alerts ADD COLUMN mqtt_sent_timestamp TEXT DEFAULT NULL")
                logging.info("Migração DB: Coluna 'mqtt_sent_timestamp' adicionada.")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    logging.debug("Migração DB: Coluna 'mqtt_sent_timestamp' já existe.")
                else:
                    raise # Levanta outros erros

            # 4. Agora que a coluna existe, cria o índice para ela
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mqtt_sent 
                ON alerts (mqtt_sent_timestamp)
                """
            )
            # --- FIM DA CORREÇÃO ---

            logging.info(
                f"Base de dados SQLite '{self.db_path}' verificada/inicializada com sucesso."
            )
        except sqlite3.Error as e:
            logging.error(
                f"!!! Erro ao inicializar a base de dados SQLite: {e}", exc_info=True
            )
        finally:
            if conn:
                conn.close()

    def run(self):
        """Loop principal da thread: espera por eventos na fila e processa-os."""
        logging.info("Thread do Gestor de Eventos (SQLite) iniciada.")

        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.2)
                if item is None:
                    break

                self.process_event(item)
                self.queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(
                    f"Erro inesperado na thread do EventHandler: {e}", exc_info=True
                )
                time.sleep(1)

        logging.info("Thread do Gestor de Eventos (SQLite) terminada.")

    def process_event(self, item):
        """Guarda os dados do evento em SQLite e a imagem em JPG."""
        conn = None
        try:
            event_data = item.get("event_data")
            frame = item.get("frame")
            if event_data is None or frame is None:
                logging.warning(f"Evento inválido. Ignorando.")
                return

            event_type = event_data.get("type", "DESCONHECIDO")
            details = event_data.get("value", None)
            timestamp_str = event_data.get("timestamp", datetime.now().isoformat() + "Z")

            try:
                timestamp_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                timestamp_dt = datetime.now()
                timestamp_str = timestamp_dt.isoformat() + "Z"

            date_path = timestamp_dt.strftime("%Y/%m/%d")
            image_dir = os.path.join(self.image_save_path, date_path)
            os.makedirs(image_dir, exist_ok=True)

            filename_base = (
                timestamp_dt.strftime("%Y-%m-%dT%H-%M-%S.%f") + f"_{event_type}"
            )
            image_filename = filename_base + ".jpg"
            image_full_path = os.path.join(image_dir, image_filename)

            success = cv2.imwrite(
                image_full_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            image_relative_path = None
            if success:
                image_relative_path = os.path.join(date_path, image_filename).replace(
                    os.path.sep, "/"
                )
            else:
                logging.error(f"Falha ao guardar imagem: {image_full_path}")

            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO alerts (timestamp, event_type, details, image_file)
                VALUES (?, ?, ?, ?)
            """,
                (timestamp_str, event_type, details, image_relative_path),
            )
            logging.debug("ProcessEvent: INSERT executado.")

            log_msg_img = f"Imagem: {image_relative_path}" if success else "Imagem falhou"
            logging.warning(
                f"*** ALERTA GUARDADO (SQLite) *** Tipo: {event_type}, {log_msg_img}"
            )

        except sqlite3.Error as db_err:
            logging.error(f"Erro SQLite ao processar/gravar evento: {db_err}", exc_info=True)
        except Exception as e:
            logging.error(f"Falha inesperada ao processar/gravar evento: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()

    def get_alerts(self, limit=50):
        """Busca os últimos 'limit' alertas da base de dados (para a UI)."""
        conn = None
        alerts = []
        try:
            conn = self._get_db_connection()
            conn.row_factory = sqlite3.Row  # Retorna como dicionários
            cursor = conn.cursor()

            safe_limit = int(limit)
            cursor.execute(
                """
                SELECT id, timestamp, event_type, details, image_file, mqtt_sent_timestamp
                FROM alerts
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (safe_limit,),
            )
            alerts = [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logging.error(f"Erro SQLite ao buscar alertas: {e}", exc_info=True)
            alerts = []
        finally:
            if conn:
                conn.close()
        return alerts

    # --- MÉTODOS NOVOS (para MQTTUploader) ---

    def get_pending_alerts(self, limit=10):
        """Busca alertas que ainda não foram enviados via MQTT."""
        conn = None
        alerts = []
        try:
            conn = self._get_db_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, event_type, details
                FROM alerts
                WHERE mqtt_sent_timestamp IS NULL
                ORDER BY timestamp ASC
                LIMIT ?
            """,
                (int(limit),),
            )
            alerts = [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logging.error(f"Erro SQLite ao buscar alertas pendentes: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()
        return alerts

    def mark_alert_as_sent(self, alert_id: int, sent_timestamp: str):
        """Atualiza um alerta como enviado no DB."""
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE alerts
                SET mqtt_sent_timestamp = ?
                WHERE id = ? AND mqtt_sent_timestamp IS NULL
            """,
                (sent_timestamp, alert_id),
            )
            if cursor.rowcount == 0:
                 logging.warning(f"MarkAsSent: Alerta ID {alert_id} não encontrado ou já marcado.")
        except sqlite3.Error as e:
            logging.error(f"Erro SQLite ao marcar alerta {alert_id} como enviado: {e}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()

    def cleanup_sent_alerts(self, days_old=10):
        """
        Apaga alertas E imagens que foram enviados há mais de 'days_old' dias.
        Retorna (deleted_count, failed_count)
        """
        conn = None
        deleted_count = 0
        failed_count = 0
        
        cutoff_date_sql = f"date('now', '-{int(days_old)} days', 'localtime')"

        try:
            conn = self._get_db_connection()
            conn.isolation_level = "DEFERRED" 
            cursor = conn.cursor()
            
            # 1. Encontra os alertas e imagens para apagar
            cursor.execute(
                f"""
                SELECT id, image_file FROM alerts
                WHERE mqtt_sent_timestamp IS NOT NULL
                AND mqtt_sent_timestamp < {cutoff_date_sql}
                """
            )
            alerts_to_delete = cursor.fetchall()
            if not alerts_to_delete:
                return 0, 0

            logging.info(f"Cleanup: Encontrados {len(alerts_to_delete)} alertas antigos para apagar.")

            for alert_id, image_relative_path in alerts_to_delete:
                # 2. Apaga a imagem associada
                if image_relative_path:
                    try:
                        full_image_path = os.path.join(
                            self.image_save_path, image_relative_path.replace("/", os.path.sep)
                        )
                        if os.path.exists(full_image_path):
                            os.remove(full_image_path)
                        else:
                            logging.warning(f"Cleanup: Imagem {full_image_path} não encontrada.")
                    except OSError as e:
                        logging.error(f"Cleanup: Falha ao apagar imagem {full_image_path}: {e}")
                        failed_count += 1

                # 3. Apaga a entrada do DB
                try:
                    cursor.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
                    deleted_count += 1
                except sqlite3.Error as e:
                     logging.error(f"Cleanup: Falha ao apagar alerta ID {alert_id} do DB: {e}")
                     failed_count += 1
            
            # 4. Confirma a transação
            conn.commit()

        except sqlite3.Error as e:
            logging.error(f"Erro SQLite na transação de limpeza: {e}", exc_info=True)
            if conn:
                conn.rollback()
        except Exception as e:
            logging.error(f"Erro inesperado na limpeza: {e}", exc_info=True)
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
                
        return deleted_count, failed_count