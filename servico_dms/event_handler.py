# Documentação: Gestor de Eventos (Central de Alertas - SQLite)
# Responsável por receber eventos (com frames) de uma fila e guardá-los
# de forma assíncrona (imagem JPG + linha SQLite).
# (Atualizado para ser Thread-Safe com SQLite)

import threading
import queue
import logging
import os
import cv2
# import json (F401 - Removido)
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

    def _init_db(self):
        """Inicializa a base de dados SQLite e cria a tabela se não existir."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details TEXT,
                    image_file TEXT
                )
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts (timestamp)
            """
            )
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
        """Guarda os dados do evento em SQLite e a imagem em JPG. Cria a sua própria ligação DB."""
        conn = None
        try:
            # --- Extração de Dados ---
            event_data = item.get("event_data")
            frame = item.get("frame")
            if event_data is None or frame is None:
                logging.warning(
                    f"Evento inválido recebido (event_data={event_data is None}, "
                    f"frame={frame is None}). Ignorando."
                )
                return

            event_type = event_data.get("type", "DESCONHECIDO")
            details = event_data.get("value", None)
            timestamp_str = event_data.get("timestamp", datetime.now().isoformat() + "Z")

            try:
                timestamp_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                logging.warning(
                    f"Timestamp inválido no evento: {timestamp_str}. Usando hora atual."
                )
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

            # --- Guardar Imagem ---
            quality = 90
            success = cv2.imwrite(
                image_full_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            image_relative_path = None
            if success:
                image_relative_path = os.path.join(date_path, image_filename).replace(
                    os.path.sep, "/"
                )
            else:
                logging.error(f"Falha ao guardar imagem: {image_full_path}")

            # --- Inserir na Base de Dados (com ligação local) ---
            logging.debug(f"ProcessEvent: A conectar a {self.db_path}...")
            conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=10.0)
            cursor = conn.cursor()
            logging.debug("ProcessEvent: Ligação criada, a executar INSERT...")
            cursor.execute(
                """
                INSERT INTO alerts (timestamp, event_type, details, image_file)
                VALUES (?, ?, ?, ?)
            """,
                (timestamp_str, event_type, details, image_relative_path),
            )
            logging.debug("ProcessEvent: INSERT executado.")

            log_msg_img = (
                f"Imagem: {image_relative_path}" if image_relative_path else "Imagem falhou"
            )
            logging.warning(
                f"*** ALERTA GUARDADO (SQLite) *** Tipo: {event_type}, {log_msg_img}"
            )

        except sqlite3.Error as db_err:
            logging.error(
                f"Erro SQLite ao processar/gravar evento: {db_err}", exc_info=True
            )
        except Exception as e:
            logging.error(
                f"Falha inesperada ao processar/gravar evento: {e}", exc_info=True
            )
        finally:
            if conn:
                logging.debug("ProcessEvent: A fechar ligação DB.")
                conn.close()

    def get_alerts(self, limit=50):
        """Busca os últimos 'limit' alertas da base de dados. Cria a sua própria ligação DB."""
        conn = None
        alerts = []
        try:
            logging.debug(f"GetAlerts: A conectar a {self.db_path}...")
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row  # Retorna como dicionários
            cursor = conn.cursor()

            safe_limit = int(limit)

            logging.debug("GetAlerts: Ligação criada, a executar SELECT...")
            cursor.execute(
                """
                SELECT id, timestamp, event_type, details, image_file
                FROM alerts
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (safe_limit,),
            )
            alerts = [dict(row) for row in cursor.fetchall()]
            logging.debug(f"GetAlerts: SELECT retornou {len(alerts)} alertas.")
        except sqlite3.Error as e:
            logging.error(f"Erro SQLite ao buscar alertas: {e}", exc_info=True)
            alerts = []  # Retorna vazio em caso de erro
        except Exception as e:
            logging.error(f"Erro inesperado ao buscar alertas: {e}", exc_info=True)
            alerts = []
        finally:
            if conn:
                logging.debug("GetAlerts: A fechar ligação DB.")
                conn.close()

        return alerts