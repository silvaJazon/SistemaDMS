import threading
import queue
import logging
import os
import cv2
from datetime import datetime
import sqlite3 # (NOVO) Importa a biblioteca SQLite

class EventHandler(threading.Thread):
    """
    Processa eventos de alerta numa thread separada para guardar
    informações numa base de dados SQLite e imagens em pastas organizadas por data.
    """
    def __init__(self, queue, base_save_path="/app/alerts"):
        threading.Thread.__init__(self)
        self.daemon = True
        self.queue = queue
        self.base_save_path = base_save_path
        self.image_save_path = os.path.join(self.base_save_path, "images") # (NOVO) Subpasta para imagens
        self.db_path = os.path.join(self.base_save_path, "alerts.db") # (NOVO) Caminho para a base de dados
        self.running = False

        # Cria a pasta base e a subpasta de imagens se não existirem
        os.makedirs(self.image_save_path, exist_ok=True)

        # (NOVO) Inicializa a base de dados
        self._init_db()

        logging.info(f"Gestor de Eventos inicializado. Base de dados: {self.db_path}, Imagens em: {self.image_save_path}")

    def _init_db(self):
        """Cria a base de dados e a tabela 'alerts' se não existirem."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details TEXT,
                    image_file TEXT
                )
            ''')
            # (Opcional) Adicionar um índice na coluna timestamp para pesquisas mais rápidas
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts (timestamp DESC)')
            conn.commit()
            conn.close()
            logging.info(f"Base de dados SQLite '{self.db_path}' verificada/inicializada com sucesso.")
        except sqlite3.Error as e:
            logging.error(f"Erro ao inicializar a base de dados SQLite: {e}", exc_info=True)
            # Considerar parar a aplicação se a BD falhar aqui? Por agora, apenas loga.

    def run(self):
        """Loop principal da thread: espera por eventos na fila e processa-os."""
        self.running = True
        logging.info("Thread do Gestor de Eventos (SQLite) iniciada.")

        while self.running:
            conn = None # (NOVO) Garante que a conexão é definida
            try:
                item = self.queue.get(timeout=1)
                if item is None: # Sinal de paragem
                    break

                # (NOVO) Abre a conexão com a BD *antes* de processar
                conn = sqlite3.connect(self.db_path, timeout=10) # Timeout de 10s para operações

                self.process_event(item, conn)
                self.queue.task_done()

            except queue.Empty:
                continue
            except sqlite3.Error as e:
                logging.error(f"Erro SQLite na thread do EventHandler: {e}", exc_info=True)
                # Poderíamos tentar reconectar ou logar o evento falhado
            except Exception as e:
                logging.error(f"Erro na thread do EventHandler: {e}", exc_info=True)
            finally:
                # (NOVO) Garante que a conexão é fechada, mesmo em caso de erro
                if conn:
                    conn.close()

        logging.info("Thread do Gestor de Eventos (SQLite) terminada.")

    def process_event(self, item, conn):
        """Guarda os dados do evento na BD SQLite e a imagem em JPG."""
        try:
            event_data = item.get("event_data")
            frame = item.get("frame")

            if event_data is None:
                logging.warning("Evento recebido com event_data=None. Ignorando.")
                return
            if frame is None:
                logging.warning("Evento recebido sem frame. Ignorando.")
                return

            event_type = event_data.get("type", "DESCONHECIDO")
            details = event_data.get("value", None) # Pode ser nulo
            timestamp_str = event_data.get("timestamp", datetime.now().isoformat() + "Z") # ISO 8601 format

            # Converte timestamp para objeto datetime
            try:
                # O SQLite guarda TEXT, mas usar datetime ajuda a extrair partes
                timestamp_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                logging.warning(f"Timestamp inválido recebido: {timestamp_str}. Usando 'agora'.")
                timestamp_dt = datetime.now()
                timestamp_str = timestamp_dt.isoformat() + "Z" # Atualiza o string

            # --- (NOVO) Cria estrutura de pastas Ano/Mês/Dia ---
            year_str = timestamp_dt.strftime("%Y")
            month_str = timestamp_dt.strftime("%m")
            day_str = timestamp_dt.strftime("%d")
            image_dir = os.path.join(self.image_save_path, year_str, month_str, day_str)
            os.makedirs(image_dir, exist_ok=True) # Cria as pastas se não existirem

            # --- (NOVO) Cria nome de ficheiro e caminho relativo ---
            filename_base = timestamp_dt.strftime("%Y-%m-%dT%H-%M-%S.%fZ") + f"_{event_type}"
            image_filename = filename_base + ".jpg"
            relative_image_path = os.path.join(year_str, month_str, day_str, image_filename) # Ex: 2025/10/28/....jpg
            full_image_path = os.path.join(self.image_save_path, relative_image_path) # Caminho completo para guardar

            # Guarda a imagem JPG
            save_success = cv2.imwrite(full_image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not save_success:
                 logging.error(f"Falha ao guardar a imagem: {full_image_path}")
                 # Decide se continua ou não. Por agora, continua e guarda na BD sem path da imagem.
                 relative_image_path = None # Não guarda o path na BD se a imagem falhou

            # --- (NOVO) Insere na base de dados SQLite ---
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (timestamp, event_type, details, image_file)
                VALUES (?, ?, ?, ?)
            ''', (timestamp_str, event_type, details, relative_image_path))
            conn.commit()

            logging.info(f"*** ALERTA GUARDADO (SQLite) *** Tipo: {event_type}, Imagem: {relative_image_path or 'N/A'}")

        # Não apanhar sqlite3.Error aqui, deixa o loop principal tratar disso
        except Exception as e:
            logging.error(f"Falha ao processar e guardar evento: {e}", exc_info=True)
            # Se a conexão existir e ocorreu um erro ANTES do commit, faz rollback
            if conn:
                try:
                    conn.rollback()
                except Exception as rb_e:
                    logging.error(f"Erro adicional durante o rollback: {rb_e}")

    def stop(self):
        """Sinaliza à thread para parar."""
        logging.info("A sinalizar paragem para o Gestor de Eventos (SQLite)...")
        self.running = False
        self.queue.put(None) # Envia um item 'None' para desbloquear o get()
