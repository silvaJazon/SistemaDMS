# Documentação: Gestor de Eventos (Central de Alertas - SQLite)
# Responsável por receber eventos (com frames) de uma fila e guardá-los
# de forma assíncrona (imagem JPG + linha SQLite).
# (Atualizado para aceitar 'stop_event' e usar paths relativos consistentes)

import threading
import queue
import logging
import os
import cv2
import json
from datetime import datetime
import sqlite3
import time # Para o retry

class EventHandler(threading.Thread):
    """
    Processa eventos de alerta numa thread separada para guardar
    informações (SQLite) e imagens (JPG) sem bloquear a thread principal.
    """
    # (NOVO) Aceita stop_event no construtor
    def __init__(self, queue, stop_event, save_path="/app/alerts", db_name="alerts.db"):
        threading.Thread.__init__(self)
        self.daemon = True # Permite que a thread termine quando a app principal fechar
        self.queue = queue
        self.stop_event = stop_event # Guarda o evento de paragem
        self.save_path = save_path
        self.image_save_path = os.path.join(self.save_path, "images") # Pasta específica para imagens
        self.db_path = os.path.join(self.save_path, db_name)
        self.conn = None # Conexão SQLite será estabelecida no run()

        os.makedirs(self.image_save_path, exist_ok=True) # Cria a pasta de imagens

        logging.info(f"Gestor de Eventos inicializado. Base de dados: {self.db_path}, Imagens em: {self.image_save_path}")
        self._init_db() # Verifica/Cria a tabela imediatamente

    def _init_db(self):
        """Inicializa a base de dados SQLite e cria a tabela se não existir."""
        try:
            # isolation_level=None para autocommit (mais simples para inserções únicas)
            # timeout aumentado para evitar busy errors em escritas concorrentes (raro aqui, mas boa prática)
            conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=10.0)
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
            # (Opcional) Adicionar índice no timestamp para acelerar futuras consultas por data
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts (timestamp)
            ''')
            conn.close()
            logging.info(f"Base de dados SQLite '{self.db_path}' verificada/inicializada com sucesso.")
        except sqlite3.Error as e:
            logging.error(f"!!! Erro ao inicializar a base de dados SQLite: {e}", exc_info=True)
            # Considerar parar a aplicação se a BD não puder ser inicializada?
            # Por agora, apenas loga. A thread tentará conectar novamente.

    def _get_db_connection(self):
        """Obtém uma conexão à base de dados, com retry simples."""
        if self.conn is None:
            try:
                # isolation_level=None (autocommit)
                self.conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=10.0)
                logging.debug("Conexão SQLite estabelecida.")
            except sqlite3.Error as e:
                logging.error(f"Erro ao conectar à base de dados SQLite: {e}")
                self.conn = None # Garante que continua None
        return self.conn # Retorna None se a conexão falhou


    def run(self):
        """Loop principal da thread: espera por eventos na fila e processa-os."""
        logging.info("Thread do Gestor de Eventos (SQLite) iniciada.")

        while not self.stop_event.is_set(): # Usa o evento de paragem
            try:
                # Espera por um item (timeout curto para verificar stop_event frequentemente)
                item = self.queue.get(timeout=0.2)

                if item is None: # Sinal de paragem explícito (opcional)
                    break

                self.process_event(item)
                self.queue.task_done()

            except queue.Empty:
                # Timeout - normal, apenas continua a verificar stop_event
                continue
            except Exception as e:
                logging.error(f"Erro inesperado na thread do EventHandler: {e}", exc_info=True)
                # Pausa curta para evitar spam de logs em caso de erro rápido
                time.sleep(1)

        # Fecha a conexão SQLite ao terminar
        if self.conn:
            logging.info("A fechar conexão SQLite...")
            self.conn.close()
        logging.info("Thread do Gestor de Eventos (SQLite) terminada.")

    def process_event(self, item):
        """Guarda os dados do evento em SQLite e a imagem em JPG."""
        conn = self._get_db_connection()
        if not conn:
             logging.error("Impossível processar evento: sem conexão à base de dados.")
             # Poderíamos tentar colocar o item de volta na fila? Por agora, perdemos o evento.
             return

        try:
            event_data = item.get("event_data")
            frame = item.get("frame")

            if event_data is None or frame is None:
                logging.warning(f"Evento inválido recebido (event_data={event_data is None}, frame={frame is None}). Ignorando.")
                return

            event_type = event_data.get("type", "DESCONHECIDO")
            details = event_data.get("value", None) # Usar 'value' como 'details'
            timestamp_str = event_data.get("timestamp", datetime.now().isoformat() + "Z") # ISO 8601 com Z

            # Converte timestamp para objeto datetime para extrair data
            try:
                 timestamp_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                 logging.warning(f"Timestamp inválido no evento: {timestamp_str}. Usando hora atual.")
                 timestamp_dt = datetime.now()
                 timestamp_str = timestamp_dt.isoformat() + "Z" # Garante formato ISO

            # Cria estrutura de pastas Ano/Mês/Dia
            date_path = timestamp_dt.strftime("%Y/%m/%d")
            image_dir = os.path.join(self.image_save_path, date_path)
            os.makedirs(image_dir, exist_ok=True)

            # Cria nome de ficheiro único (sem Z no nome, mas Z no timestamp guardado)
            filename_base = timestamp_dt.strftime("%Y-%m-%dT%H-%M-%S.%f") + f"_{event_type}"
            image_filename = filename_base + ".jpg"
            image_full_path = os.path.join(image_dir, image_filename)

            # Guarda a imagem JPG
            quality = 90
            success = cv2.imwrite(image_full_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

            if not success:
                 logging.error(f"Falha ao guardar imagem: {image_full_path}")
                 image_relative_path = None # Não guarda referência se falhou
            else:
                 # Guarda o caminho RELATIVO à pasta 'images' na BD
                 image_relative_path = os.path.join(date_path, image_filename)
                 # (NOVO) Garante barras '/' para consistência web
                 image_relative_path = image_relative_path.replace(os.path.sep, '/')


            # Insere na base de dados SQLite
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO alerts (timestamp, event_type, details, image_file)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp_str, event_type, details, image_relative_path))
                # conn.commit() # Não necessário com isolation_level=None

                log_msg_img = f"Imagem: {image_relative_path}" if image_relative_path else "Imagem falhou"
                logging.warning(f"*** ALERTA GUARDADO (SQLite) *** Tipo: {event_type}, {log_msg_img}")

            except sqlite3.Error as e:
                logging.error(f"Erro ao inserir alerta na base de dados: {e}")
                # Se a inserção falhar, talvez apagar a imagem? Por agora, deixamos.

        except Exception as e:
            logging.error(f"Falha inesperada ao processar/gravar evento: {e}", exc_info=True)

    def get_alerts(self, limit=50):
        """Busca os últimos 'limit' alertas da base de dados."""
        conn = self._get_db_connection()
        if not conn:
             logging.error("Impossível buscar alertas: sem conexão à base de dados.")
             return [] # Retorna lista vazia

        alerts = []
        try:
            # Usar row_factory para retornar dicionários em vez de tuplos
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, event_type, details, image_file
                FROM alerts
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            # Converte os resultados de sqlite3.Row para dicionários padrão
            alerts = [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logging.error(f"Erro ao buscar alertas da base de dados: {e}")
            alerts = [] # Retorna vazio em caso de erro

        return alerts

