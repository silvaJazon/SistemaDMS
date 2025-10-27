# Documentação: Gestor de Eventos (Event Handler)
# Responsável por receber alertas do loop principal e guardá-los
# de forma assíncrona (noutra thread) para não bloquear a deteção.

import threading
import queue
import os
import cv2
import datetime
import logging
import json
import sys # Adicionado para sys.exit

class EventHandler:
    """
    Gere uma fila de eventos e um 'worker' (trabalhador)
    para guardar alertas (metadados + imagem) no disco.
    """
    
    def __init__(self, save_path="/app/alerts"):
        self.save_path = save_path
        # .jsonl (JSON Lines) é um formato onde cada linha é um JSON válido
        self.log_file_path = os.path.join(self.save_path, "alerts_log.jsonl") 
        
        # Garante que o diretório de destino existe
        if not os.path.exists(self.save_path):
            try:
                os.makedirs(self.save_path)
            except OSError as e:
                logging.error(f"!!! ERRO FATAL: Não foi possível criar o diretório de alertas: {e}")
                sys.exit(1) # Para a aplicação se não puder guardar

        # A Fila (Queue) é 'thread-safe' (segura para threads)
        self.event_queue = queue.Queue(maxsize=100) # Define um limite para a fila
        
        # O 'None' é um sinal para a thread parar ('sentinel')
        self.stop_signal = None 
        
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.running = False
        
        logging.info(f"Gestor de Eventos inicializado. Alertas serão guardados em: {self.save_path}")

    def start(self):
        """Inicia a thread do 'worker'."""
        self.running = True
        self.worker_thread.start()
        logging.info("Thread do Gestor de Eventos iniciada.")

    def stop(self):
        """Sinaliza ao 'worker' para parar."""
        if not self.running:
            return
            
        self.running = False
        self.event_queue.put(self.stop_signal) # Envia o sinal de paragem
        logging.info("A aguardar 'worker' de eventos terminar...")
        self.worker_thread.join(timeout=2.0) # Espera 2s pela thread
        logging.info("'Worker' de eventos terminado.")

    def _worker_loop(self):
        """O loop que corre na thread 'worker'."""
        while self.running:
            try:
                # Bloqueia até um item estar disponível
                event = self.event_queue.get(timeout=1.0) # Timeout de 1s para verificar self.running
                
                if event is None: # Se .get() der timeout
                    continue

                # Verifica se é o sinal de paragem
                if event is self.stop_signal:
                    break
                    
                # Processa o evento
                self._save_event_to_disk(event)
                
                # Informa a fila que a tarefa foi concluída
                self.event_queue.task_done()
                
            except queue.Empty:
                # Isto é normal, acontece se não houver eventos durante 1s
                continue 
            except Exception as e:
                logging.error(f"Erro no 'worker' de eventos: {e}", exc_info=True)

    def log_event(self, event_type, value, frame):
        """
        Método PÚBLICO chamado pelo app.py para registar um novo evento.
        Isto é rápido e não-bloqueante.
        """
        if not self.running:
            return
            
        try:
            # Cria o carimbo de data/hora
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            
            event_data = {
                "timestamp_utc": timestamp.isoformat(),
                "event_type": event_type,
                "value": round(value, 3) if isinstance(value, float) else value,
                "frame": frame.copy() # Copia o frame para processamento seguro
            }
            
            # Coloca o evento na fila (sem bloquear)
            self.event_queue.put_nowait(event_data)
            
        except queue.Full:
            logging.warning("Fila de eventos está cheia! A descartar alerta.")
        except Exception as e:
            logging.error(f"Erro ao colocar evento na fila: {e}", exc_info=True)

    def _save_event_to_disk(self, event):
        """
        Método PRIVADO que faz o trabalho lento de I/O (guardar no disco).
        """
        try:
            # 1. Preparar nomes e metadados
            # Converte o timestamp para string (seguro para nomes de ficheiro)
            ts_str = event['timestamp_utc'].replace(":", "-").replace("+00-00", "Z")
            
            image_filename = f"{ts_str}_{event['event_type']}.jpg"
            image_path = os.path.join(self.save_path, image_filename)
            
            # 2. Guardar a Imagem (Operação Lenta)
            cv2.imwrite(image_path, event['frame'], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            
            # 3. Preparar o Log (JSON)
            log_entry = {
                "timestamp_utc": event['timestamp_utc'],
                "event_type": event['event_type'],
                "value": event['value'],
                "image_path_container": image_path # Caminho *dentro* do contentor
            }
            
            # 4. Guardar o Log (Operação Lenta)
            # Usamos 'a' (append) para adicionar ao ficheiro de log
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n") # .jsonl (JSON Lines)

            logging.warning(f"*** ALERTA ARMAZENADO *** Tipo: {event['event_type']}, Imagem: {image_filename}")

        except Exception as e:
            logging.error(f"Falha ao guardar evento no disco: {e}", exc_info=True)

