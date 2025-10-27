# Documentação: Gestor de Eventos Assíncrono (A "Central")
# Responsável por:
# 1. Receber "eventos" (alertas + imagens) através de uma Fila (Queue).
# 2. Guardar a imagem e o log .jsonl numa thread separada (para não bloquear a deteção).

import threading
import time
import logging
import os
import cv2
import json
from datetime import datetime
import queue # Importa o módulo queue

class EventHandler(threading.Thread):
    """
    Esta thread consome eventos da fila e guarda-os no disco.
    Isto evita que a operação "lenta" de guardar ficheiros
    atrase o loop principal de deteção.
    """
    
    # (NOVO) A assinatura __init__ está correta agora
    def __init__(self, queue, save_path="/app/alerts"):
        threading.Thread.__init__(self)
        self.daemon = True
        self.running = False
        
        # Esta é a fila (Queue) partilhada com o app.py
        self.queue = queue 
        
        # Caminho onde os alertas serão guardados
        self.save_path = save_path
        
        # Caminho completo para o ficheiro de log JSONL
        self.log_file_path = os.path.join(self.save_path, "alerts_log.jsonl")
        
        logging.info(f"Gestor de Eventos inicializado. Alertas serão guardados em: {self.save_path}")

    def run(self):
        """O loop principal da thread worker."""
        self.running = True
        logging.info("Thread do Gestor de Eventos iniciada.")
        
        while self.running:
            try:
                # Espera por um item na fila (bloqueia até receber algo)
                # O timeout de 1 segundo permite que a thread verifique self.running
                item = self.queue.get(timeout=1.0)
                
                if item:
                    self.process_event(item)
                    self.queue.task_done()
                    
            except queue.Empty:
                # Isto é normal, significa que a fila esteve vazia durante 1s
                continue
            except Exception as e:
                logging.error(f"Erro na thread do Gestor de Eventos: {e}", exc_info=True)
        
        logging.info("Thread do Gestor de Eventos terminada.")

    def process_event(self, item):
        """Processa e guarda um único evento de alerta."""
        try:
            event_data = item.get("event")
            frame = item.get("frame")
            event_type = event_data.get("type", "DESCONHECIDO")
            
            # 1. Gerar carimbo de data/hora e nome do ficheiro
            # Formato ISO 8601 (seguro para nomes de ficheiro)
            timestamp_iso = datetime.utcnow().isoformat(timespec='microseconds') + "Z"
            filename = f"{timestamp_iso}_{event_type}.jpg"
            image_path = os.path.join(self.save_path, filename)

            # 2. Guardar a imagem (operação "lenta")
            cv2.imwrite(image_path, frame)
            
            # 3. Preparar o log
            log_entry = {
                "timestamp": timestamp_iso,
                "event_type": event_type,
                "image_file": filename,
                "details": event_data.get("details", {})
            }

            # 4. Anexar ao ficheiro JSONL (JSON Lines)
            # O 'a' significa "append" (anexar)
            with open(self.log_file_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            logging.warning(f"*** ALERTA ARMAZENADO *** Tipo: {event_type}, Imagem: {filename}")

        except Exception as e:
            logging.error(f"Falha ao processar e guardar evento: {e}", exc_info=True)

    def stop(self):
        """Sinaliza à thread para parar."""
        self.running = False

