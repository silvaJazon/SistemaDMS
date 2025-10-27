# Documentação: Gestor de Eventos (Central de Alertas)
# Responsável por receber eventos (com frames) de uma fila e guardá-los
# de forma assíncrona (imagem JPG + linha JSONL).
# (Atualizado com correção para 'NoneType' em event_data)

import threading
import queue
import logging
import os
import cv2
import json
from datetime import datetime

class EventHandler(threading.Thread):
    """
    Processa eventos de alerta numa thread separada para guardar
    informações e imagens sem bloquear a thread principal.
    """
    def __init__(self, queue, save_path="/app/alerts"):
        threading.Thread.__init__(self)
        self.daemon = True # Permite que a thread termine quando a app principal fechar
        self.queue = queue
        self.save_path = save_path
        self.running = False
        
        # Cria a pasta de alertas se não existir
        os.makedirs(self.save_path, exist_ok=True)
        
        # Caminho para o ficheiro de log JSONL
        self.log_file_path = os.path.join(self.save_path, "alerts_log.jsonl")
        
        logging.info(f"Gestor de Eventos inicializado. Alertas serão guardados em: {self.save_path}")

    def run(self):
        """Loop principal da thread: espera por eventos na fila e processa-os."""
        self.running = True
        logging.info("Thread do Gestor de Eventos iniciada.")
        
        while self.running:
            try:
                # Espera por um item na fila (com timeout para poder parar)
                item = self.queue.get(timeout=1) 
                
                # Processa o evento (guardar JSONL e JPG)
                self.process_event(item)
                
                # Marca a tarefa como concluída na fila
                self.queue.task_done()
                
            except queue.Empty:
                # Timeout - normal, apenas continua a esperar
                continue
            except Exception as e:
                # Loga qualquer outro erro inesperado na thread
                logging.error(f"Erro na thread do EventHandler: {e}", exc_info=True)
                # Decide se deve continuar ou parar em caso de erro
                # Por agora, apenas loga e continua
                
        logging.info("Thread do Gestor de Eventos terminada.")

    def process_event(self, item):
        """Guarda os dados do evento em JSONL e a imagem em JPG."""
        try:
            event_data = item.get("event_data")
            frame = item.get("frame")

            # --- (NOVO) Verificação de Segurança ---
            if event_data is None:
                logging.warning("Evento recebido com event_data=None. Ignorando.")
                return
            # ----------------------------------------

            if frame is None:
                logging.warning("Evento recebido sem frame. Ignorando.")
                return

            # Obtém dados do evento (com valores padrão seguros)
            event_type = event_data.get("type", "DESCONHECIDO")
            timestamp_str = event_data.get("timestamp", datetime.now().isoformat() + "Z")
            
            # Cria um nome de ficheiro único para a imagem
            # Ex: 2025-10-27T18-30-05.123456Z_SONOLENCIA.jpg
            timestamp_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            filename_base = timestamp_dt.strftime("%Y-%m-%dT%H-%M-%S.%fZ") + f"_{event_type}"
            image_filename = filename_base + ".jpg"
            image_path = os.path.join(self.save_path, image_filename)
            
            # Guarda a imagem JPG
            # Usamos uma qualidade JPEG ligeiramente mais alta para a imagem guardada
            cv2.imwrite(image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            
            # Prepara os dados para guardar no JSONL
            log_entry = event_data.copy() # Começa com os dados originais
            log_entry["image_file"] = image_filename # Adiciona o nome do ficheiro da imagem

            # Guarda a entrada no ficheiro JSONL (modo 'a' para append)
            # Usamos 'ensure_ascii=False' para suportar caracteres acentuados, se houver
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write('\n') # Adiciona nova linha para o formato JSONL
                
            logging.warning(f"*** ALERTA ARMAZENADO *** Tipo: {event_type}, Imagem: {image_filename}")

        except Exception as e:
            # Loga o erro específico do processamento/gravação
            logging.error(f"Falha ao processar e guardar evento: {e}", exc_info=True)

    def stop(self):
        """Sinaliza à thread para parar."""
        logging.info("A sinalizar paragem para o Gestor de Eventos...")
        self.running = False
        # (Opcional) Poderia adicionar um item especial na fila para desbloquear get() imediatamente
        # self.queue.put(None) 

