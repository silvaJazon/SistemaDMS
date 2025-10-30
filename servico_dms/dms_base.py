# Documentação: Classe Base Abstrata para Monitores DMS
# Define a interface que todos os backends de deteção (Dlib, MediaPipe, etc.)
# devem implementar.

import abc
import numpy as np
import logging
import threading # (NOVO)

class BaseMonitor(abc.ABC):
    """
    Interface abstrata para um monitor de condutor.
    Define os métodos essenciais que o app.py espera que existam.
    """

    @abc.abstractmethod
    # ================== ALTERAÇÃO (Passar o stop_event) ==================
    def __init__(self, frame_size, stop_event: threading.Event, default_settings: dict = None):
        """
        Inicializa o monitor.
        :param frame_size: Uma tupla (height, width) do frame de entrada.
        :param stop_event: O evento global para sinalizar o encerramento.
        :param default_settings: (Opcional) Um dict com os padrões (ear_threshold, etc.)
        """
        self.frame_height, self.frame_width = frame_size
        self.stop_event = stop_event # (NOVO)
        if default_settings is None: 
            default_settings = {}
        self.default_settings = default_settings 
        # ===================================================================
        # O log real virá da subclasse (ex: "Inicializando DlibMonitor...")

    @abc.abstractmethod
    def process_frame(self, frame: np.ndarray, gray: np.ndarray):
        """
        Processa um único frame para detetar sonolência, distração, etc.
        
        :param frame: O frame de vídeo original (BGR).
        :param gray: O frame de vídeo em escala de cinza.
        :return: Uma tupla (processed_frame, events_list, status_data)
                 - processed_frame: O frame com anotações de depuração (olhos, rosto, etc.)
                 - events_list: Uma lista de dicionários de eventos (ex: {"type": "SONOLENCIA", ...})
                 - status_data: Um dicionário com métricas atuais (ex: {"ear": 0.25, ...})
        """
        pass

    @abc.abstractmethod
    def update_settings(self, settings: dict) -> bool:
        """
        Atualiza as configurações do monitor (limiares, etc.) em tempo real.
        
        :param settings: Um dicionário com as novas configurações.
        :return: True se a atualização foi bem-sucedida, False caso contrário.
        """
        pass

    @abc.abstractmethod
    def get_settings(self) -> dict:
        """
        Obtém as configurações atuais do monitor.
        
        :return: Um dicionário com as configurações atuais.
        """
        pass