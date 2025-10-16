# Documentação: Ponto de entrada principal para o serviço de IA.
# Versão 1.1: Adicionado flush=True aos prints para visualização em tempo real.

import cv2
import time

def main():
    """
    Função principal que imprime a versão do OpenCV e uma mensagem de teste.
    """
    # Adicionamos flush=True para garantir que a mensagem aparece imediatamente.
    print(">>> Serviço de IA do FluxoAI a iniciar...", flush=True)
    print(f">>> Versão do OpenCV instalada: {cv2.__version__}", flush=True)
    print(">>> O contentor está a funcionar corretamente!", flush=True)
    
    # Mantém o script a correr para podermos inspecionar o contentor se necessário.
    print(">>> A entrar em loop infinito. Pressione Ctrl+C para sair dos logs.", flush=True)
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()