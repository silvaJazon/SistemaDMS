# Documentação: Ponto de entrada principal para o serviço de IA.
# Versão inicial para testes do fluxo de build e deploy.

import cv2
import time

def main():
    """
    Função principal que imprime a versão do OpenCV e uma mensagem de teste.
    """
    print(">>> Serviço de IA do FluxoAI a iniciar...")
    print(f">>> Versão do OpenCV instalada: {cv2.__version__}")
    print(">>> O contentor está a funcionar corretamente!")
    
    # Mantém o script a correr para podermos inspecionar o contentor se necessário.
    print(">>> A entrar em loop infinito. Pressione Ctrl+C para sair dos logs.")
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()
