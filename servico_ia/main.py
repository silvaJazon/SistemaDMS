import cv2
import time
import os
import sys

def main():
    """Função principal que captura vídeo de uma câmara local ou stream RTSP."""

    print(">>> Serviço de IA do FluxoAI a iniciar (Fase 2: Fonte Flexível)...")
    print(f">>> Versão do OpenCV: {cv2.__version__}")

    # Determina a fonte do vídeo a partir da variável de ambiente
    video_source = os.environ.get('VIDEO_SOURCE')

    if video_source is None:
        print("!!! ERRO: A variável de ambiente VIDEO_SOURCE não foi definida.", flush=True)
        print("!!! Exemplo de uso: -e VIDEO_SOURCE=0 (para câmara USB) ou -e VIDEO_SOURCE=\"rtsp://...\" (para DVR)", flush=True)
        sys.exit(1) # Termina o script com erro

    is_rtsp = video_source.startswith("rtsp://")

    # Tenta conectar-se à fonte de vídeo
    cap = None
    retry_delay = 5 # segundos
    while cap is None or not cap.isOpened():
        try:
            if is_rtsp:
                print(f">>> A tentar conectar-se ao stream de rede: {video_source}", flush=True)
                # Tenta abrir o stream RTSP
                # A variável de ambiente OPENCV_FFMPEG_CAPTURE_OPTIONS pode ser usada para definir timeouts, ex: "rtsp_transport;tcp|timeout;5000"
                cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
            else:
                # Tenta abrir a câmara local (convertendo para inteiro)
                camera_index = int(video_source)
                print(f">>> A tentar conectar-se à câmara local no índice: {camera_index}", flush=True)
                cap = cv2.VideoCapture(camera_index)

            if cap is None or not cap.isOpened():
                raise ValueError("Falha ao abrir VideoCapture")

            print(">>> Fonte de vídeo conectada com sucesso!", flush=True)

        except Exception as e:
            print(f"!!! ERRO ao abrir a fonte de vídeo: {e}", flush=True)
            print(f">>> A tentar novamente em {retry_delay} segundos...", flush=True)
            cap = None # Garante que cap é None para o loop continuar
            time.sleep(retry_delay)

    frame_count = 0
    start_time = time.time()

    print(">>> A iniciar loop de captura...", flush=True)
    while True:
        # Lê um frame da câmara/stream
        ret, frame = cap.read()

        # Se ret for False, significa que houve um erro ou o stream terminou
        if not ret:
            print("!!! ERRO: Não foi possível ler o frame. O stream pode ter sido perdido.", flush=True)
            # Liberta o objeto VideoCapture
            cap.release()
            cap = None
            print(">>> A tentar reconectar...", flush=True)
            # Volta ao início do loop para tentar reconectar
            while cap is None or not cap.isOpened():
                 try:
                    if is_rtsp:
                        print(f">>> A tentar conectar-se ao stream de rede: {video_source}", flush=True)
                        cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
                    else:
                        camera_index = int(video_source)
                        print(f">>> A tentar conectar-se à câmara local no índice: {camera_index}", flush=True)
                        cap = cv2.VideoCapture(camera_index)

                    if cap is None or not cap.isOpened():
                         raise ValueError("Falha ao abrir VideoCapture")

                    print(">>> Fonte de vídeo reconectada com sucesso!", flush=True)
                    start_time = time.time() # Reinicia o tempo para cálculo de FPS
                    frame_count = 0 # Reinicia a contagem de frames

                 except Exception as e:
                    print(f"!!! ERRO ao reconectar: {e}", flush=True)
                    print(f">>> A tentar novamente em {retry_delay} segundos...", flush=True)
                    cap = None
                    time.sleep(retry_delay)
            continue # Volta ao início do loop while True

        frame_count += 1

        # Mostra o progresso a cada 50 frames para não sobrecarregar o log
        if frame_count % 50 == 0:
            height, width, _ = frame.shape
            print(f">>> Captura a funcionar! Frame {frame_count} | Resolução: {width}x{height}", flush=True)

        # --- Área para o nosso código de IA futuro ---
        # Exemplo: detetar pessoas, desenhar caixas, etc.
        # results = model.predict(frame)
        # --- Fim da área de IA ---

        # Pequena pausa para evitar uso excessivo da CPU (opcional)
        # time.sleep(0.01)

    # Liberta o objeto VideoCapture quando o loop terminar (teoricamente nunca neste caso)
    # print(">>> A libertar recursos...", flush=True)
    # cap.release()

if __name__ == "__main__":
    main()

