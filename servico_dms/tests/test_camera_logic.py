import pytest
import cv2
import sys
import os

# (CORRIGIDO) Adiciona o diretório raiz '/app' ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from camera_thread import CameraThread  # noqa: E402


@pytest.fixture
def camera_thread_instance():
    """Cria uma instância mock (não conectada) do CameraThread."""
    # (Não queremos que ele tente conectar à câmara nos testes)
    # Passamos um stop_event simulado
    class MockEvent:
        def is_set(self):
            return False

        def wait(self, timeout=None):
            pass

    # Suprimimos a conexão real sobrescrevendo 'connect_camera'
    CameraThread.connect_camera = lambda self: None

    thread = CameraThread(
        video_source_str="0",
        frame_width=640,
        frame_height=480,
        rotation_degrees=0,
        stop_event=MockEvent()
    )
    return thread


def test_get_rotation_code(camera_thread_instance):
    """
    Testa a lógica interna de conversão de graus para códigos OpenCV.
    """
    ct = camera_thread_instance

    assert ct._get_rotation_code(0) is None
    assert ct._get_rotation_code(90) == cv2.ROTATE_90_CLOCKWISE
    assert ct._get_rotation_code(180) == cv2.ROTATE_180
    assert ct._get_rotation_code(270) == cv2.ROTATE_90_COUNTERCLOCKWISE
    assert ct._get_rotation_code(360) is None  # 360 é o mesmo que 0
    assert ct._get_rotation_code("90") == cv2.ROTATE_90_CLOCKWISE
    assert ct._get_rotation_code("invalid") is None