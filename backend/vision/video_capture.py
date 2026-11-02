"""
Módulo de captura de video desde cámara o archivo.
Responsabilidad única: abrir la fuente de video y entregar frames.
"""

import cv2
import logging
from typing import Generator
import numpy as np

from config.config_loader import config

logger = logging.getLogger(__name__)


class VideoCapture:
    """Gestiona la captura de frames desde una fuente de video."""

    def __init__(self, source: int | str | None = None):
        self._source = source if source is not None else config.get("camera", "source", default=0)
        self._width = config.get("camera", "width", default=640)
        self._height = config.get("camera", "height", default=480)
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Abre la fuente de video y configura resolución."""
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {self._source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        logger.info(f"Cámara abierta: fuente={self._source}, resolución={self._width}x{self._height}")

    def read_frame(self) -> np.ndarray | None:
        """Lee un frame. Retorna None si falla."""
        if self._cap is None:
            raise RuntimeError("La cámara no está abierta. Llama a open() primero.")
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("No se pudo leer frame de la cámara.")
            return None
        return frame

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generador que entrega frames continuamente hasta que se detenga."""
        self.open()
        try:
            while True:
                frame = self.read_frame()
                if frame is None:
                    break
                yield frame
        finally:
            self.release()

    def release(self) -> None:
        """Libera el recurso de la cámara."""
        if self._cap:
            self._cap.release()
            logger.info("Cámara liberada.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()