"""
Detector de personas usando YOLOv8.
Responsabilidad: recibir un frame y retornar lista de detecciones.
"""

import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from config.config_loader import config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Representa una persona detectada en el frame."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int


class PersonDetector:
    """Detecta personas en frames usando YOLOv8."""

    def __init__(self):
        model_path = config.get("detection", "model", default="yolov8n.pt")
        self._confidence = config.get("detection", "confidence", default=0.5)
        self._classes = config.get("detection", "classes", default=[0])
        logger.info(f"Cargando modelo YOLO: {model_path}")
        self._model = YOLO(model_path)
        logger.info("Modelo YOLO cargado correctamente.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detecta personas en un frame.
        Retorna lista de Detection con bbox, confianza y clase.
        """
        results = self._model(
            frame,
            conf=self._confidence,
            classes=self._classes,
            verbose=False
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id
                ))

        return detections