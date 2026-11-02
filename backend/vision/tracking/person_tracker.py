"""
Tracker de personas usando el tracker integrado de YOLOv8 (ByteTrack).
Responsabilidad: mantener IDs persistentes entre frames.
"""

import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from config.config_loader import config

logger = logging.getLogger(__name__)


@dataclass
class TrackedPerson:
    """Persona detectada y rastreada con ID persistente."""
    track_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


class PersonTracker:
    """Detecta y rastrea personas con IDs persistentes usando ByteTrack."""

    def __init__(self):
        model_path = config.get("detection", "model", default="yolov8n.pt")
        self._confidence = config.get("detection", "confidence", default=0.5)
        self._classes = config.get("detection", "classes", default=[0])
        logger.info(f"Cargando modelo para tracking: {model_path}")
        self._model = YOLO(model_path)

    def update(self, frame: np.ndarray) -> list[TrackedPerson]:
        """
        Procesa un frame y retorna personas con IDs persistentes.
        """
        results = self._model.track(
            frame,
            conf=self._confidence,
            classes=self._classes,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        tracked = []
        for result in results:
            if result.boxes.id is None:
                continue
            for box, track_id in zip(result.boxes, result.boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                tracked.append(TrackedPerson(
                    track_id=int(track_id),
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0])
                ))

        return tracked