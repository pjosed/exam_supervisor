"""
Estimación de pose usando MediaPipe.
Detecta orientación de cabeza y visibilidad de manos.
"""

import logging
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PoseData:
    track_id: int
    head_down: bool = False
    looking_left: bool = False
    looking_right: bool = False
    left_hand_visible: bool = False
    right_hand_visible: bool = False


class PoseEstimator:

    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        # MediaPipe Face Mesh para orientación de cabeza más precisa
        self._mp_face = mp.solutions.face_mesh
        self._face = self._mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        logger.info("PoseEstimator inicializado con FaceMesh.")

    def analyze(self, frame: np.ndarray, track_id: int) -> "PoseData | None":
        if frame.size == 0:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_data = PoseData(track_id=track_id)

        # --- Orientación de cabeza con FaceMesh (más preciso) ---
        face_results = self._face.process(rgb)
        if face_results.multi_face_landmarks:
            lm = face_results.multi_face_landmarks[0].landmark
            pose_data.head_down    = self._detect_head_down(lm, frame.shape)
            pose_data.looking_left  = self._detect_looking_left(lm)
            pose_data.looking_right = self._detect_looking_right(lm)

        # --- Manos con Pose ---
        pose_results = self._pose.process(rgb)
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            pose_data.left_hand_visible  = self._hand_visible(lm, "left")
            pose_data.right_hand_visible = self._hand_visible(lm, "right")

        return pose_data

    def _detect_head_down(self, landmarks, shape) -> bool:
        """
        Usa la punta de la nariz vs frente para detectar inclinación.
        En FaceMesh: 1=nariz, 10=frente, 152=mentón.
        """
        nose    = landmarks[1]
        forehead = landmarks[10]
        chin    = landmarks[152]
        # Si el mentón está muy cerca de la nariz en Y = cabeza inclinada
        face_height = abs(chin.y - forehead.y)
        nose_to_chin = abs(chin.y - nose.y)
        if face_height < 0.01:
            return False
        ratio = nose_to_chin / face_height
        return ratio < 0.35  # cabeza muy inclinada hacia abajo

    def _detect_looking_left(self, landmarks) -> bool:
        """
        Usa oreja izquierda vs nariz vs oreja derecha.
        FaceMesh: 234=oreja izq, 454=oreja der, 1=nariz.
        """
        nose      = landmarks[1]
        left_ear  = landmarks[234]
        right_ear = landmarks[454]
        face_width = abs(right_ear.x - left_ear.x)
        if face_width < 0.01:
            return False
        center_x = (left_ear.x + right_ear.x) / 2
        offset = (nose.x - center_x) / face_width
        return offset < -0.15

    def _detect_looking_right(self, landmarks) -> bool:
        nose      = landmarks[1]
        left_ear  = landmarks[234]
        right_ear = landmarks[454]
        face_width = abs(right_ear.x - left_ear.x)
        if face_width < 0.01:
            return False
        center_x = (left_ear.x + right_ear.x) / 2
        offset = (nose.x - center_x) / face_width
        return offset > 0.15

    def _hand_visible(self, landmarks, side: str) -> bool:
        lm_idx = self._mp_pose.PoseLandmark.LEFT_WRIST if side == "left" \
                 else self._mp_pose.PoseLandmark.RIGHT_WRIST
        wrist = landmarks[lm_idx]
        return wrist.visibility > 0.5

    def close(self):
        self._pose.close()
        self._face.close()