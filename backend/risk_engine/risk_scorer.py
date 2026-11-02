"""
Motor de puntuación de riesgo basado en comportamientos reales de copia en parciales.
Solo detecta patrones sospechosos en contexto de examen presencial.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field

from config.config_loader import config

logger = logging.getLogger(__name__)

LEVEL_GREEN  = "GREEN"
LEVEL_YELLOW = "YELLOW"
LEVEL_RED    = "RED"

MAX_SCORE = 100


@dataclass
class PersonRisk:
    """Estado de riesgo acumulado para una persona."""
    track_id: int
    score: float = 0.0
    level: str = LEVEL_GREEN
    last_update: float = field(default_factory=time.time)
    head_down_since: float = None
    head_was_down: bool = False
    hand_hidden_since: float = None
    lateral_events: deque = field(default_factory=lambda: deque(maxlen=50))
    last_lateral_direction: str = None
    vertical_transitions: deque = field(default_factory=lambda: deque(maxlen=20))
    last_head_state: bool = False


class RiskScorer:

    def __init__(self):
        self._persons: dict[int, PersonRisk] = {}
        self._yellow = config.get("risk", "thresholds", "yellow", default=30)
        self._red    = config.get("risk", "thresholds", "red",    default=60)
        logger.info("RiskScorer inicializado.")

    def update(self, pose_data) -> PersonRisk:
        tid = pose_data.track_id
        now = time.time()

        if tid not in self._persons:
            self._persons[tid] = PersonRisk(track_id=tid)

        p = self._persons[tid]
        elapsed = now - p.last_update
        p.last_update = now

        # Decay agresivo — baja 8pts/seg cuando no hay comportamiento sospechoso
        p.score = max(0.0, p.score - 8.0 * elapsed)

        self._check_lateral_looks(p, pose_data, now, elapsed)
        self._check_phone_in_lap(p, pose_data, now)
        self._check_hidden_hand_with_head_down(p, pose_data, now, elapsed)
        self._check_extreme_head_down(p, pose_data, now, elapsed)

        p.score = min(p.score, MAX_SCORE)
        p.level = self._calculate_level(p.score)
        return p

    def _check_lateral_looks(self, p: PersonRisk, pose_data, now: float, elapsed: float):
        looking_lateral = pose_data.looking_left or pose_data.looking_right
        direction = "left" if pose_data.looking_left else ("right" if pose_data.looking_right else None)

        if direction and direction != p.last_lateral_direction:
            p.lateral_events.append(now)
            p.last_lateral_direction = direction
        elif not looking_lateral:
            p.last_lateral_direction = None

        recent = [t for t in p.lateral_events if now - t < 15]
        count = len(recent)

        if count >= 4:
            points_per_sec = min(count * 2.0, 15)
            p.score += points_per_sec * elapsed

    def _check_phone_in_lap(self, p: PersonRisk, pose_data, now: float):
        current_down = pose_data.head_down

        if current_down != p.last_head_state:
            p.vertical_transitions.append(now)
            p.last_head_state = current_down

        recent = [t for t in p.vertical_transitions if now - t < 20]

        if len(recent) >= 4:
            p.score += 3.0
            logger.info(f"ID {p.track_id}: movimiento vertical repetitivo detectado")
            p.vertical_transitions.clear()

    def _check_hidden_hand_with_head_down(self, p: PersonRisk, pose_data, now: float, elapsed: float):
        hands_hidden = not pose_data.left_hand_visible and not pose_data.right_hand_visible
        head_down = pose_data.head_down

        if hands_hidden and head_down:
            if p.hand_hidden_since is None:
                p.hand_hidden_since = now
            duration = now - p.hand_hidden_since
            if duration > 3:
                p.score += 8.0 * elapsed
        else:
            p.hand_hidden_since = None

    def _check_extreme_head_down(self, p: PersonRisk, pose_data, now: float, elapsed: float):
        hands_visible = pose_data.left_hand_visible or pose_data.right_hand_visible
        head_down = pose_data.head_down

        if head_down and not hands_visible:
            if p.head_down_since is None:
                p.head_down_since = now
            duration = now - p.head_down_since
            if duration > 5:
                p.score += 4.0 * elapsed
        else:
            p.head_down_since = None

    def _calculate_level(self, score: float) -> str:
        if score >= self._red:
            return LEVEL_RED
        elif score >= self._yellow:
            return LEVEL_YELLOW
        return LEVEL_GREEN

    def get(self, track_id: int) -> PersonRisk | None:
        return self._persons.get(track_id)