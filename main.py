"""
Punto de entrada principal del sistema.
Fase 4: cámara + pose + riesgo + escritura de estado para dashboard.
"""

import json
import logging
import cv2
from datetime import datetime
from pathlib import Path

from backend.vision.video_capture import VideoCapture
from backend.vision.tracking.person_tracker import PersonTracker
from backend.vision.pose.pose_estimator import PoseEstimator
from backend.risk_engine.risk_scorer import RiskScorer, LEVEL_GREEN, LEVEL_YELLOW, LEVEL_RED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

STATE_FILE = Path("runtime_state.json")

COLORS = {
    LEVEL_GREEN:  (0, 255, 0),
    LEVEL_YELLOW: (0, 215, 255),
    LEVEL_RED:    (0, 0, 255),
}


def build_alerts(pose, risk) -> list[str]:
    alerts = []
    if pose.head_down:
        alerts.append("Cabeza inclinada hacia abajo")
    if pose.looking_left:
        alerts.append("Mirando a la derecha repetidamente")
    if pose.looking_right:
        alerts.append("Mirando a la izquierda repetidamente")
    if not pose.left_hand_visible and not pose.right_hand_visible:
        alerts.append("Manos no visibles")
    return alerts


def write_state(persons_data: list):
    state = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "persons": persons_data
    }
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False),
        encoding="utf-8"
    )


def draw_info(frame, tracked_persons, pose_results, risk_scorer):
    for person in tracked_persons:
        x1, y1, x2, y2 = person.bbox
        tid = person.track_id
        pose = pose_results.get(tid)
        risk = risk_scorer.get(tid)

        level = risk.level if risk else LEVEL_GREEN
        color = COLORS[level]
        score = risk.score if risk else 0.0

        indicators = []
        if pose:
            if pose.head_down:
                indicators.append("CABEZA ABAJO")
            if pose.looking_left:
                indicators.append("MIRA DER")
            if pose.looking_right:
                indicators.append("MIRA IZQ")
            if pose.left_hand_visible or pose.right_hand_visible:
                indicators.append("MANOS VISIBLES")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tid} | {level} | {score:.0f}pts",
                    (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for i, text in enumerate(reversed(indicators)):
            y_text = y2 - 8 - i * 18
            if y_text > y1:
                cv2.putText(frame, text, (x1 + 4, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return frame


def main():
    logger.info("Iniciando sistema de supervisión...")
    tracker   = PersonTracker()
    estimator = PoseEstimator()
    scorer    = RiskScorer()
    capture   = VideoCapture()
    frame_count = 0

    with capture:
        for frame in capture.frames():
            tracked_persons = tracker.update(frame)

            pose_results = {}
            persons_data = []

            for person in tracked_persons:
                x1, y1, x2, y2 = person.bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                pose = estimator.analyze(crop, person.track_id)
                if pose:
                    pose_results[person.track_id] = pose
                    risk = scorer.update(pose)
                    alerts = build_alerts(pose, risk)
                    persons_data.append({
                        "track_id": person.track_id,
                        "level":    risk.level,
                        "score":    round(risk.score, 1),
                        "alerts":   alerts
                    })

            # Escribir estado cada 30 frames (~1 segundo)
            frame_count += 1
            if frame_count % 30 == 0:
                write_state(persons_data)

            frame = draw_info(frame, tracked_persons, pose_results, scorer)
            cv2.imshow("Exam Supervisor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Sistema detenido por el usuario.")
                break

    estimator.close()
    cv2.destroyAllWindows()
    STATE_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()