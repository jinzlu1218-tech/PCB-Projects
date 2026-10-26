from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .chip.types import PoseResult
from .text.recognizer import TextResult


def _as_int_polygon(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    return np.round(np.asarray(points, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)


def draw_result(
    image: np.ndarray,
    pose: PoseResult,
    text_result: TextResult,
    image_name: str,
) -> np.ndarray:
    canvas = image.copy()
    overlay = canvas.copy()

    chip_polygon = _as_int_polygon(pose.corners)
    cv2.fillPoly(overlay, [chip_polygon], (40, 180, 40))
    canvas = cv2.addWeighted(overlay, 0.13, canvas, 0.87, 0)
    cv2.polylines(canvas, [chip_polygon], True, (40, 220, 40), 2, cv2.LINE_AA)

    center = tuple(np.round(pose.center).astype(int))
    cv2.drawMarker(canvas, center, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    _angle_deg, signed_deg = pose.angles
    cv2.putText(
        canvas,
        f"pose {signed_deg:+.1f} deg",
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for line in text_result.lines:
        polygon = _as_int_polygon(line.polygon_px)
        cv2.polylines(canvas, [polygon], True, (255, 220, 0), 1, cv2.LINE_AA)
        x, y = polygon.reshape(-1, 2).min(axis=0)
        cv2.putText(
            canvas,
            f"{line.text} {line.confidence:.2f}",
            (int(x), max(12, int(y) - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 220, 0),
            1,
            cv2.LINE_AA,
        )

    header = f"{image_name} | {pose.method} | conf={pose.confidence:.2f}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        header,
        (7, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return canvas
