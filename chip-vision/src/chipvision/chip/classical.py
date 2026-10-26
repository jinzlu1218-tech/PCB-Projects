from __future__ import annotations

import cv2
import numpy as np

from ..config import Config
from ..geometry import (
    order_quad_clockwise,
    side_lengths,
)
from .types import PoseResult


class ClassicalChipDetector:
    """A label-free fallback detector based on dark-region geometry.

    It is intentionally conservative: the reference matcher is the primary
    method for this dataset, and this detector is only used when matching
    fails.  Candidate regions are scored by area, aspect ratio, rectangularity
    and fill ratio.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        section = config.data["chip"]["fallback"]
        self.enabled = bool(section.get("enabled", True))
        self.min_area_ratio = float(section.get("min_area_ratio", 0.02))
        self.max_area_ratio = float(section.get("max_area_ratio", 0.30))

    def detect(self, image: np.ndarray) -> PoseResult | None:
        if not self.enabled:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        height, width = gray.shape[:2]
        image_area = float(height * width)

        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        candidates: list[np.ndarray] = []

        # Dark package against a textured board.
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            8,
        )
        for mask in (otsu, adaptive):
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < self.min_area_ratio * image_area:
                    continue
                candidates.append(contour)

        # Edge contours as a second source.  They often survive when the
        # package and background have similar mean intensity.
        edges = cv2.Canny(blurred, 40, 120)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(contours)

        best: tuple[float, np.ndarray] | None = None
        for contour in candidates:
            area = float(cv2.contourArea(contour))
            if area < self.min_area_ratio * image_area or area > self.max_area_ratio * image_area:
                continue
            rectangle = cv2.minAreaRect(contour)
            (_center, (rw, rh), _angle) = rectangle
            long_side, short_side = max(rw, rh), min(rw, rh)
            if short_side < 20 or long_side < 50:
                continue
            aspect = long_side / short_side
            if not 1.5 <= aspect <= 3.6:
                continue
            box_area = rw * rh
            if box_area <= 0:
                continue
            fill = area / box_area
            if fill < 0.62:
                continue
            quad = cv2.boxPoints(rectangle)
            quad = order_quad_clockwise(quad)
            sides = side_lengths(quad)
            if abs(sides[0] - sides[2]) / max(sides[0], sides[2]) > 0.18:
                continue
            if abs(sides[1] - sides[3]) / max(sides[1], sides[3]) > 0.18:
                continue
            area_score = 1.0 - min(1.0, abs(np.log(max(area / image_area, 1e-9) / 0.14)))
            aspect_score = 1.0 - min(1.0, abs(aspect - 2.4) / 1.2)
            score = 0.45 * fill + 0.25 * area_score + 0.20 * aspect_score + 0.10 * min(1.0, area / 30000)
            if best is None or score > best[0]:
                best = (score, quad)

        if best is None:
            return None
        score, quad = best
        return PoseResult(
            corners=quad.astype(np.float64),
            confidence=float(np.clip(0.35 + 0.45 * score, 0.0, 0.9)),
            method="classical_dark_rect_fallback",
            metadata={"note": "fallback detector; inspect the overlay before trusting this result"},
        )
