from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geometry import angle_from_quad, axis_aligned_bbox, quad_center, side_lengths


@dataclass
class PoseResult:
    """A chip body pose in original image pixels."""

    corners: np.ndarray
    confidence: float
    method: str
    good_matches: int = 0
    inliers: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.corners = np.asarray(self.corners, dtype=np.float64).reshape(4, 2)

    @property
    def center(self) -> np.ndarray:
        return quad_center(self.corners)

    @property
    def side_lengths(self) -> np.ndarray:
        return side_lengths(self.corners)

    @property
    def angles(self) -> tuple[float, float]:
        return angle_from_quad(self.corners)

    def to_dict(self, image_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        axis_angle, signed_angle = self.angles
        lengths = self.side_lengths
        long_side = float((lengths[0] + lengths[2]) / 2.0)
        short_side = float((lengths[1] + lengths[3]) / 2.0)
        result = {
            "center_px": [round(float(v), 3) for v in self.center],
            "angle_deg": round(signed_angle, 3),
            "angle_axis_deg": round(axis_angle, 3),
            "angle_convention": (
                "long edge, counter-clockwise in a y-up frame; "
                "angle_deg in [-90,90), angle_axis_deg in [0,180)"
            ),
            "size_px": {
                "long": round(long_side, 3),
                "short": round(short_side, 3),
            },
            "corners_px": [[round(float(x), 3), round(float(y), 3)] for x, y in self.corners],
            "bbox_px": [round(v, 3) for v in axis_aligned_bbox(self.corners)],
            "mask_polygon_px": [
                [round(float(x), 3), round(float(y), 3)] for x, y in self.corners
            ],
            "confidence": round(float(self.confidence), 4),
            "method": self.method,
            "diagnostics": {
                "good_matches": self.good_matches,
                "inliers": self.inliers,
                **self.metadata,
            },
        }
        if image_shape is not None:
            result["image_size"] = [int(image_shape[1]), int(image_shape[0])]
        return result
