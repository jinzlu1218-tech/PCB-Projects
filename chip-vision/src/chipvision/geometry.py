from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import cv2
import numpy as np

Point = tuple[float, float]


def as_points(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return an (N, 2) float32 point array."""
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected Nx2 points, got shape {array.shape}")
    return array


def polygon_area(points: Sequence[Sequence[float]] | np.ndarray) -> float:
    return float(abs(cv2.contourArea(as_points(points).reshape(-1, 1, 2))))


def signed_polygon_area(points: Sequence[Sequence[float]] | np.ndarray) -> float:
    pts = as_points(points)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def side_lengths(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    pts = as_points(points)
    return np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)


def normalize_axis_angle(angle_deg: float) -> float:
    """Normalize an undirected line angle to [0, 180)."""
    return float(angle_deg % 180.0)


def normalize_signed_angle(angle_deg: float) -> float:
    """Normalize an undirected line angle to (-90, 90]."""
    angle = normalize_axis_angle(angle_deg)
    if angle > 90.0:
        angle -= 180.0
    return float(angle)


def angle_from_quad(corners: Sequence[Sequence[float]] | np.ndarray) -> tuple[float, float]:
    """Estimate the long-edge angle of an ordered body quadrilateral.

    Returns ``(axis_angle, signed_angle)`` in degrees.  The axis angle is in
    [0, 180), the signed angle in (-90, 90].  Both are measured counter-
    clockwise in a conventional Cartesian frame; image coordinates are
    converted internally so a clockwise image rotation has a negative angle.
    """
    pts = as_points(corners)
    # Long edges are p0-p1 and p3-p2.  In image coordinates +y points down.
    vectors = [pts[1] - pts[0], pts[2] - pts[3]]
    vector = np.sum([v / (np.linalg.norm(v) + 1e-9) for v in vectors], axis=0)
    angle_image = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    # Convert to a y-up angle so the sign follows the usual mathematical
    # convention used in the course report.
    angle = -angle_image
    return normalize_axis_angle(angle), normalize_signed_angle(angle)


def quad_center(corners: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    return np.mean(as_points(corners), axis=0)


def axis_aligned_bbox(points: Sequence[Sequence[float]] | np.ndarray) -> list[float]:
    pts = as_points(points)
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    return [float(x0), float(y0), float(x1), float(y1)]


def polygon_mask(
    shape: tuple[int, int],
    corners: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Rasterize a filled polygon mask (uint8, 0/255)."""
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.round(as_points(corners)).astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def order_quad_clockwise(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    This is only used by the classical fallback detector; the reference
    matcher keeps the semantic reference-corner order instead.
    """
    pts = as_points(points)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    pts = pts[order]
    # Start at the point closest to the top-left in image coordinates.
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def homography_from_quad(
    corners: Sequence[Sequence[float]] | np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Homography mapping a canonical rectangle to an image quadrilateral."""
    src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dst = as_points(corners)
    return cv2.getPerspectiveTransform(src, dst)


def map_points(points: Sequence[Sequence[float]] | np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = as_points(points).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def clip_points(points: Iterable[Sequence[float]], width: int, height: int) -> list[list[float]]:
    clipped = []
    for x, y in points:
        clipped.append([float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))])
    return clipped
