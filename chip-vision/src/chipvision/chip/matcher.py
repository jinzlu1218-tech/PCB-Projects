from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..config import Config
from ..geometry import (
    map_points,
    polygon_area,
    side_lengths,
    signed_polygon_area,
)
from ..io import read_image
from .types import PoseResult


class PoseEstimationError(RuntimeError):
    """Raised when the reference matcher cannot produce a valid pose."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class ReferencePoseEstimator:
    """Locate the chip by matching a reference package template.

    All twenty images in this assignment come from the same board and package.
    A SIFT template match therefore gives a very precise six-degree-of-freedom
    alignment, while remaining robust to arbitrary in-plane rotation and mild
    perspective.  The homography is validated geometrically before it is
    accepted; the classical detector is used as a fallback.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        section = config.data["reference"]
        matcher = config.data["chip"]["matcher"]
        self.reference_path = config.reference_image
        if not self.reference_path.exists():
            raise FileNotFoundError(f"Reference image does not exist: {self.reference_path}")

        self.reference = read_image(self.reference_path)
        height, width = self.reference.shape[:2]
        x0, y0, x1, y1 = [round(v) for v in section["template_bbox"]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("reference.template_bbox is empty after clipping")
        self.template_bbox = (x0, y0, x1, y1)
        self.template = self.reference[y0:y1, x0:x1]
        self._gray_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        corners = np.asarray(section["body_corners_px"], dtype=np.float32)
        self.body_corners_ref = corners
        self.body_corners_template = corners - np.array([x0, y0], dtype=np.float32)
        self.reference_area = polygon_area(corners)
        ref_sides = side_lengths(corners)
        self.reference_long = float((ref_sides[0] + ref_sides[2]) / 2.0)
        self.reference_short = float((ref_sides[1] + ref_sides[3]) / 2.0)
        self.reference_signed_area = signed_polygon_area(corners)

        self.nfeatures = int(matcher.get("nfeatures", 3000))
        self.contrast_threshold = float(matcher.get("contrast_threshold", 0.015))
        self.ratio_test = float(matcher.get("ratio_test", 0.72))
        self.ransac_reproj = float(matcher.get("ransac_reproj_px", 3.0))
        self.min_good = int(matcher.get("min_good_matches", 8))
        self.min_inliers = int(matcher.get("min_inliers", 12))
        self.min_inlier_ratio = float(matcher.get("min_inlier_ratio", 0.20))
        self.min_area_ratio = float(matcher.get("min_area_ratio", 0.65))
        self.max_area_ratio = float(matcher.get("max_area_ratio", 1.35))
        self.min_edge_ratio = float(matcher.get("min_edge_ratio", 0.85))
        self.max_edge_ratio = float(matcher.get("max_edge_ratio", 1.15))

        self.sift = cv2.SIFT_create(
            nfeatures=self.nfeatures,
            contrastThreshold=self.contrast_threshold,
        )
        self.keypoints, self.descriptors = self.sift.detectAndCompute(self._gray_template, None)
        if self.descriptors is None or len(self.keypoints) < 10:
            raise PoseEstimationError(
                "The reference template has too few SIFT features",
                {"keypoints": len(self.keypoints) if self.keypoints else 0},
            )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def locate(self, image: np.ndarray) -> PoseResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None:
            raise PoseEstimationError("No SIFT features in the input image")
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [m for m, n in pairs if m.distance < self.ratio_test * n.distance]
        diagnostics: dict[str, Any] = {
            "keypoints": len(keypoints),
            "good_matches": len(good),
        }
        if len(good) < self.min_good:
            raise PoseEstimationError(
                f"Not enough good SIFT matches ({len(good)} < {self.min_good})", diagnostics
            )

        src = np.float32([self.keypoints[m.queryIdx].pt for m in good])
        dst = np.float32([keypoints[m.trainIdx].pt for m in good])
        homography, inlier_mask = cv2.findHomography(
            src,
            dst,
            cv2.RANSAC,
            self.ransac_reproj,
            maxIters=5000,
            confidence=0.999,
        )
        if homography is None or inlier_mask is None:
            raise PoseEstimationError("RANSAC homography estimation failed", diagnostics)
        inliers = int(inlier_mask.ravel().sum())
        inlier_ratio = inliers / max(len(good), 1)
        diagnostics.update({"inliers": inliers, "inlier_ratio": round(inlier_ratio, 4)})

        if inliers < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            raise PoseEstimationError(
                f"Homography is not supported by enough inliers ({inliers})", diagnostics
            )
        if np.linalg.det(homography[:2, :2]) <= 0:
            raise PoseEstimationError("Homography contains a reflection", diagnostics)

        corners = map_points(self.body_corners_template, homography)
        area = polygon_area(corners)
        area_ratio = area / max(self.reference_area, 1e-9)
        sides = side_lengths(corners)
        long_side = float((sides[0] + sides[2]) / 2.0)
        short_side = float((sides[1] + sides[3]) / 2.0)
        long_ratio = long_side / max(self.reference_long, 1e-9)
        short_ratio = short_side / max(self.reference_short, 1e-9)
        signed_area = signed_polygon_area(corners)
        diagnostics.update(
            {
                "area_px": round(area, 2),
                "area_ratio": round(area_ratio, 4),
                "long_ratio": round(long_ratio, 4),
                "short_ratio": round(short_ratio, 4),
            }
        )

        if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
            raise PoseEstimationError(
                f"Chip area ratio outside the accepted range: {area_ratio:.2f}", diagnostics
            )
        if not (self.min_edge_ratio <= long_ratio <= self.max_edge_ratio):
            raise PoseEstimationError(
                f"Chip long-edge ratio outside the accepted range: {long_ratio:.2f}", diagnostics
            )
        if not (self.min_edge_ratio <= short_ratio <= self.max_edge_ratio):
            raise PoseEstimationError(
                f"Chip short-edge ratio outside the accepted range: {short_ratio:.2f}", diagnostics
            )
        if signed_area * self.reference_signed_area <= 0:
            raise PoseEstimationError("Matched quadrilateral has the wrong winding order", diagnostics)

        image_height, image_width = gray.shape[:2]
        x, y = corners[:, 0], corners[:, 1]
        inside_fraction = float(
            np.mean((x >= -0.1 * image_width) & (x <= 1.1 * image_width)
                    & (y >= -0.1 * image_height) & (y <= 1.1 * image_height))
        )
        if inside_fraction < 1.0:
            raise PoseEstimationError("Matched chip is largely outside the image", diagnostics)

        confidence = (
            0.55 * min(1.0, inliers / 40.0)
            + 0.25 * min(1.0, inlier_ratio / 0.60)
            + 0.20 * max(0.0, 1.0 - abs(area_ratio - 1.0) / 0.35)
        )
        return PoseResult(
            corners=corners,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            method="sift_reference_homography",
            good_matches=len(good),
            inliers=inliers,
            metadata={
                "template_bbox": list(self.template_bbox),
                "reference": str(self.reference_path.relative_to(self.config.root))
                if self.reference_path.is_relative_to(self.config.root)
                else str(self.reference_path),
                "homography": homography.tolist(),
            },
        )
