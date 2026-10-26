from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .chip.classical import ClassicalChipDetector
from .chip.matcher import PoseEstimationError, ReferencePoseEstimator
from .chip.types import PoseResult
from .config import Config
from .geometry import polygon_mask
from .io import list_images, read_image, write_csv, write_image, write_json
from .text.recognizer import ChipTextRecognizer, TextResult
from .text.rectify import rectify_chip
from .visualize import draw_result


@dataclass
class ImageResult:
    name: str
    image: np.ndarray
    chip: PoseResult | None
    text: TextResult | None
    error: str | None = None


class ChipVisionPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.pose_estimator = ReferencePoseEstimator(config)
        self.fallback = ClassicalChipDetector(config)
        self.text_recognizer = ChipTextRecognizer(config)

    def _estimate_pose(self, image: np.ndarray) -> PoseResult:
        try:
            return self.pose_estimator.locate(image)
        except PoseEstimationError as primary_error:
            fallback = self.fallback.detect(image)
            if fallback is None:
                raise PoseEstimationError(
                    f"Reference matcher failed ({primary_error}); fallback also found no chip",
                    primary_error.diagnostics,
                ) from primary_error
            fallback.metadata["primary_failure"] = str(primary_error)
            fallback.metadata["primary_failure_diagnostics"] = primary_error.diagnostics
            return fallback

    def process_image(self, path: str | Path) -> ImageResult:
        image = read_image(path)
        name = Path(path).name
        try:
            pose = self._estimate_pose(image)
        except PoseEstimationError as exc:
            return ImageResult(name=name, image=image, chip=None, text=None, error=str(exc))

        canonical = self.config.data["chip"]["canonical"]
        width = int(canonical["width"])
        height = int(canonical["height"])
        crop, homography = rectify_chip(image, pose.corners, width, height)
        pose.metadata["canonical_size_px"] = [width, height]

        try:
            text_result = self.text_recognizer.read(crop, homography, image.shape[:2])
        except Exception as exc:  # noqa: BLE001 - OCR failure must not destroy the pose output
            text_result = TextResult(
                metadata={"enabled": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            return ImageResult(
                name=name,
                image=image,
                chip=pose,
                text=text_result,
                error=f"OCR failed: {exc}",
            )
        return ImageResult(name=name, image=image, chip=pose, text=text_result)

    def run(
        self,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        limit: int | None = None,
        ocr: bool | None = None,
        progress: bool = True,
    ) -> dict[str, Any]:
        if ocr is False:
            self.text_recognizer.enabled = False
        input_path = Path(input_dir) if input_dir is not None else self.config.input_dir
        output_path = Path(output_dir) if output_dir is not None else self.config.output_dir
        images = list_images(input_path)
        if limit is not None:
            images = images[: max(0, int(limit))]
        output_path.mkdir(parents=True, exist_ok=True)
        for subdir in ("overlays", "masks", "canonical", "ocr_input"):
            (output_path / subdir).mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        csv_rows: list[dict[str, Any]] = []
        for index, image_path in enumerate(images, start=1):
            if progress:
                print(f"[{index:02d}/{len(images):02d}] {image_path.name}", flush=True)
            result = self.process_image(image_path)
            if result.chip is None:
                records.append(
                    {
                        "image": result.name,
                        "status": "error",
                        "error": result.error,
                    }
                )
                csv_rows.append(
                    {
                        "image": result.name,
                        "status": "error",
                        "error": result.error or "",
                    }
                )
                continue

            image_h, image_w = result.image.shape[:2]
            chip_dict = result.chip.to_dict(image_shape=(image_h, image_w))
            text_dict = result.text.to_dict() if result.text is not None else {
                "full_text": "",
                "lines": [],
                "metadata": {},
            }
            record = {
                "image": result.name,
                "status": "error" if result.error else "ok",
                "error": result.error,
                "image_size": [image_w, image_h],
                "chip": chip_dict,
                "text": text_dict,
            }
            records.append(record)

            canonical = Path(output_path) / "canonical" / f"{image_path.stem}.png"
            crop, _ = rectify_chip(
                result.image,
                result.chip.corners,
                int(self.config.data["chip"]["canonical"]["width"]),
                int(self.config.data["chip"]["canonical"]["height"]),
            )
            write_image(canonical, crop)
            if result.text is not None and result.text.ocr_image is not None:
                write_image(Path(output_path) / "ocr_input" / f"{image_path.stem}.png", result.text.ocr_image)
            if self.config.data["outputs"].get("save_mask", True):
                mask = polygon_mask(
                    (image_h, image_w),
                    result.chip.corners,
                )
                write_image(Path(output_path) / "masks" / f"{image_path.stem}.png", mask)
            if self.config.data["outputs"].get("save_overlay", True):
                overlay = draw_result(
                    result.image,
                    result.chip,
                    result.text or TextResult(),
                    result.name,
                )
                write_image(Path(output_path) / "overlays" / f"{image_path.stem}.png", overlay)

            lengths = result.chip.side_lengths
            csv_rows.append(
                {
                    "image": result.name,
                    "status": record["status"],
                    "error": result.error or "",
                    "center_x_px": round(float(result.chip.center[0]), 3),
                    "center_y_px": round(float(result.chip.center[1]), 3),
                    "angle_deg": round(result.chip.angles[1], 3),
                    "angle_axis_deg": round(result.chip.angles[0], 3),
                    "long_px": round(float((lengths[0] + lengths[2]) / 2.0), 3),
                    "short_px": round(float((lengths[1] + lengths[3]) / 2.0), 3),
                    "pose_confidence": round(result.chip.confidence, 4),
                    "method": result.chip.method,
                    "full_text": (result.text.full_text.replace("\n", " | ") if result.text else ""),
                    "ocr_confidence": (
                        result.text.metadata.get("mean_confidence", "") if result.text else ""
                    ),
                }
            )

        predictions = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": str(self.config.root / "configs" / "default.yaml"),
            "images": records,
        }
        write_json(output_path / "predictions.json", predictions)
        write_json(output_path / "run_manifest.json", self._manifest(images, output_path))
        write_csv(
            output_path / "predictions.csv",
            csv_rows,
            [
                "image",
                "status",
                "error",
                "center_x_px",
                "center_y_px",
                "angle_deg",
                "angle_axis_deg",
                "long_px",
                "short_px",
                "pose_confidence",
                "method",
                "full_text",
                "ocr_confidence",
            ],
        )
        return {
            "num_images": len(images),
            "num_ok": sum(1 for record in records if record["status"] == "ok"),
            "num_errors": sum(1 for record in records if record["status"] != "ok"),
            "output_dir": str(output_path),
            "predictions": str(output_path / "predictions.json"),
        }

    def _manifest(self, images: list[Path], output_path: Path) -> dict[str, Any]:
        digest = hashlib.sha256(json.dumps(self.config.data, sort_keys=True).encode("utf-8")).hexdigest()
        try:
            import torch

            torch_version = torch.__version__
        except ImportError:
            torch_version = None
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torch": torch_version,
            "config_sha256": digest,
            "input_images": [str(path.name) for path in images],
            "output_dir": str(output_path),
        }
