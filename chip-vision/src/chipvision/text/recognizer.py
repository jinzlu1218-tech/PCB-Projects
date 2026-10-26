from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Config
from ..geometry import map_points
from .lexicon import LexiconCorrector
from .preprocess import make_ocr_image


@dataclass
class TextLine:
    text: str
    confidence: float
    polygon_canonical: np.ndarray
    polygon_px: np.ndarray
    raw_text: str | None = None
    corrected: bool = False
    band_index: int = 0


@dataclass
class TextResult:
    lines: list[TextLine] = field(default_factory=list)
    full_text: str = ""
    ocr_image: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_text": self.full_text,
            "lines": [
                {
                    "text": line.text,
                    "raw_text": line.raw_text if line.raw_text is not None else line.text,
                    "corrected": bool(line.corrected),
                    "confidence": round(float(line.confidence), 4),
                    "polygon_px": [
                        [round(float(x), 3), round(float(y), 3)] for x, y in line.polygon_px
                    ],
                }
                for line in self.lines
            ],
            "metadata": self.metadata,
        }


class ChipTextRecognizer:
    """Read the laser markings on the rectified chip surface.

    The chip is first rectified.  Because the three marking rows have a stable
    vertical layout, each row is passed to EasyOCR separately.  This avoids the
    text detector merging the part number and the ``JAPAN 2CB3`` row on
    low-contrast or strongly rotated samples.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        section = config.data["ocr"]
        self.enabled = bool(section.get("enabled", True))
        self.scale = int(section.get("scale", 4))
        self.border_px = int(section.get("border_px", 24))
        self.clahe_clip = float(section.get("clahe_clip", 2.5))
        self.clahe_grid = int(section.get("clahe_grid", 8))
        self.allowlist = str(section.get("allowlist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ."))
        self.min_confidence = float(section.get("min_confidence", 0.10))
        self.min_box_height = float(section.get("min_box_height_px", 4))
        self.invert = bool(section.get("invert", True))
        self.paragraph = bool(section.get("paragraph", False))
        self.use_gpu = bool(section.get("use_gpu", False))
        self.download_enabled = bool(section.get("download_enabled", True))
        self.model_storage_dir = config.model_storage_dir
        self.languages = list(section.get("languages", ["en"]))
        self.line_bands = [
            (float(band[0]), float(band[1])) for band in section.get("line_bands", [[0.18, 0.41], [0.40, 0.61], [0.60, 0.80]])
        ]
        for start, end in self.line_bands:
            if not (0.0 <= start < end <= 1.0):
                raise ValueError(f"Invalid OCR line band: {(start, end)}")
        self.corrector = LexiconCorrector(
            list(section.get("lexicon", [])),
            max_distance=int(section.get("lexicon_max_distance", 2)),
            min_similarity=float(section.get("lexicon_min_similarity", 0.45)),
        )
        self._reader = None

    def _lazy_reader(self):
        if self._reader is not None:
            return self._reader
        try:
            import easyocr
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("EasyOCR is not installed. Run: pip install -e .") from exc

        import torch

        self.model_storage_dir.mkdir(parents=True, exist_ok=True)
        gpu = self.use_gpu and torch.cuda.is_available()
        self._reader = easyocr.Reader(
            self.languages,
            gpu=gpu,
            model_storage_directory=str(self.model_storage_dir),
            download_enabled=self.download_enabled,
            verbose=False,
        )
        return self._reader

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\n", " ").strip()
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" .,:;|_")
        return text

    def _read_strip(self, reader, strip: np.ndarray) -> list[tuple[Any, str, float]]:
        ocr_image, _ = make_ocr_image(
            strip,
            scale=self.scale,
            border_px=self.border_px,
            clahe_clip=self.clahe_clip,
            clahe_grid=self.clahe_grid,
            invert=self.invert,
        )
        return reader.readtext(
            ocr_image,
            detail=1,
            paragraph=self.paragraph,
            allowlist=self.allowlist,
            text_threshold=0.45,
            low_text=0.30,
            link_threshold=0.30,
            mag_ratio=1.0,
            canvas_size=2560,
            min_size=4,
            decoder="greedy",
        )

    def read(
        self,
        chip_crop: np.ndarray,
        homography_canonical_to_image: np.ndarray,
        image_shape: tuple[int, int],
    ) -> TextResult:
        if not self.enabled:
            return TextResult(metadata={"enabled": False})

        # Keep a full debug image as well as reading row strips.
        full_ocr_image, prep_meta = make_ocr_image(
            chip_crop,
            scale=self.scale,
            border_px=self.border_px,
            clahe_clip=self.clahe_clip,
            clahe_grid=self.clahe_grid,
            invert=self.invert,
        )
        reader = self._lazy_reader()
        image_height, image_width = image_shape
        crop_height, crop_width = chip_crop.shape[:2]
        lines: list[TextLine] = []
        raw_count = 0
        corrections: list[dict[str, Any]] = []

        for band_index, (start_ratio, end_ratio) in enumerate(self.line_bands):
            y0 = int(np.floor(start_ratio * crop_height))
            y1 = int(np.ceil(end_ratio * crop_height))
            y0 = max(0, min(crop_height - 1, y0))
            y1 = max(y0 + 1, min(crop_height, y1))
            strip = chip_crop[y0:y1]
            raw_results = self._read_strip(reader, strip)
            raw_count += len(raw_results)

            for box, text, confidence in raw_results:
                cleaned = self._clean_text(str(text))
                if not cleaned or float(confidence) < self.min_confidence:
                    continue
                if not re.search(r"[A-Z0-9]", cleaned):
                    continue
                points = np.asarray(box, dtype=np.float64)
                if len(points) != 4:
                    continue
                canonical = points.copy()
                canonical -= np.array([self.border_px, self.border_px], dtype=np.float64)
                canonical /= float(self.scale)
                canonical[:, 1] += y0  # restore absolute canonical y
                if np.ptp(canonical[:, 1]) < self.min_box_height:
                    continue
                if canonical[:, 0].max() < 0 or canonical[:, 0].min() > crop_width:
                    continue
                correction = self.corrector.correct(cleaned)
                corrected_text = correction.text
                if correction.corrected:
                    corrections.append(
                        {
                            "raw": correction.raw_text,
                            "corrected": corrected_text,
                            "distance": correction.distance,
                            "similarity": round(correction.similarity, 4),
                        }
                    )
                polygon_px = map_points(canonical, homography_canonical_to_image)
                polygon_px[:, 0] = np.clip(polygon_px[:, 0], 0, image_width - 1)
                polygon_px[:, 1] = np.clip(polygon_px[:, 1], 0, image_height - 1)
                lines.append(
                    TextLine(
                        text=corrected_text,
                        confidence=float(confidence),
                        polygon_canonical=canonical,
                        polygon_px=polygon_px,
                        raw_text=cleaned,
                        corrected=correction.corrected,
                        band_index=band_index,
                    )
                )

        # Deduplicate detections that straddle overlapping bands.
        deduplicated: list[TextLine] = []
        for line in sorted(lines, key=lambda item: item.confidence, reverse=True):
            keep = True
            for existing in deduplicated:
                x0 = max(line.polygon_canonical[:, 0].min(), existing.polygon_canonical[:, 0].min())
                x1 = min(line.polygon_canonical[:, 0].max(), existing.polygon_canonical[:, 0].max())
                y0 = max(line.polygon_canonical[:, 1].min(), existing.polygon_canonical[:, 1].min())
                y1 = min(line.polygon_canonical[:, 1].max(), existing.polygon_canonical[:, 1].max())
                if x1 > x0 and y1 > y0:
                    inter = (x1 - x0) * (y1 - y0)
                    area_a = np.ptp(line.polygon_canonical[:, 0]) * np.ptp(line.polygon_canonical[:, 1])
                    area_b = np.ptp(existing.polygon_canonical[:, 0]) * np.ptp(existing.polygon_canonical[:, 1])
                    if inter / max(min(area_a, area_b), 1e-6) > 0.5:
                        keep = False
                        break
            if keep:
                deduplicated.append(line)
        lines = sorted(
            deduplicated,
            key=lambda item: (
                item.band_index,
                float(np.mean(item.polygon_canonical[:, 0])),
            ),
        )
        full_text = "\n".join(line.text for line in lines)
        metadata = {
            "enabled": True,
            "raw_detections": raw_count,
            "kept_detections": len(lines),
            "mean_confidence": round(
                float(np.mean([line.confidence for line in lines])) if lines else 0.0, 4
            ),
            "line_bands": [[start, end] for start, end in self.line_bands],
            "corrections": corrections,
            **prep_meta,
        }
        return TextResult(lines=lines, full_text=full_text, ocr_image=full_ocr_image, metadata=metadata)
