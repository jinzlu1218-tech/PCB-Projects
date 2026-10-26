from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_image(path: str | Path) -> np.ndarray:
    """Read an image, including paths that contain non-ASCII characters.

    ``cv2.imread`` uses the ANSI code page on Windows and fails for paths such
    as the Chinese directory used by this assignment, so the file is read via
    NumPy and decoded in memory instead.
    """
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise OSError(f"Could not read image file: {path}") from exc
    if data.size == 0:
        raise OSError(f"Image file is empty: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not decode image: {path}")
    return image


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image safely for Unicode paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise OSError(f"Could not encode image for: {path}")
    encoded.tofile(str(path))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
