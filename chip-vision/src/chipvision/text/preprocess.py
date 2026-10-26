from __future__ import annotations

import cv2
import numpy as np


def make_ocr_image(
    chip_crop: np.ndarray,
    scale: int = 4,
    border_px: int = 24,
    clahe_clip: float = 2.5,
    clahe_grid: int = 8,
    invert: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Enhance and enlarge a rectified chip image for OCR.

    The package markings are low-contrast light grey letters on a dark mould
    compound.  Local contrast enhancement, unsharp masking and inversion make
    them closer to the dark-text-on-light-background domain used by generic
    OCR models.  A white border prevents the detector from clipping lines at
    the image edge.
    """
    if chip_crop.ndim == 3:
        gray = cv2.cvtColor(chip_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = chip_crop.copy()

    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_grid), int(clahe_grid)))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(enhanced, 1.65, blurred, -0.65, 0)
    if invert:
        sharpened = 255 - sharpened

    scale = max(1, int(scale))
    enlarged = cv2.resize(
        sharpened,
        (sharpened.shape[1] * scale, sharpened.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    border_px = max(0, int(border_px))
    ocr_image = cv2.copyMakeBorder(
        enlarged,
        border_px,
        border_px,
        border_px,
        border_px,
        cv2.BORDER_CONSTANT,
        value=255 if invert else 0,
    )
    metadata = {
        "scale": float(scale),
        "border_px": float(border_px),
        "inverted": float(bool(invert)),
    }
    return ocr_image, metadata
