from __future__ import annotations

import cv2
import numpy as np

from ..geometry import homography_from_quad


def rectify_chip(
    image: np.ndarray,
    corners: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the chip body to a canonical horizontal rectangle.

    Returns ``(crop_bgr, H_canonical_to_image)``.  The homography is needed to
    map OCR boxes in the rectified image back to original pixels.
    """
    homography = homography_from_quad(corners, width, height)
    inverse = np.linalg.inv(homography)
    crop = cv2.warpPerspective(
        image,
        inverse,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, homography
