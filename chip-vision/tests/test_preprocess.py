import numpy as np

from chipvision.text.preprocess import make_ocr_image
from chipvision.text.rectify import rectify_chip


def test_rectification_shape_and_homography_identity():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    corners = np.array([[10, 20], [150, 20], [150, 80], [10, 80]], dtype=np.float32)
    crop, homography = rectify_chip(image, corners, 140, 60)
    assert crop.shape == (60, 140, 3)
    mapped = homography @ np.array([0.0, 0.0, 1.0])
    mapped /= mapped[2]
    assert np.allclose(mapped[:2], [10, 20], atol=1e-4)


def test_ocr_image_shape_and_contrast():
    crop = np.full((40, 100, 3), 30, dtype=np.uint8)
    crop[15:22, 10:90] = 180
    ocr, metadata = make_ocr_image(crop, scale=3, border_px=5, invert=True)
    assert ocr.shape == ((40 * 3) + 10, (100 * 3) + 10)
    assert metadata["scale"] == 3
    assert ocr[2, 2] == 255
