from pathlib import Path

import numpy as np
import pytest

from chipvision.chip.matcher import ReferencePoseEstimator
from chipvision.config import Config
from chipvision.io import read_image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.load(ROOT / "configs" / "default.yaml", root=ROOT)


def test_reference_image_matches_its_own_annotation(config):
    estimator = ReferencePoseEstimator(config)
    image = read_image(ROOT / "input" / "board-01.png")
    pose = estimator.locate(image)
    expected = np.asarray(config.data["reference"]["body_corners_px"], dtype=float)
    error = np.sqrt(np.mean(np.sum((pose.corners - expected) ** 2, axis=1)))
    assert error < 1.0
    assert pose.confidence > 0.9
    assert pose.inliers >= 50


def test_all_twenty_images_are_localized(config):
    estimator = ReferencePoseEstimator(config)
    paths = sorted((ROOT / "input").glob("board-*.png"))
    assert len(paths) == 20
    for path in paths:
        image = read_image(path)
        pose = estimator.locate(image)
        assert pose.confidence > 0.4, path.name
        assert np.isfinite(pose.corners).all(), path.name
        height, width = image.shape[:2]
        assert (pose.corners[:, 0] > -width * 0.1).all(), path.name
        assert (pose.corners[:, 0] < width * 1.1).all(), path.name
        assert (pose.corners[:, 1] > -height * 0.1).all(), path.name
        assert (pose.corners[:, 1] < height * 1.1).all(), path.name
