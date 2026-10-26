import math

import numpy as np

from chipvision.geometry import (
    angle_from_quad,
    normalize_axis_angle,
    normalize_signed_angle,
    polygon_area,
    polygon_mask,
)


def test_angle_horizontal():
    corners = [[0, 0], [100, 0], [100, 50], [0, 50]]
    axis, signed = angle_from_quad(corners)
    assert abs(axis) < 1e-6
    assert abs(signed) < 1e-6


def test_angle_rotation_is_counter_clockwise_in_y_up_frame():
    angle = math.radians(30.0)
    # Image coordinates are y-down, so a +30 degree mathematical rotation is
    # a -30 degree rotation in the image.
    rotation = np.array(
        [[math.cos(-angle), -math.sin(-angle)], [math.sin(-angle), math.cos(-angle)]]
    )
    base = np.array([[-50, -25], [50, -25], [50, 25], [-50, 25]], dtype=float)
    corners = base @ rotation.T
    axis, signed = angle_from_quad(corners)
    assert abs(axis - 30.0) < 1e-6
    assert abs(signed - 30.0) < 1e-6


def test_vertical_long_edge():
    corners = [[50, 0], [50, 100], [0, 100], [0, 0]]
    axis, signed = angle_from_quad(corners)
    assert abs(axis - 90.0) < 1e-6
    assert abs(signed - 90.0) < 1e-6


def test_angle_normalization():
    assert normalize_axis_angle(-10) == 170
    assert normalize_axis_angle(190) == 10
    assert normalize_signed_angle(170) == -10
    assert normalize_signed_angle(-170) == 10


def test_polygon_area_and_mask():
    corners = np.array([[10, 10], [30, 10], [30, 20], [10, 20]])
    assert polygon_area(corners) == 200
    mask = polygon_mask((40, 40), corners)
    # OpenCV rasterizes polygon boundaries inclusively, so the pixel count is
    # slightly larger than the geometric area (21 x 11 instead of 20 x 10).
    assert 200 <= int(mask.sum()) // 255 <= 231
