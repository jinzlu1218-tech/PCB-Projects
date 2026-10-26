import numpy as np

from chipvision.io import read_image, write_image


def test_unicode_path_roundtrip(tmp_path):
    directory = tmp_path / "中文路径-测试"
    directory.mkdir()
    path = directory / "芯片.png"
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    image[5:15, 8:22, 2] = 240

    write_image(path, image)
    loaded = read_image(path)

    assert loaded.shape == image.shape
    assert np.array_equal(loaded, image)
