"""Build a labelled contact sheet from the generated overlay images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from chipvision.io import list_images, read_image, write_image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlays", default="outputs/overlays")
    parser.add_argument("--output", default="docs/results/contact_sheet.jpg")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=430)
    args = parser.parse_args()

    paths = list_images(args.overlays)
    if not paths:
        raise FileNotFoundError(f"No overlay images found in {args.overlays}")
    thumbs = []
    for path in paths:
        image = read_image(path)
        ratio = args.thumb_width / image.shape[1]
        thumb = cv2.resize(
            image,
            (args.thumb_width, round(image.shape[0] * ratio)),
            interpolation=cv2.INTER_AREA,
        )
        thumbs.append((path.stem, thumb))

    thumb_height = max(image.shape[0] for _, image in thumbs) + 26
    columns = max(1, args.columns)
    rows = (len(thumbs) + columns - 1) // columns
    cell_width = args.thumb_width + 10
    sheet = np.full((rows * thumb_height, columns * cell_width, 3), 235, dtype=np.uint8)
    for index, (name, thumb) in enumerate(thumbs):
        row, column = divmod(index, columns)
        x = column * cell_width + 5
        y = row * thumb_height + 22
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
        cv2.putText(
            sheet,
            name,
            (x, row * thumb_height + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    output = Path(args.output)
    write_image(output, sheet)
    print(f"Wrote {len(thumbs)} overlays to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
