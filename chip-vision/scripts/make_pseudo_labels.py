"""Create chip masks from the reference matcher for U-Net training.

The 20 images have no manual segmentation labels.  The validated SIFT +
homography registration is very accurate on this fixed board, so it is used
to bootstrap masks.  These masks are still *pseudo-labels*; inspect them
before training a model that will be used outside this dataset.
"""

from __future__ import annotations

import argparse

from chipvision.chip.matcher import PoseEstimationError, ReferencePoseEstimator
from chipvision.config import Config
from chipvision.geometry import polygon_mask
from chipvision.io import list_images, read_image, write_image, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="data/pseudo")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    estimator = ReferencePoseEstimator(config)
    output = config.resolve(args.output)
    image_dir = output / "images"
    mask_dir = output / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    records = []
    images = list_images(config.input_dir)
    if args.limit:
        images = images[: args.limit]
    for path in images:
        image = read_image(path)
        try:
            pose = estimator.locate(image)
        except PoseEstimationError as exc:
            print(f"skip {path.name}: {exc}")
            continue
        mask = polygon_mask(image.shape[:2], pose.corners)
        write_image(image_dir / path.name, image)
        write_image(mask_dir / f"{path.stem}.png", mask)
        records.append(
            {
                "image": path.name,
                "mask": f"{path.stem}.png",
                "corners_px": pose.corners.round(3).tolist(),
                "confidence": round(pose.confidence, 4),
                "inliers": pose.inliers,
            }
        )
        print(f"{path.name}: mask area={int((mask > 0).sum())} inliers={pose.inliers}")

    write_json(output / "labels.json", {"images": records})
    print(f"Wrote {len(records)} pseudo-labelled images to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
