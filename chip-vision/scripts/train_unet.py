"""Train the optional U-Net chip segmentation model.

This is the PyTorch part of the pose branch.  The production pipeline uses the
reference matcher first because it is more precise for this fixed board; the
U-Net can be trained from pseudo-labels and used as a learned fallback.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from chipvision.chip.unet import SmallUNet
from chipvision.io import read_image, write_json


def _augment(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    angle = random.uniform(-180, 180)
    scale = random.uniform(0.80, 1.18)
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    rotation[0, 2] += random.uniform(-0.08, 0.08) * width
    rotation[1, 2] += random.uniform(-0.08, 0.08) * height
    warped_image = cv2.warpAffine(
        image, rotation, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    warped_mask = cv2.warpAffine(
        mask, rotation, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT
    )
    warped_image = cv2.resize(warped_image, (size, size), interpolation=cv2.INTER_AREA)
    warped_mask = cv2.resize(warped_mask, (size, size), interpolation=cv2.INTER_NEAREST)

    if random.random() < 0.7:
        gain = random.uniform(0.65, 1.35)
        bias = random.uniform(-25, 25)
        warped_image = np.clip(warped_image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if random.random() < 0.25:
        sigma = random.uniform(0.4, 1.8)
        warped_image = cv2.GaussianBlur(warped_image, (0, 0), sigma)

    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    tensor_image = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0)
    tensor_mask = torch.from_numpy((warped_mask > 127).astype(np.float32)).unsqueeze(0)
    return tensor_image, tensor_mask


class PseudoChipDataset(Dataset):
    def __init__(self, records: list[dict], root: Path, size: int, train: bool) -> None:
        self.records = records
        self.root = root
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = read_image(self.root / "images" / record["image"])
        mask = read_image(self.root / "masks" / record["mask"])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if self.train:
            return _augment(image, mask, self.size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_tensor = torch.from_numpy(
            cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32)
            / 255.0
        ).unsqueeze(0)
        mask_tensor = torch.from_numpy(
            (cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST) > 127).astype(
                np.float32
            )
        ).unsqueeze(0)
        return image_tensor, mask_tensor


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    union = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2 * intersection + epsilon) / (union + epsilon)).mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/pseudo")
    parser.add_argument("--output", default="weights/chip_unet.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setRNGSeed(args.seed)

    data_root = Path(args.data)
    labels_path = data_root / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} does not exist. Run scripts/make_pseudo_labels.py first."
        )
    records = json.loads(labels_path.read_text(encoding="utf-8"))["images"]
    if len(records) < 4:
        raise ValueError("At least four pseudo-labelled images are required")

    random.shuffle(records)
    validation_count = max(1, round(len(records) * 0.2))
    train_records = records[validation_count:]
    validation_records = records[:validation_count]

    train_loader = DataLoader(
        PseudoChipDataset(train_records, data_root, args.size, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        PseudoChipDataset(validation_records, data_root, args.size, train=False),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(args.device)
    model = SmallUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    best_iou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = functional.binary_cross_entropy_with_logits(logits, masks) + _dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * images.shape[0]
        scheduler.step()

        model.eval()
        intersections = 0.0
        unions = 0.0
        with torch.inference_mode():
            for images, masks in validation_loader:
                images, masks = images.to(device), masks.to(device)
                probabilities = torch.sigmoid(model(images)) > 0.5
                intersections += float((probabilities & (masks > 0.5)).sum().item())
                unions += float((probabilities | (masks > 0.5)).sum().item())
        iou = intersections / max(unions, 1.0)
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss / max(len(train_records), 1), 5),
            "val_iou": round(iou, 5),
        }
        history.append(epoch_record)
        print(epoch_record)
        if iou > best_iou:
            best_iou = iou
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "base_channels": 32,
                    "size": args.size,
                    "val_iou": iou,
                    "epoch": epoch,
                },
                output,
            )

    write_json(Path(args.output).with_suffix(".history.json"), {"history": history})
    print(f"Best validation IoU: {best_iou:.4f}; checkpoint: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
