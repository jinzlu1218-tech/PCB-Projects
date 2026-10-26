from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .geometry import angle_from_quad, quad_center
from .io import read_json, write_json


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (char_a != char_b)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def _character_error_rate(reference: str, hypothesis: str) -> float:
    reference = "".join(reference.upper().split())
    hypothesis = "".join(hypothesis.upper().split())
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(reference, hypothesis) / len(reference)


def _angle_error_axis(prediction: float, target: float) -> float:
    difference = (prediction - target) % 180.0
    return float(min(difference, 180.0 - difference))


def evaluate_predictions(
    predictions_path: Path,
    annotations_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    predictions = read_json(predictions_path)
    annotations = read_json(annotations_path)
    prediction_by_name = {item["image"]: item for item in predictions["images"]}
    annotation_by_name = {item["image"]: item for item in annotations["images"]}

    rows: list[dict[str, Any]] = []
    for name, ground_truth in annotation_by_name.items():
        prediction = prediction_by_name.get(name)
        if prediction is None or prediction.get("status") != "ok":
            rows.append({"image": name, "error": "missing prediction"})
            continue
        gt_corners = np.asarray(ground_truth["chip"]["corners_px"], dtype=np.float64)
        pred_corners = np.asarray(prediction["chip"]["corners_px"], dtype=np.float64)
        gt_center = quad_center(gt_corners)
        pred_center = quad_center(pred_corners)
        center_error = float(np.linalg.norm(gt_center - pred_center))
        gt_angle = angle_from_quad(gt_corners)[0]
        pred_angle = float(prediction["chip"]["angle_deg"])
        corner_rmse = float(np.sqrt(np.mean(np.sum((gt_corners - pred_corners) ** 2, axis=1))))
        gt_text = ground_truth.get("text", {}).get("full_text", "")
        pred_text = prediction.get("text", {}).get("full_text", "")
        rows.append(
            {
                "image": name,
                "center_error_px": center_error,
                "angle_error_deg": _angle_error_axis(pred_angle, gt_angle),
                "corner_rmse_px": corner_rmse,
                "text_cer": _character_error_rate(gt_text, pred_text),
                "text_exact": int("".join(gt_text.upper().split()) == "".join(pred_text.upper().split())),
                "prediction_text": pred_text,
                "ground_truth_text": gt_text,
            }
        )

    valid = [row for row in rows if "center_error_px" in row]
    metrics = {
        "num_annotated": len(rows),
        "num_matched": len(valid),
        "mean_center_error_px": float(np.mean([r["center_error_px"] for r in valid])) if valid else None,
        "median_center_error_px": float(np.median([r["center_error_px"] for r in valid])) if valid else None,
        "mean_angle_error_deg": float(np.mean([r["angle_error_deg"] for r in valid])) if valid else None,
        "mean_corner_rmse_px": float(np.mean([r["corner_rmse_px"] for r in valid])) if valid else None,
        "mean_text_cer": float(np.mean([r["text_cer"] for r in valid])) if valid else None,
        "text_exact_match_rate": float(np.mean([r["text_exact"] for r in valid])) if valid else None,
        "per_image": rows,
    }
    write_json(output_path, metrics)
    return {key: value for key, value in metrics.items() if key != "per_image"}


__all__ = ["evaluate_predictions"]
