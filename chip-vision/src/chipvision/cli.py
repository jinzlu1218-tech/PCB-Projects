from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config, ConfigError
from .pipeline import ChipVisionPipeline
from .version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chipvision",
        description="Chip pose estimation and chip-surface OCR on PCB images.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config")
    parser.add_argument("--root", default=None, help="Project root (defaults to the config's parent)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run the pipeline on a directory of images")
    infer.add_argument("--input", default=None, help="Input image directory")
    infer.add_argument("--output", default=None, help="Output directory")
    infer.add_argument("--limit", type=int, default=None, help="Process at most N images")
    infer.add_argument("--no-ocr", action="store_true", help="Skip text recognition")
    infer.add_argument("--quiet", action="store_true", help="Suppress per-image progress")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions against annotations")
    evaluate.add_argument("--predictions", default="outputs/predictions.json")
    evaluate.add_argument("--annotations", required=True)
    evaluate.add_argument("--output", default="outputs/metrics.json")
    return parser


def _load_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config, root=args.root)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load_config(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.command == "infer":
        pipeline = ChipVisionPipeline(config)
        summary = pipeline.run(
            input_dir=args.input,
            output_dir=args.output,
            limit=args.limit,
            ocr=not args.no_ocr,
            progress=not args.quiet,
        )
        print(
            f"Done: {summary['num_ok']}/{summary['num_images']} images ok, "
            f"{summary['num_errors']} errors. Results: {summary['predictions']}"
        )
        return 0 if summary["num_errors"] == 0 else 1

    if args.command == "evaluate":
        from .evaluate import evaluate_predictions

        metrics = evaluate_predictions(
            predictions_path=Path(args.predictions),
            annotations_path=Path(args.annotations),
            output_path=Path(args.output),
        )
        print(f"Metrics written to {args.output}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
