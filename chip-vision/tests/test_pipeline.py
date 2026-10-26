from pathlib import Path

from chipvision.config import Config
from chipvision.pipeline import ChipVisionPipeline

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_pose_only(tmp_path):
    config = Config.load(ROOT / "configs" / "default.yaml", root=ROOT)
    pipeline = ChipVisionPipeline(config)
    summary = pipeline.run(
        input_dir=ROOT / "input",
        output_dir=tmp_path,
        limit=1,
        ocr=False,
        progress=False,
    )
    assert summary["num_images"] == 1
    assert summary["num_ok"] == 1
    assert (tmp_path / "predictions.json").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "overlays" / "board-01.png").exists()
    assert (tmp_path / "masks" / "board-01.png").exists()
    assert (tmp_path / "canonical" / "board-01.png").exists()
