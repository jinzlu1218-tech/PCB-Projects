from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the project configuration is invalid."""


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class Config:
    """YAML configuration with paths resolved against the project root."""

    data: dict[str, Any]
    root: Path

    @classmethod
    def load(cls, path: str | Path = "configs/default.yaml", root: str | Path | None = None) -> Config:
        config_path = Path(path)
        if root is None:
            root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
        root = Path(root).resolve()
        if not config_path.is_absolute():
            config_path = (root / config_path).resolve()
        if not config_path.exists():
            raise ConfigError(f"Configuration file does not exist: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ConfigError("The top-level YAML value must be a mapping")
        cfg = cls(data=data, root=root)
        cfg.validate()
        return cfg

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        return Config(data=_merge(self.data, overrides), root=self.root)

    def validate(self) -> None:
        required = ["paths", "reference", "chip", "ocr"]
        missing = [key for key in required if key not in self.data]
        if missing:
            raise ConfigError(f"Missing configuration sections: {', '.join(missing)}")
        corners = self.data["reference"].get("body_corners_px")
        if not isinstance(corners, list) or len(corners) != 4:
            raise ConfigError("reference.body_corners_px must contain four points")
        bbox = self.data["reference"].get("template_bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ConfigError("reference.template_bbox must be [x0, y0, x1, y1]")

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path)

    @property
    def input_dir(self) -> Path:
        return self.resolve(self.data["paths"]["input_dir"])

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.data["paths"]["output_dir"])

    @property
    def reference_image(self) -> Path:
        return self.resolve(self.data["reference"]["image"])

    @property
    def model_storage_dir(self) -> Path:
        return self.resolve(self.data["ocr"].get("model_storage_dir", "weights/easyocr"))
