"""Label-map loading and lookup."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
from pathlib import Path

from counter.domain.errors import ModelLoadError

PACKAGE_RESOURCES = "counter.resources"


@lru_cache(maxsize=8)
def load_labels(source: str = "mscoco_label_map.json") -> Mapping[int, str]:
    """Load a label map by packaged resource name or by filesystem path.

    Two formats are supported so that internally trained models can ship the
    simpler one:
      * TF object detection JSON: `[{"id": 1, "display_name": "person"}, ...]`
      * plain text: one class name per line, index = line number (YOLO style)
    """
    path = Path(source)
    if path.is_file():
        raw = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
    else:
        try:
            raw = resources.files(PACKAGE_RESOURCES).joinpath(source).read_text(encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            raise ModelLoadError(f"label map not found: {source}") from exc
        suffix = Path(source).suffix.lower()

    if suffix == ".json":
        return _parse_json_label_map(raw, source)
    return _parse_text_label_map(raw)


def _parse_json_label_map(raw: str, source: str) -> Mapping[int, str]:
    try:
        entries = json.loads(raw)
        return {int(entry["id"]): str(entry["display_name"]) for entry in entries}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ModelLoadError(f"label map {source} is malformed: {exc}") from exc


def _parse_text_label_map(raw: str) -> Mapping[int, str]:
    names = [line.strip() for line in raw.splitlines() if line.strip()]
    return dict(enumerate(names))


def class_name_for(labels: Mapping[int, str], class_id: float | int) -> str:
    """Resolve numeric class IDs and retain unknown classes."""
    key = int(class_id)
    return labels.get(key, f"class_{key}")
