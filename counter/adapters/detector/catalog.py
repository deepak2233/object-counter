"""Validated model catalog."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from counter.domain.errors import ModelLoadError

Framework = Literal["tensorflow-serving", "onnx", "torchscript", "fake"]
ENV_REFERENCE = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")


class ArtifactSpec(BaseModel):
    """Model artifact location and optional checksum."""

    model_config = {"extra": "forbid"}

    uri: str = Field(min_length=1)
    sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
        description="Expected SHA-256 digest.",
    )


class ModelSpec(BaseModel):
    model_config = {"extra": "forbid", "protected_namespaces": ()}

    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    framework: Framework
    version: str = Field(default="1", min_length=1, max_length=64)
    labels: str | None = None

    # onnx / torchscript
    artifact: ArtifactSpec | None = None
    input_size: int = Field(default=640, gt=0)
    input_contract: Literal["yolov8", "torchvision"] = "yolov8"
    score_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    device: str = "cpu"

    # tensorflow-serving
    endpoint: str | None = None
    remote_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._-]+$",
        description="Model name inside TF Serving when it differs from the catalog name.",
    )

    @model_validator(mode="after")
    def _check_backend_requirements(self) -> ModelSpec:
        if self.framework in {"onnx", "torchscript"} and self.artifact is None:
            raise ValueError(f"model '{self.name}' ({self.framework}) needs an artifact")
        if self.framework not in {"onnx", "torchscript"} and self.artifact is not None:
            raise ValueError(f"model '{self.name}' does not use an artifact")
        if self.framework != "tensorflow-serving" and (self.endpoint or self.remote_name):
            raise ValueError(f"model '{self.name}' does not use a serving endpoint")
        if self.artifact and self.artifact.uri.startswith("s3://") and not self.artifact.sha256:
            raise ValueError(f"model '{self.name}' must pin its S3 artifact with sha256")
        if self.framework == "tensorflow-serving" and not self.version.isdigit():
            raise ValueError(f"model '{self.name}' needs a numeric TensorFlow Serving version")
        if self.input_contract == "torchvision" and self.framework != "torchscript":
            raise ValueError("torchvision input_contract is only valid for torchscript")
        return self


class ModelCatalog(BaseModel):
    model_config = {"extra": "forbid", "protected_namespaces": ()}

    default_model: str
    models: list[ModelSpec]

    @model_validator(mode="after")
    def _check_names(self) -> ModelCatalog:
        names = [spec.name for spec in self.models]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate model names in catalog: {sorted(duplicates)}")
        if self.default_model not in names:
            raise ValueError(f"default_model '{self.default_model}' is not in the catalog {names}")
        return self

    def spec_for(self, name: str) -> ModelSpec | None:
        return next((spec for spec in self.models if spec.name == name), None)

    @classmethod
    def from_file(cls, path: str | Path) -> ModelCatalog:
        """Load YAML or JSON and expand environment references."""
        catalog_path = Path(path)
        if not catalog_path.is_file():
            raise ModelLoadError(f"model catalog not found: {catalog_path}")

        raw = os.path.expandvars(catalog_path.read_text(encoding="utf-8"))
        try:
            data: Any = (
                json.loads(raw) if catalog_path.suffix.lower() == ".json" else yaml.safe_load(raw)
            )
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise ModelLoadError(f"model catalog {catalog_path} is not parseable: {exc}") from exc

        unresolved = _find_environment_reference(data)
        if unresolved:
            raise ModelLoadError(f"unresolved environment reference {unresolved}")

        try:
            return cls.model_validate(data)
        except ValueError as exc:
            raise ModelLoadError(f"model catalog {catalog_path} is invalid: {exc}") from exc


def _find_environment_reference(value: Any) -> str | None:
    if isinstance(value, str):
        regex_match = ENV_REFERENCE.search(value)
        return regex_match.group(0) if regex_match else None
    if isinstance(value, dict):
        return next(
            (
                reference
                for item in value.values()
                if (reference := _find_environment_reference(item))
            ),
            None,
        )
    if isinstance(value, list):
        return next(
            (reference for item in value if (reference := _find_environment_reference(item))),
            None,
        )
    return None
