"""The model catalog: a declarative list of what this service can serve.

Adding an internally trained model is a catalog entry plus an artifact in the
model store — no code change, no rebuild. Validation happens at load time with
pydantic, so a typo in the catalog fails at startup with a precise message
instead of at 3am with a KeyError.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from counter.domain.errors import ModelLoadError

Framework = Literal["tensorflow-serving", "onnx", "torchscript", "fake"]


class ArtifactSpec(BaseModel):
    """Where a model file lives and what it must hash to."""

    uri: str
    sha256: str | None = Field(
        default=None,
        description="Expected digest. Set it for anything that leaves the build: "
        "it is the difference between 'we serve model 1.4.2' and 'we serve "
        "whatever was in the bucket'.",
    )


class ModelSpec(BaseModel):
    model_config = {"extra": "forbid", "protected_namespaces": ()}

    name: str
    framework: Framework
    version: str = "1"
    labels: str | None = None

    # onnx / torchscript
    artifact: ArtifactSpec | None = None
    input_size: int = 640
    score_floor: float = 0.05
    iou_threshold: float = 0.45
    device: str = "cpu"

    # tensorflow-serving
    endpoint: str | None = None
    remote_name: str | None = Field(
        default=None,
        description="Model name inside TF Serving when it differs from the catalog name.",
    )

    @model_validator(mode="after")
    def _check_backend_requirements(self) -> ModelSpec:
        if self.framework in {"onnx", "torchscript"} and self.artifact is None:
            raise ValueError(f"model '{self.name}' ({self.framework}) needs an artifact")
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
        """Load a YAML or JSON catalog, expanding ${ENV_VAR} references.

        Environment expansion keeps endpoints and bucket names out of the file,
        so the same catalog is deployable to dev, staging and production.
        """
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

        try:
            return cls.model_validate(data)
        except ValueError as exc:
            raise ModelLoadError(f"model catalog {catalog_path} is invalid: {exc}") from exc
