"""Typed runtime configuration."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["dev", "test", "prod"]
Persistence = Literal["memory", "sql"]

PACKAGED_CATALOGS = {
    "prod": "models.yaml",
    "dev": "models.dev.yaml",
    "test": "models.dev.yaml",
}


class Settings(BaseSettings):
    """Runtime configuration, read from the environment and `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="COUNTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    env: Environment = "dev"

    # --- API -------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 5000
    default_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_image_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    cors_allow_origins: list[str] = Field(default_factory=list)

    # --- Persistence -----------------------------------------------------
    persistence: Persistence = "memory"
    database_url: str = "postgresql+psycopg://counter:counter@localhost:5432/object_counter"
    db_echo: bool = False
    db_pool_size: int = Field(default=5, gt=0)

    # --- Models ----------------------------------------------------------
    model_catalog: Path | None = None
    default_model: str | None = None
    preload_models: bool = False
    artifact_cache_dir: Path = Path("var/models")
    max_image_side: int = Field(default=1024, gt=0)

    # --- TensorFlow Serving ---------------------------------------------
    tfs_base_url: str = "http://localhost:8501"
    tfs_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Observability ---------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> Settings:
        """Apply profile defaults and production invariants."""
        explicit = self.model_fields_set

        if "persistence" not in explicit and self.env == "prod":
            self.persistence = "sql"

        if "model_catalog" not in explicit or self.model_catalog is None:
            self.model_catalog = packaged_catalog_path(PACKAGED_CATALOGS[self.env])

        if self.env == "prod" and self.persistence != "sql":
            raise ValueError("prod requires sql persistence")

        return self

    @property
    def is_production(self) -> bool:
        return self.env == "prod"


def packaged_catalog_path(filename: str) -> Path:
    """Absolute path to a catalog shipped inside the package."""
    return Path(str(resources.files("counter.resources").joinpath(filename)))
