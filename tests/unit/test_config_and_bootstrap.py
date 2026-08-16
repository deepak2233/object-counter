from __future__ import annotations

import pytest

from counter.adapters.repo.memory import InMemoryObjectCountRepo
from counter.adapters.repo.sql import SqlObjectCountRepo
from counter.bootstrap import build_registry, build_repo, build_services
from counter.config import Settings
from counter.domain.errors import ModelLoadError

pytestmark = pytest.mark.unit


class TestSettings:
    def test_dev_defaults_to_fakes_and_memory(self) -> None:
        settings = Settings(env="dev")

        assert settings.persistence == "memory"
        assert settings.model_catalog is not None
        assert settings.model_catalog.name == "models.dev.yaml"

    def test_prod_defaults_to_the_relational_store(self) -> None:
        settings = Settings(env="prod")

        assert settings.persistence == "sql"
        assert settings.model_catalog is not None
        assert settings.model_catalog.name == "models.yaml"

    def test_prod_rejects_in_memory_persistence(self) -> None:
        with pytest.raises(ValueError, match="requires sql"):
            Settings(env="prod", persistence="memory")

    def test_reads_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COUNTER_ENV", "prod")
        monkeypatch.setenv("COUNTER_DEFAULT_THRESHOLD", "0.85")

        settings = Settings()

        assert (settings.env, settings.default_threshold) == ("prod", 0.85)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("default_threshold", 1.5), ("max_image_bytes", 0), ("env", "prd")],
    )
    def test_rejects_impossible_values_at_startup(self, field: str, value: object) -> None:
        # An invalid profile used to surface as KeyError: 'prd_count_action'
        # from inside a globals() lookup, at the first request rather than boot.
        with pytest.raises(ValueError, match=field):
            Settings(**{field: value})  # type: ignore[arg-type]


class TestBootstrap:
    def test_memory_profile_wires_the_in_memory_repo(self) -> None:
        repo, engine = build_repo(Settings(env="dev"))

        assert isinstance(repo, InMemoryObjectCountRepo)
        assert engine is None

    def test_sql_profile_wires_the_relational_repo(self, tmp_path: object) -> None:
        repo, engine = build_repo(
            Settings(env="test", persistence="sql", database_url="sqlite+pysqlite:///:memory:")
        )

        assert isinstance(repo, SqlObjectCountRepo)
        assert engine is not None
        engine.dispose()

    def test_registry_comes_from_the_catalog(self) -> None:
        registry = build_registry(Settings(env="dev"))

        assert {info.name for info in registry.available()} == {"fake", "rfcn"}

    def test_default_model_can_be_overridden_by_configuration(self) -> None:
        registry = build_registry(Settings(env="dev", default_model="rfcn"))

        assert registry.default_model == "rfcn"  # type: ignore[attr-defined]

    def test_prod_rejects_a_fake_catalog(self) -> None:
        dev_catalog = Settings(env="dev").model_catalog

        with pytest.raises(ModelLoadError, match="fake detectors"):
            build_registry(Settings(env="prod", model_catalog=dev_catalog))

    def test_build_services_returns_a_usable_container(self) -> None:
        services = build_services(Settings(env="dev"))

        try:
            assert services.detect_action().execute is not None
            assert services.count_action() is not None
        finally:
            services.close()
