from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from counter.adapters.repo.memory import InMemoryObjectCountRepo
from counter.domain.models import ObjectCount

pytestmark = pytest.mark.unit


class TestInMemoryObjectCountRepo:
    def test_reads_back_what_was_written(self) -> None:
        repo = InMemoryObjectCountRepo()
        repo.update_values([ObjectCount("cat", 2)])

        assert repo.read_values() == [ObjectCount("cat", 2)]

    def test_update_increments_instead_of_replacing(self) -> None:
        repo = InMemoryObjectCountRepo()
        repo.update_values([ObjectCount("cat", 2)])
        repo.update_values([ObjectCount("cat", 3), ObjectCount("dog", 1)])

        assert repo.read_values() == [ObjectCount("cat", 5), ObjectCount("dog", 1)]

    def test_unknown_classes_read_as_zero_not_none(self) -> None:
        repo = InMemoryObjectCountRepo()
        repo.update_values([ObjectCount("cat", 1)])

        assert repo.read_values(["cat", "giraffe"]) == [
            ObjectCount("cat", 1),
            ObjectCount("giraffe", 0),
        ]

    def test_reads_are_ordered_by_class(self) -> None:
        repo = InMemoryObjectCountRepo()
        repo.update_values([ObjectCount("zebra", 1), ObjectCount("ant", 1)])

        assert [item.object_class for item in repo.read_values()] == ["ant", "zebra"]

    def test_concurrent_updates_do_not_lose_counts(self) -> None:
        repo = InMemoryObjectCountRepo()

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: repo.update_values([ObjectCount("cat", 1)]), range(500)))

        assert repo.read_values(["cat"]) == [ObjectCount("cat", 500)]
