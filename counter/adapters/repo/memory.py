"""In-memory counts. Default for the dev profile and for unit tests."""

from __future__ import annotations

import threading
from collections.abc import Sequence

from counter.domain.models import ObjectCount
from counter.domain.ports import ObjectCountRepo


class InMemoryObjectCountRepo(ObjectCountRepo):
    def __init__(self) -> None:
        self._store: dict[tuple[str, str, str], int] = {}
        self._lock = threading.Lock()

    def read_values(
        self,
        model_name: str,
        model_version: str,
        object_classes: Sequence[str] | None = None,
    ) -> list[ObjectCount]:
        with self._lock:
            if object_classes is None:
                items = sorted(
                    (key[2], count)
                    for key, count in self._store.items()
                    if key[:2] == (model_name, model_version)
                )
            else:
                items = [
                    (name, self._store.get((model_name, model_version, name), 0))
                    for name in object_classes
                ]
        return [ObjectCount(object_class=name, count=count) for name, count in items]

    def update_values(
        self,
        model_name: str,
        model_version: str,
        new_values: Sequence[ObjectCount],
    ) -> None:
        with self._lock:
            for value in new_values:
                if value.count == 0:
                    continue
                key = (model_name, model_version, value.object_class)
                self._store[key] = self._store.get(key, 0) + value.count
