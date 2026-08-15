"""In-memory counts. Default for the dev profile and for unit tests."""

from __future__ import annotations

import threading
from collections.abc import Sequence

from counter.domain.models import ObjectCount
from counter.domain.ports import ObjectCountRepo


class InMemoryObjectCountRepo(ObjectCountRepo):
    """Process-local counts.

    Two deliberate differences from the original in-memory adapter:

    * a lock. `store[key] = ObjectCount(key, stored.count + new.count)` is a
      read-modify-write, and the app is served by a threadpool, so two requests
      counting a cat each could leave the store at one.
    * unknown classes come back as a zero count instead of `None`. The old
      `read_values(['giraffe'])` returned `[None]` for a class never seen, and
      every caller had to remember to filter it out.
    """

    def __init__(self) -> None:
        self._store: dict[str, int] = {}
        self._lock = threading.Lock()

    def read_values(self, object_classes: Sequence[str] | None = None) -> list[ObjectCount]:
        with self._lock:
            if object_classes is None:
                items = sorted(self._store.items())
            else:
                items = [(name, self._store.get(name, 0)) for name in object_classes]
        return [ObjectCount(object_class=name, count=count) for name, count in items]

    def update_values(self, new_values: Sequence[ObjectCount]) -> None:
        with self._lock:
            for value in new_values:
                self._store[value.object_class] = (
                    self._store.get(value.object_class, 0) + value.count
                )
