"""Remember a bounded number of computed values; forget the stalest first.

Summaries and answers cost a model call each, and the same question tends to
come back. This keeps the most recent results within a fixed budget. Two
routes share one instance, a response reports whether its value came from
here, and switching graphs has to be able to forget everything at once, so it
is a plain object rather than a function decorator.
"""

from __future__ import annotations

from typing import Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")

_ABSENT = object()


class LRUCache(Generic[K, V]):
    """Up to ``capacity`` values, keyed; touching a key renews it.

    Python dictionaries iterate in insertion order, so re-inserting a key on
    every touch keeps the entries sorted from stalest to freshest, and the
    first key the dictionary yields is always the one to give up.
    """

    def __init__(self, capacity: int = 256) -> None:
        if capacity < 1:
            raise ValueError(f"a cache must hold at least one entry, not {capacity}")
        self.capacity = capacity
        self._entries: dict[K, V] = {}

    def _renew(self, key: K, value: V) -> None:
        self._entries.pop(key, None)
        self._entries[key] = value

    def get(self, key: K) -> Optional[V]:
        """The stored value, now counted as fresh, or ``None`` if absent."""
        value = self._entries.get(key, _ABSENT)
        if value is _ABSENT:
            return None
        self._renew(key, value)
        return value

    def set(self, key: K, value: V) -> None:
        """Keep ``value`` under ``key`` as the freshest entry, within capacity."""
        self._renew(key, value)
        while len(self._entries) > self.capacity:
            del self._entries[next(iter(self._entries))]

    def __contains__(self, key: K) -> bool:
        # A look that does not read the value leaves its age alone.
        return key in self._entries

    def clear(self) -> None:
        self._entries = {}

    def __len__(self) -> int:
        return len(self._entries)
