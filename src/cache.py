"""A small bounded cache that forgets the least recently used entry first.

Shared by the routes that read and write the same entries, which is why it is
an object rather than a decorator: two handlers need one store, the response
needs to say whether it came from here, and switching graphs needs to empty it.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """At most ``capacity`` entries; the coldest one goes first.

    The oldest entry sits at the front of the ordered dict and the most
    recently used at the back. Reading or writing a key moves it to the back,
    so whatever is at the front is the one to drop.
    """

    def __init__(self, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("LRUCache capacity must be positive")
        self.capacity = capacity
        self._store: "OrderedDict[K, V]" = OrderedDict()

    def get(self, key: K) -> Optional[V]:
        """The value, marked as just used, or ``None``."""
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def __contains__(self, key: K) -> bool:
        # Asking is not using: membership does not change the order.
        return key in self._store

    def set(self, key: K, value: V) -> None:
        """Store a value, then drop the oldest entries past capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
