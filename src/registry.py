"""Which graphs this process currently holds open.

A map from a key to an open handle, a lock around it, and the ability to point
a key somewhere new while readers are still using the old target. That is the
whole module.

It stores **opaque handles**. It does not import the retrieval engine, the
store driver, or anything that loads a model, and it never constructs a handle
itself — the function that knows how to open one is injected by the layer that
does know. That is what lets this be tested with no database, no model, and no
native library, and it is why a registry entry can hold anything a caller
wants to keep beside a key.

What the key is
---------------

**The tenant identifier.** One graph per tenant.

This is chosen from what the tenancy layer actually produces, not from what a
registry could support. A verified request resolves exactly one string — an
organisation id — and the credential behind it carries `key_id`, `org_id` and
a revocation time and nothing else. No route parameter, no configuration
value, and no field on a credential names a graph. A compound key of tenant
and graph name would need that second axis to come from somewhere, and there
is nowhere for it to come from that would not be invented here.

The accepted consequence: serving two repositories to one customer means two
tenants. If a caller should ever choose between graphs, the key becomes a
compound and *that* is the change — a new axis at the call sites, not a
rewrite of this module, because the key here is an opaque string and this
module attaches no meaning to it.

**A graph name is deliberately not smuggled into the value.** The value holds
the path and version of the graph a key points at, which answers "which build
is loaded"; it does not hold a second identifier that would make the key a
compound in everything but name.

The closing rule
----------------

**Nothing here closes a handle except ``close_all``.**

``attach`` and ``replace`` both return whatever they displaced and leave it
open. The alternative — closing on the caller's behalf — is a use-after-close:
a reader that fetched the old handle a microsecond before the swap is still
reading through it, and the swap has no way to know. The caller holds the
displaced entry and decides when nothing is using it.

One rule, both methods, stated on both. An inconsistency between them is the
bug this paragraph exists to prevent.

Ordering, and what a reader can see
-----------------------------------

A replacement opens the new handle **before** taking the lock, then swaps with
a single assignment while holding it. So a reader sees the old entry or the
new one and never a partially built one, and readers are not blocked for the
seconds an open can take.

Two concurrent replacements of one key resolve last-writer-wins, and each
caller is handed whatever its own swap displaced. No handle is dropped without
being returned to somebody.

If the loader raises, the map is untouched — the failure happens before the
lock is taken, so the previous entry stays in place and reachable.

One per process
---------------

``REGISTRY`` at the bottom of this module is the one a running process uses.
Consumers take a registry and fall back to it, which is how a pass called
with no argument still acts on the graphs requests are being served from.
See the note beside its declaration for why *sharing* one is a different
thing from *defaulting to a fresh* one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class GraphEntry:
    """One open graph, and enough to say which build of it is loaded.

    ``version`` is whatever the caller uses to tell one build from another — a
    commit, a timestamp, a digest. This module never interprets it; it exists
    so "is the loaded graph the current one" can be answered without opening
    anything.
    """

    handle: Any
    path: str
    version: str | None = None
    loaded_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class GraphRegistry:
    """Open graphs by key, with replacement that does not disturb readers.

    The lock is reentrant because ``replace`` finishes by calling ``attach``,
    and a plain lock would deadlock on the second acquisition.
    """

    def __init__(self, loader: Callable[[str], Any] | None = None) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, GraphEntry] = {}
        self._loader = loader

    # -- configuration -----------------------------------------------------

    def set_loader(self, loader: Callable[[str], Any] | None) -> None:
        """Install the function that opens a graph from a path.

        Injected rather than imported so this module stays free of the engine.
        Without one, ``attach`` still works and ``replace`` refuses — which is
        the state every test of this module runs in.
        """
        with self._lock:
            self._loader = loader

    @property
    def has_loader(self) -> bool:
        with self._lock:
            return self._loader is not None

    # -- writing -----------------------------------------------------------

    def attach(
        self, key: str, handle: Any, *, path: str, version: str | None = None
    ) -> GraphEntry | None:
        """Point ``key`` at an already-open ``handle``.

        **Returns whatever it displaced, still open.** Closing it is the
        caller's decision — see the module docstring. Returns ``None`` when the
        key was not previously loaded.
        """
        entry = GraphEntry(handle=handle, path=path, version=version)
        with self._lock:
            displaced = self._entries.get(key)
            # The single visible assignment. Everything that could fail has
            # already happened by the time this line runs.
            self._entries[key] = entry
        return displaced

    def replace(
        self, key: str, *, path: str, version: str | None = None
    ) -> GraphEntry | None:
        """Open ``path`` and point ``key`` at it.

        **Returns whatever it displaced, still open**, on the same rule as
        ``attach``.

        The open happens before the lock is taken. A loader that raises
        therefore leaves the map exactly as it was, and readers are not held
        up for the duration of an open.
        """
        with self._lock:
            loader = self._loader
        if loader is None:
            raise RuntimeError(
                "no loader is installed; call set_loader before replace, or "
                "use attach with a handle you opened yourself"
            )

        handle = loader(path)
        return self.attach(key, handle, path=path, version=version)

    def detach(self, key: str) -> GraphEntry | None:
        """Forget ``key``. Returns its entry, still open, or ``None``."""
        with self._lock:
            return self._entries.pop(key, None)

    # -- reading -----------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """The handle for ``key``, or ``None``.

        ``None`` rather than an exception: "not loaded in this process" is an
        ordinary answer, and a caller that has to catch to ask a question ends
        up catching more than it meant to.
        """
        with self._lock:
            entry = self._entries.get(key)
        return entry.handle if entry is not None else None

    def entry(self, key: str) -> GraphEntry | None:
        """The whole record for ``key`` — handle, path, version, load time."""
        with self._lock:
            return self._entries.get(key)

    def keys(self) -> list[str]:
        """Every key currently loaded, in insertion order. A copy, not a view."""
        with self._lock:
            return list(self._entries)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # -- shutdown ----------------------------------------------------------

    def close_all(self) -> list[tuple[str, BaseException]]:
        """Close every handle and empty the map. The one place this closes.

        Closing at shutdown is the registry's job rather than the caller's,
        because the caller would have to enumerate the keys and race anything
        still replacing them. Here the map is emptied under the lock first, so
        nothing can be handed out while the closes are running.

        Every handle is attempted even if an earlier one raises — one store
        refusing to close must not leave the rest open — and the failures are
        returned rather than logged or swallowed, so a caller can report them
        without this module deciding how.
        """
        with self._lock:
            entries = list(self._entries.items())
            self._entries.clear()

        failures: list[tuple[str, BaseException]] = []
        for key, entry in entries:
            close = getattr(entry.handle, "close", None)
            if close is None:
                continue
            try:
                close()
            except BaseException as exc:  # noqa: BLE001 - reported, not hidden
                failures.append((key, exc))
        return failures

    def close_entries(self, entries: Iterable[GraphEntry]) -> None:
        """Close handles the caller displaced and has decided are safe.

        Here rather than at the call site so the displaced-handle rule has one
        implementation: the registry never decides *when*, only *how*.
        """
        for entry in entries:
            close = getattr(entry.handle, "close", None)
            if close is not None:
                close()


#: The registry this process serves from.
#:
#: One object, for the life of the process. Startup attaches the store it
#: opens to this, every route resolves through this, and the pod agent's
#: passes act on this — so what a request can read is exactly what the agent
#: has loaded, by construction rather than by everyone being handed the same
#: argument.
#:
#: **A note for anyone who reads the reasoning that used to sit here.** The
#: consumers of this object once refused to default at all, on the grounds
#: that a default would hand each caller its own graphs that no request ever
#: reads — every run looking like it worked while serving nothing. That
#: reasoning was about defaulting to a **fresh instance**, and about that it
#: is still exactly right. It says nothing against a **shared** one: this is
#: the same object the application holds, with the same handles in it, and a
#: caller that falls back to it lands on the process's real graphs rather
#: than on an empty private copy.
#:
#: The distinction is the whole of the difference, and it is worth keeping in
#: view — a future default that constructs rather than shares would look like
#: this line and behave like the bug.
REGISTRY = GraphRegistry()


def reset_registry() -> None:
    """Empty the process registry, closing whatever it holds.

    Module-level state outlives a test unless something clears it, and a
    registry carried from one test into the next is a graph the next test did
    not attach. This is the same shape as the reset the graph store's
    per-process schema guard has, and for the same reason.

    Not for use in a running process: closing every handle underneath live
    readers is precisely what the registry's own rules exist to prevent.
    """
    REGISTRY.close_all()
    REGISTRY.set_loader(None)
