"""Tests for the map of open graphs.

No store, no models, no native library. The registry holds opaque handles, so
a handle here is a plain object and every property worth asserting — what a
reader sees during a swap, what a failed load leaves behind, who closes what —
is a property of the map rather than of what it holds.

The concurrency test is the one that earns its length. Calling the methods in
sequence proves nothing about a lock; it has to observe reads that genuinely
overlap a replacement, and assert both that they never saw a half-built entry
and that they were not blocked for the duration of the open.
"""

from __future__ import annotations

import threading
import time

import pytest

from src.registry import GraphEntry, GraphRegistry


class Handle:
    """Something closable, distinguishable, and otherwise inert."""

    def __init__(self, name):
        self.name = name
        self.closed = 0

    def close(self):
        self.closed += 1

    def __repr__(self):
        return f"Handle({self.name!r})"


# --------------------------------------------------------------------------
# attaching and reading
# --------------------------------------------------------------------------


def test_attach_then_fetch_returns_the_same_handle():
    registry = GraphRegistry()
    handle = Handle("a")

    registry.attach("org_a", handle, path="a.db")

    assert registry.get("org_a") is handle


def test_a_missing_key_returns_nothing_rather_than_raising():
    """'not loaded here' is an ordinary answer, not an exceptional one."""
    registry = GraphRegistry()

    assert registry.get("absent") is None
    assert registry.entry("absent") is None
    assert "absent" not in registry


def test_an_entry_records_the_path_and_version_it_was_loaded_from():
    registry = GraphRegistry()
    registry.attach("org_a", Handle("a"), path="a.db", version="build-7")

    entry = registry.entry("org_a")

    assert entry.path == "a.db"
    assert entry.version == "build-7"
    assert entry.loaded_at is not None


def test_listing_keys_reflects_what_is_actually_loaded():
    registry = GraphRegistry()
    registry.attach("org_a", Handle("a"), path="a.db")
    registry.attach("org_b", Handle("b"), path="b.db")

    assert registry.keys() == ["org_a", "org_b"]
    assert len(registry) == 2

    registry.detach("org_a")

    assert registry.keys() == ["org_b"]


def test_the_key_list_is_a_copy_not_a_live_view():
    registry = GraphRegistry()
    registry.attach("org_a", Handle("a"), path="a.db")

    keys = registry.keys()
    registry.attach("org_b", Handle("b"), path="b.db")

    assert keys == ["org_a"]


# --------------------------------------------------------------------------
# replacement, and who closes what
# --------------------------------------------------------------------------


def test_attach_returns_what_it_displaced_and_leaves_it_open():
    """The rule: this module never closes on the caller's behalf."""
    registry = GraphRegistry()
    first = Handle("first")
    registry.attach("org_a", first, path="a.db")

    displaced = registry.attach("org_a", Handle("second"), path="b.db")

    assert displaced.handle is first
    assert first.closed == 0


def test_replace_returns_what_it_displaced_and_leaves_it_open():
    """Same rule as attach. An inconsistency here is a use-after-close."""
    registry = GraphRegistry()
    first = Handle("first")
    registry.attach("org_a", first, path="a.db")
    registry.set_loader(lambda path: Handle(path))

    displaced = registry.replace("org_a", path="b.db")

    assert displaced.handle is first
    assert first.closed == 0
    assert registry.get("org_a").name == "b.db"


def test_a_later_read_sees_the_replacement():
    registry = GraphRegistry()
    registry.attach("org_a", Handle("first"), path="a.db")
    registry.set_loader(lambda path: Handle(path))

    registry.replace("org_a", path="b.db", version="build-9")

    assert registry.entry("org_a").path == "b.db"
    assert registry.entry("org_a").version == "build-9"


def test_detach_returns_the_entry_still_open():
    registry = GraphRegistry()
    handle = Handle("a")
    registry.attach("org_a", handle, path="a.db")

    entry = registry.detach("org_a")

    assert entry.handle is handle
    assert handle.closed == 0
    assert registry.get("org_a") is None


def test_close_entries_is_how_a_caller_closes_what_it_displaced():
    registry = GraphRegistry()
    handle = Handle("a")
    registry.attach("org_a", handle, path="a.db")
    displaced = registry.attach("org_a", Handle("b"), path="b.db")

    registry.close_entries([displaced])

    assert handle.closed == 1


# --------------------------------------------------------------------------
# a loader that fails
# --------------------------------------------------------------------------


def test_a_loader_that_raises_leaves_the_previous_entry_intact():
    """A key must never point at a partially opened graph."""
    registry = GraphRegistry()
    original = Handle("first")
    registry.attach("org_a", original, path="a.db")

    def broken(path):
        raise RuntimeError("could not open")

    registry.set_loader(broken)

    with pytest.raises(RuntimeError, match="could not open"):
        registry.replace("org_a", path="b.db")

    assert registry.get("org_a") is original
    assert registry.entry("org_a").path == "a.db"
    assert original.closed == 0


def test_a_loader_that_raises_on_a_new_key_leaves_it_unloaded():
    registry = GraphRegistry()
    registry.set_loader(lambda path: (_ for _ in ()).throw(RuntimeError("no")))

    with pytest.raises(RuntimeError):
        registry.replace("org_new", path="b.db")

    assert "org_new" not in registry
    assert registry.keys() == []


def test_replacing_without_a_loader_refuses_rather_than_guessing():
    registry = GraphRegistry()

    assert registry.has_loader is False
    with pytest.raises(RuntimeError, match="no loader"):
        registry.replace("org_a", path="a.db")


# --------------------------------------------------------------------------
# concurrency
# --------------------------------------------------------------------------


def test_readers_during_a_replacement_never_see_a_half_built_entry():
    """The property the single assignment under the lock exists to give.

    A reader must observe the old (handle, path) pair or the new one. Never a
    new handle with an old path, never a handle that is not yet usable.

    The loader sleeps, so the reads genuinely overlap the open rather than
    happening to land before or after it, and the observation count is
    asserted so the test cannot pass by never having read anything.
    """
    registry = GraphRegistry()
    old_handle = Handle("old")
    registry.attach("org_a", old_handle, path="old.db")

    opening = threading.Event()
    new_handle = Handle("new")

    def slow_loader(path):
        opening.set()
        time.sleep(0.2)
        return new_handle

    registry.set_loader(slow_loader)

    observations: list[tuple[str, str]] = []
    stop = threading.Event()

    def read():
        opening.wait(timeout=2)
        while not stop.is_set():
            entry = registry.entry("org_a")
            observations.append((entry.handle.name, entry.path))

    readers = [threading.Thread(target=read) for _ in range(4)]
    for reader in readers:
        reader.start()

    registry.replace("org_a", path="new.db")
    stop.set()
    for reader in readers:
        reader.join(timeout=2)

    assert observations, "the readers never ran"
    # Every observation is one consistent pair or the other.
    assert set(observations) <= {("old", "old.db"), ("new", "new.db")}
    # And both were actually seen, so the swap really happened mid-read.
    assert ("old", "old.db") in observations
    assert registry.get("org_a") is new_handle


def test_readers_are_not_blocked_while_a_graph_is_opening():
    """The open happens before the lock is taken, and this is the evidence.

    If the open ran under the lock, no read could complete until it finished.
    """
    registry = GraphRegistry()
    registry.attach("org_a", Handle("old"), path="old.db")

    opening = threading.Event()
    reads_during_open = []

    def slow_loader(path):
        opening.set()
        time.sleep(0.3)
        return Handle("new")

    registry.set_loader(slow_loader)

    def read():
        opening.wait(timeout=2)
        # While the loader is still sleeping.
        for _ in range(5):
            reads_during_open.append(registry.get("org_a").name)
            time.sleep(0.01)

    reader = threading.Thread(target=read)
    reader.start()
    registry.replace("org_a", path="new.db")
    reader.join(timeout=2)

    assert len(reads_during_open) == 5, "reads were blocked by the open"
    assert reads_during_open[0] == "old"


def test_concurrent_replacements_hand_every_handle_back_to_somebody():
    """Last writer wins, and no handle is dropped without being returned."""
    registry = GraphRegistry()
    registry.attach("org_a", Handle("original"), path="a.db")
    registry.set_loader(lambda path: Handle(path))

    displaced: list[GraphEntry] = []
    lock = threading.Lock()

    def swap(path):
        entry = registry.replace("org_a", path=path)
        with lock:
            displaced.append(entry)

    threads = [
        threading.Thread(target=swap, args=(f"{n}.db",)) for n in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    names = {entry.handle.name for entry in displaced}
    survivor = registry.get("org_a").name

    assert len(displaced) == 6
    assert survivor not in names, "the surviving handle was also handed out"
    assert "original" in names


# --------------------------------------------------------------------------
# shutdown
# --------------------------------------------------------------------------


def test_close_all_closes_every_handle_and_empties_the_map():
    registry = GraphRegistry()
    first, second = Handle("a"), Handle("b")
    registry.attach("org_a", first, path="a.db")
    registry.attach("org_b", second, path="b.db")

    failures = registry.close_all()

    assert failures == []
    assert first.closed == 1
    assert second.closed == 1
    assert registry.keys() == []


def test_close_all_closes_the_rest_when_one_refuses():
    """One store failing to close must not leave the others open."""

    class Stubborn(Handle):
        def close(self):
            raise RuntimeError("will not close")

    registry = GraphRegistry()
    stubborn, ordinary = Stubborn("bad"), Handle("good")
    registry.attach("org_a", stubborn, path="a.db")
    registry.attach("org_b", ordinary, path="b.db")

    failures = registry.close_all()

    assert ordinary.closed == 1
    assert [key for key, _exc in failures] == ["org_a"]


def test_close_all_on_an_empty_registry_is_fine():
    assert GraphRegistry().close_all() == []


def test_a_handle_with_no_close_is_not_a_problem():
    """Handles are opaque; nothing says they must be closable."""
    registry = GraphRegistry()
    registry.attach("org_a", object(), path="a.db")

    assert registry.close_all() == []


# --------------------------------------------------------------------------
# the registry knows nothing about the engine
# --------------------------------------------------------------------------


def test_the_registry_imports_nothing_from_the_engine_or_the_store():
    """Asserted against the parse tree, so a docstring example cannot fire it."""
    import ast
    from pathlib import Path

    import src.registry as registry_module

    tree = ast.parse(Path(registry_module.__file__).read_text(encoding="utf-8"))
    forbidden = {"engine", "graphdb", "analysis", "retrieval", "knowledge"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (forbidden & set((node.module or "").split("."))), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not (forbidden & set(alias.name.split("."))), alias.name


def test_the_registry_imports_with_the_store_and_the_models_blocked():
    """A fresh interpreter, because what matters is what import pulls in."""
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;"
            "sys.modules['ladybug'] = None;"
            "sys.modules['sentence_transformers'] = None;"
            "sys.modules['torch'] = None;"
            "from src.registry import GraphRegistry;"
            "r = GraphRegistry();"
            "r.attach('k', object(), path='p');"
            "print(r.keys())",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['k']"


# --------------------------------------------------------------------------
# model sharing
# --------------------------------------------------------------------------


def test_a_second_graph_reuses_the_first_engine_s_models(monkeypatch):
    """The reason a tenth graph does not cost a gigabyte.

    Asserted on object identity rather than equality: two embedders holding
    the same weights would compare equal in every way that matters and still
    cost twice the memory.
    """
    from src.api.app import graph_loader

    built = []

    class RecordingEngine:
        def __init__(self, path, *, embedder=None, extractor=None, judge=None, now=None):
            self.path = path
            self.embedder = embedder
            self.extractor = extractor
            self.judge = judge
            self.now = now
            built.append(self)

        def __enter__(self):
            return self

    monkeypatch.setattr("src.engine.Engine", RecordingEngine)

    class FirstEngine:
        path = "first.db"
        embedder = object()
        extractor = object()
        judge = object()
        now = None

    first = FirstEngine()
    load = graph_loader(first)

    second = load("second.db")
    third = load("third.db")

    assert second.embedder is first.embedder
    assert third.embedder is first.embedder
    assert second.extractor is first.extractor
    assert second.embedder is third.embedder
    assert len(built) == 2


# ==========================================================================
# the one this process actually serves from
# ==========================================================================


def test_every_consumer_falls_back_to_one_shared_object():
    """Identity, not behaviour.

    Two registries that behave identically but are different objects is the
    exact failure the consumers used to refuse to default at all in order to
    prevent: a pass loads graphs into one, requests read the other, and every
    part looks like it worked. So this asserts the object, not what it does.
    """
    import src.pod as pod_module
    import src.reconcile as reconcile_module
    from src.api import app as app_module
    from src.registry import REGISTRY

    assert reconcile_module.REGISTRY is REGISTRY
    assert pod_module.REGISTRY is REGISTRY
    assert app_module.REGISTRY is REGISTRY


def test_startup_and_the_routes_resolve_through_that_same_object():
    """What startup attaches is what a request finds, by identity.

    Behaving the same is not enough: a second registry holding the same
    handles would pass any behavioural check and diverge the moment either
    side attached anything.
    """
    from fastapi.testclient import TestClient

    from src.api import auth as auth_module
    from src.api.app import create_app, engine_for
    from src.common.config import DEFAULT_TENANT_ORG_ID
    from src.registry import REGISTRY

    class FakeEngine:
        path = "fake-store.db"
        embedder = extractor = judge = now = None

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def close(self):
            pass

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(auth_module, "CLERK_ENABLED", False)
        patch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)

        app = create_app(engine_factory=FakeEngine)
        with TestClient(app) as client:
            attached = REGISTRY.get(DEFAULT_TENANT_ORG_ID)
            assert attached is not None

            class FakeRequest:
                pass

            request = FakeRequest()
            request.app = client.app

            assert engine_for(request, "any-tenant") is attached


class _EmptyRows:
    @staticmethod
    def all():
        return []


class _ReadingNothing:
    """A control plane with no assignments for this pod."""

    def exec(self, *args, **kwargs):
        return _EmptyRows()


class _Sessions:
    def __call__(self):
        return self

    def __enter__(self):
        return _ReadingNothing()

    def __exit__(self, *exc_info):
        return False


def test_the_reconcile_pass_with_no_registry_uses_the_process_one():
    """Defaulted now, not required — and to the shared one, not a fresh one.

    A fresh instance would be the original bug: graphs loaded where nothing
    reads them. So this attaches to the process registry, calls with no
    registry at all, and asserts what is still there afterwards.
    """
    import src.reconcile as reconcile_module
    from src.reconcile import reconcile
    from src.registry import REGISTRY

    handle = object()
    REGISTRY.attach("org_seen", handle, path="somewhere.db")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            reconcile_module, "control_plane_sessions", lambda engine: _Sessions()
        )
        # No registry argument. This used to raise.
        assert reconcile(engine=object()) == []

    assert REGISTRY.get("org_seen") is handle


def test_boot_with_no_registry_uses_the_process_one():
    import src.pod as pod_module
    from src.registry import REGISTRY

    class BrokenSession:
        def exec(self, *args, **kwargs):
            raise RuntimeError("nothing to read here")

    handle = object()
    REGISTRY.attach("org_seen", handle, path="somewhere.db")

    # No registry argument, and it does not raise for want of one.
    assert pod_module.hydrate(BrokenSession(), pod_id="pod_here") == []
    assert REGISTRY.get("org_seen") is handle


def test_the_loop_with_no_registry_hands_the_process_one_to_the_pass():
    """What the tick acts on is the process's registry, asserted by identity."""
    import asyncio

    import src.pod as pod_module
    from src.registry import REGISTRY

    seen = {}

    def watch(**kwargs):
        seen["registry"] = kwargs["registry"]
        return []

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pod_module, "reconcile", watch)

        async def drive():
            stop = asyncio.Event()
            # One tick, then stop: the wait wakes on the signal set below.
            task = asyncio.create_task(
                pod_module.poll(stop=stop, pod_id="pod_here", interval=0.01)
            )
            await asyncio.sleep(0.05)
            stop.set()
            await task

        asyncio.run(drive())

    assert seen["registry"] is REGISTRY


def test_an_explicitly_passed_registry_is_still_the_one_used():
    """The singleton is a fallback, not an override."""
    import src.reconcile as reconcile_module
    from src.reconcile import reconcile
    from src.registry import REGISTRY, GraphRegistry

    mine = GraphRegistry()
    assert mine is not REGISTRY

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            reconcile_module, "control_plane_sessions", lambda engine: _Sessions()
        )
        assert reconcile(engine=object(), registry=mine) == []


def test_boot_honours_a_registry_it_was_given():
    import src.pod as pod_module
    from src.registry import REGISTRY, GraphRegistry

    mine = GraphRegistry()
    opened: list[str] = []
    mine.set_loader(lambda path: opened.append(path))

    class BrokenSession:
        def exec(self, *args, **kwargs):
            raise RuntimeError("nothing to read here")

    assert pod_module.hydrate(BrokenSession(), pod_id="pod_here", registry=mine) == []
    # The process one was not touched in its place.
    assert REGISTRY.keys() == []


# -- the reset, proved as an ordered pair -----------------------------------


def test_something_is_left_in_the_process_registry():
    """Half of a pair, and it deliberately leaves state behind.

    The test after this one asserts the registry starts empty, which is only
    worth anything because this one ran first and put something in it.
    """
    from src.registry import REGISTRY

    REGISTRY.attach("org_left_behind", object(), path="left.db")

    assert REGISTRY.keys() == ["org_left_behind"]


def test_nothing_from_the_previous_test_survives_into_this_one():
    """The other half.

    Written as a pair rather than as a claim about a fixture: a fixture that
    quietly stopped running would leave every other test asserting its own
    setup, and nothing would say so.
    """
    from src.registry import REGISTRY

    assert REGISTRY.keys() == []
    assert REGISTRY.get("org_left_behind") is None
