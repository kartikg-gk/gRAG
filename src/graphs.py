"""Which graph files this checkout has, and what to call them.

Two functions that do not know about each other's job. One finds paths; one
turns a path into something to show a person. Keeping them apart means the
label rule can change without touching discovery, and discovery can be tested
without any opinion about what a label looks like.

Neither opens anything, and neither imports the engine or the store.

**A store here is a single file, not a directory.** Checked rather than
assumed: opening one at ``sample.db`` leaves exactly that file beside its
siblings, with nothing inside it. So enumeration is a glob, which is the
simple case — a directory-per-store would have needed a test for "is this a
store or just a folder" that a glob does not.

Where they are found
--------------------

A fixed directory beside the source tree, not a configured value. This is a
local affordance — somebody keeping a graph per repository and moving between
them — and a deployment knob for it would be one more thing to set for a
feature that only makes sense when you are sitting in the checkout.

The configured default store comes first even though it lives outside that
directory, because it is selectable and it is the one already open. Listing it
anywhere else would put the graph you are currently using in the middle of the
list, ordered by a filename it does not share a directory with.
"""

from __future__ import annotations

from pathlib import Path

from .common.config import STORE_PATH

#: Beside the source tree: ``src/`` lives in the project root, and so does
#: this. Fixed rather than configured — see the module docstring.
GRAPHS_DIRECTORY = Path(__file__).resolve().parents[1] / "graphs"

#: What a store looks like on disk. One file, verified rather than assumed.
STORE_PATTERN = "*.db"

#: Appended to the label of whichever file is the configured default, so a
#: reader can tell at a glance which one that is without comparing paths.
DEFAULT_MARKER = " (default)"

#: A filename cannot hold a forward slash, so a repository name is written
#: with a doubled underscore and read back with a slash. That lets
#: ``owner__repository.db`` round-trip to ``owner/repository``.
NAME_SEPARATOR = "__"


def graph_paths(
    *, directory: Path | None = None, default: str | Path | None = None
) -> list[Path]:
    """Every graph this checkout can serve, in the order to offer them.

    The configured default first when it exists, then everything matching the
    store pattern in the graphs directory, sorted.

    **Deduplicated by resolved path, preserving that order.** A default store
    that also happens to live in the graphs directory is one entry at the
    front, not one at the front and another in the middle — and the comparison
    has to be on the resolved path, because the same file reached by two
    different spellings is still one graph.

    Resolution is for the comparison only. The paths come back as found, so a
    caller sees what it configured rather than an absolute rewrite of it.
    """
    root = Path(directory) if directory is not None else GRAPHS_DIRECTORY
    configured = Path(default) if default is not None else Path(STORE_PATH)

    found: list[Path] = []
    if configured.exists():
        found.append(configured)
    if root.is_dir():
        found.extend(sorted(root.glob(STORE_PATTERN)))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        try:
            key = path.resolve()
        except OSError:
            # Unresolvable is still listable: a path that cannot be resolved
            # is a path this cannot prove is a duplicate, and dropping it
            # would silently hide a graph.
            key = path.absolute()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def graph_label(path: str | Path, *, default: str | Path | None = None) -> str:
    """What to show a person for ``path``.

    The filename without its extension, with two adjustments: the configured
    default is marked as such, and a doubled underscore becomes a slash so a
    repository name reads the way it was written.

    The two are exclusive — the default is named by being the default, and
    marking it is more use than un-escaping a name it probably does not have.
    """
    path = Path(path)
    configured = Path(default) if default is not None else Path(STORE_PATH)

    stem = path.stem
    try:
        is_default = path.resolve() == configured.resolve()
    except OSError:
        is_default = path.absolute() == configured.absolute()

    if is_default:
        return stem + DEFAULT_MARKER
    return stem.replace(NAME_SEPARATOR, "/")
