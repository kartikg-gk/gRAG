"""Settings from a file, for a process started without them exported.

Every setting is read from the environment, most of them at import. A
process started with ``python -m``, ``uvicorn`` or ``celery`` sees only what
its shell exported, so a value written to ``.env`` and nowhere else is absent —
and absent means a default, or a raise far from the cause.

Called by both configuration modules before they read anything, which is the
one point every entry point already passes through.

What wins
---------

An exported variable always wins over the file. The file holds a checkout's
defaults; an exported value is a decision made for one process, and a loader
that overwrote it would make ``GRAPHRAG_X=1 python ...`` do nothing.

Which file
----------

``.env`` in the working directory, where ``docker compose`` reads it from.
``GRAPHRAG_ENV_FILE`` names another, and an empty value loads none. A named
file that does not exist raises: naming a file says it exists, and loading
nothing in its place is a silent default. A missing ``.env`` does not, because
a checkout with nothing configured is the ordinary starting state.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Names the file to load, or, set to an empty value, that none is.
ENV_FILE_VARIABLE = "GRAPHRAG_ENV_FILE"

#: Read from the working directory, as ``docker compose`` reads it.
DEFAULT_ENV_FILE = ".env"


def load_environment() -> Path | None:
    """Load the environment file into ``os.environ``, never overriding.

    Returns the file loaded, or ``None`` when there was none to load. Calling
    it twice is harmless: the second call finds every variable already set.
    """
    named = os.environ.get(ENV_FILE_VARIABLE)
    if named is not None and not named.strip():
        return None

    if named is not None:
        path = Path(named.strip())
        if not path.is_file():
            raise FileNotFoundError(
                f"{ENV_FILE_VARIABLE} names {path}, which does not exist"
            )
    else:
        path = Path(DEFAULT_ENV_FILE)
        if not path.is_file():
            return None

    # Imported only when there is a file to read, so a process with nothing to
    # load pays nothing for the parser.
    from dotenv import load_dotenv

    load_dotenv(path, override=False)
    return path
