"""Where the control plane lives, and how a process connects to it.

Two environment variables, in order
-----------------------------------

``GRAPHRAG_CONTROL_PLANE_DATABASE_URL`` names the control-plane database.
``GRAPHRAG_DATABASE_URL`` names the application's database generally, and is
used when the first is unset.

The fallback is what makes a one-database development setup a single variable
while a production deployment splits the two without touching code. The order
matters in one direction only: setting the specific one always wins, so a
deployment that has split them cannot be pulled back onto the shared database
by a general setting it did not mean to be read.

**There is no default.** Neither set is a failure that names both variables,
not a quiet fallback to a file in the working directory. A fallback would work
perfectly on the machine that wrote it and would, in the place this actually
runs, give every process its own private control plane — several machines each
convinced they alone hold the assignments, and no error anywhere.

Health-checked connections
--------------------------

The pool checks a connection before handing it out. A control plane is read on
the request path and written by a pass that runs on an interval, so
connections sit idle for long stretches — long enough for a server-side
timeout, a firewall, or a failover to have taken them away. Without the check
the process learns about it as an error on the next real query; with it, the
dead connection is discarded and replaced before the caller sees it.

Foreign keys under SQLite
-------------------------

SQLite enforces foreign keys only when asked, per connection, and the default
is off. Every connection this module opens turns them on, so a reference that
the schema declares is a reference the database actually checks. Left off, a
test suite running on SQLite would pass while the declared keys did nothing,
which is worse than not declaring them.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session

#: The control-plane database, and the general application database it falls
#: back to. Named here rather than in the configuration module because they
#: are read when a connection is opened, not when this module is imported —
#: a process that never touches the control plane must not fail at import
#: because a variable it does not need is unset.
CONTROL_PLANE_URL_VARIABLE = "GRAPHRAG_CONTROL_PLANE_DATABASE_URL"
DATABASE_URL_VARIABLE = "GRAPHRAG_DATABASE_URL"
GRAPH_STORE_URL_VARIABLE = "GRAPHRAG_GRAPH_STORE_DATABASE_URL"


class ControlPlaneNotConfigured(RuntimeError):
    """Neither environment variable names a database.

    Its own type so a caller can tell "nothing told me where to connect" from
    "I connected and the query failed". The first is fixed by setting a
    variable and the message says which; the second is not.
    """


def control_plane_url(environment: dict[str, str] | None = None) -> str:
    """The control-plane database URL, from the environment.

    Raises rather than inventing one. The failure names both variables,
    because a reader who has set the general one and expected it to be enough
    needs to see that it was consulted.
    """
    source = environment if environment is not None else os.environ

    for variable in (CONTROL_PLANE_URL_VARIABLE, DATABASE_URL_VARIABLE):
        value = (source.get(variable) or "").strip()
        if value:
            return value

    raise ControlPlaneNotConfigured(
        "the control plane has no database: set "
        f"{CONTROL_PLANE_URL_VARIABLE}, or {DATABASE_URL_VARIABLE} to share "
        "one database with the rest of the application. There is no default — "
        "a local file would give every process its own control plane and no "
        "error."
    )


def graph_store_url(environment: dict[str, str] | None = None) -> str | None:
    """Where tenants' accumulated graph rows live, when that is set apart.

    The graph store can be its own database: it grows with every tenant's
    history and is read in bulk by every build, while the control plane is a
    few small tables on the request path, and the two are sized and backed up
    differently. ``None`` means nothing names a separate one, and the rows live
    beside the control plane — the one-database setup a single variable gives.
    """
    source = environment if environment is not None else os.environ

    for variable in (GRAPH_STORE_URL_VARIABLE, DATABASE_URL_VARIABLE):
        value = (source.get(variable) or "").strip()
        if value:
            return value
    return None


def as_url(target: str | Path) -> str:
    """A URL from either a URL or a filesystem path.

    A caller that already has a URL passes it through untouched. A path
    becomes a local SQLite URL, which is what the tests use and what a single
    developer machine can use — the production path is a server, and it
    arrives here as a URL from the environment.
    """
    text = str(target)
    return text if "://" in text else f"sqlite+pysqlite:///{Path(text)}"


def create_control_plane_engine(
    target: str | Path | None = None, **options
) -> Engine:
    """An engine for the control plane.

    ``target`` may be a URL or a path; omitted, the environment decides and an
    unset environment raises.
    """
    url = as_url(target) if target is not None else control_plane_url()

    engine = create_engine(
        url,
        # A connection idle past the server's timeout is discarded and
        # reopened here rather than failing in the caller's query.
        pool_pre_ping=True,
        **options,
    )

    if engine.dialect.name == "sqlite":
        _enforce_sqlite_foreign_keys(engine)

    return engine


def create_graph_store_engine(target: str | Path, **options) -> Engine:
    """An engine for a graph store kept in its own database.

    Built the same way as the control plane's — health-checked connections,
    foreign keys on under SQLite — because it is the same kind of database
    used the same way, only holding different tables.
    """
    return create_control_plane_engine(target, **options)


def _enforce_sqlite_foreign_keys(engine: Engine) -> None:
    """Turn foreign keys on for every connection this engine opens.

    Per connection, because that is the only scope SQLite offers, and on every
    one, because a declared reference that is checked on some connections and
    not others is worse than one that is never checked.
    """

    @event.listens_for(engine, "connect")
    def _set_pragma(connection, _record):  # pragma: no branch - one statement
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def control_plane_sessions(engine: Engine) -> sessionmaker[Session]:
    """Sessions bound to ``engine``.

    The session type is the one that goes with the models, so a select over a
    model hands back model instances rather than rows a caller has to unpack.

    ``expire_on_commit`` stays off so an object read inside a session is still
    readable after the commit that closed it. Every caller here reads a row,
    commits and hands the result outward; with expiry on, that last step is a
    lazy load against a session that is gone.
    """
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
