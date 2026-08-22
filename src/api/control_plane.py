"""The mapping from an API key to the organisation it belongs to.

A separate database from the graph store, and the separation is the design.
The graph holds one organisation's entities and documents; this holds the
credentials that decide which graph a request is allowed to open. Putting both
in one file makes a cross-tenant read one join away from a bug.

SQLite, from the standard library. The control plane is a few rows read once
per request on an exact-match index; it needs no server, no driver, and no
entry in the dependency list. The graph store earns an embedded graph engine
because it does traversal and vector search — this does neither.

What is stored
--------------

**Never the key.** Only its SHA-256 digest, so a copy of this database does not
let the holder authenticate as anyone. A raw key exists exactly once, in the
response that issues it, and is never written down here.

The digest is not a password hash and does not want to be one. Argon2 or bcrypt
defend a low-entropy secret a human chose against offline guessing. An API key
here is 32 random bytes from ``secrets``; there is nothing to guess, and a
deliberately slow hash would add its cost to every request while defending
against an attack that cannot succeed.

Lookup is by digest, then the digest is compared again with
``hmac.compare_digest``. The second comparison is not redundant paranoia about
SQLite's index — it is what keeps the *verification* free of the early-exit
that ``==`` on strings has, so the code does not have to argue about whether
the engine leaked timing on the way in.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

#: Bytes of randomness in an issued key. 32 bytes is 256 bits; guessing is not
#: a threat model at that width, which is what lets the digest stay fast.
KEY_BYTES = 32

#: One table. A key belongs to exactly one organisation and is either live or
#: revoked, and neither fact needs a second table to express.
SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_id     TEXT PRIMARY KEY,
    hashed_key TEXT NOT NULL UNIQUE,
    org_id     TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    revoked_at INTEGER
);
CREATE INDEX IF NOT EXISTS api_keys_org ON api_keys (org_id);
"""


class ControlPlaneError(RuntimeError):
    """The control plane could not answer.

    Distinct from "the key is not valid". A caller must not turn a database
    failure into a rejection with a reason, because the two need different
    responses: one is a 401 the client can act on, the other is a server fault
    the operator has to see.
    """


def hash_api_key(raw: str) -> str:
    """The stored form of an API key.

    SHA-256 over the UTF-8 bytes, hex. The only function that decides what a
    stored key looks like — issuing and lookup both call it, so the two cannot
    drift into hashing differently and silently failing every comparison.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def new_api_key() -> str:
    """A fresh key. Returned once and never stored in this form."""
    return secrets.token_urlsafe(KEY_BYTES)


@dataclass(frozen=True)
class ApiKeyRecord:
    """One stored credential. Carries no secret."""

    key_id: str
    hashed_key: str
    org_id: str
    created_at: datetime | None = None
    revoked_at: datetime | None = None

    @property
    def is_revoked(self) -> bool:
        return self.revoked_at is not None


class ControlPlane:
    """Credential lookup, and the issuing side used to provision one.

    One connection guarded by a lock. Synchronous routes run in a threadpool,
    so more than one thread reaches this; SQLite connections are not safe to
    share across threads without saying so, and a lock around a single
    exact-match read is cheaper than a pool for a table this small.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        try:
            self._connection = sqlite3.connect(self._path, check_same_thread=False)
        except sqlite3.Error as exc:
            raise ControlPlaneError(f"could not open the control plane: {exc}") from exc
        self._connection.row_factory = sqlite3.Row

    def initialize_schema(self) -> None:
        with self._lock:
            try:
                self._connection.executescript(SCHEMA)
                self._connection.commit()
            except sqlite3.Error as exc:
                raise ControlPlaneError(f"could not initialise: {exc}") from exc

    # -- reading ------------------------------------------------------------

    def record_for_hash(self, hashed_key: str) -> ApiKeyRecord | None:
        """The record with this digest, or ``None``.

        ``None`` means no such key. It does not mean "revoked" and it does not
        mean "the database is down" — the second raises, and the caller is
        responsible for making all three look identical to the client while
        staying distinguishable in the log.
        """
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT key_id, hashed_key, org_id, created_at, revoked_at "
                    "FROM api_keys WHERE hashed_key = ?",
                    (hashed_key,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise ControlPlaneError(f"lookup failed: {exc}") from exc

        return _record(row) if row is not None else None

    # -- writing ------------------------------------------------------------

    def issue(self, org_id: str, *, key_id: str | None = None) -> tuple[str, ApiKeyRecord]:
        """Mint a key for ``org_id``. Returns ``(raw_key, record)``.

        The raw key is returned and not kept. This is the only moment it
        exists, which is why the caller has to hand it to its owner now or
        issue another one.
        """
        raw = new_api_key()
        hashed = hash_api_key(raw)
        identifier = key_id or secrets.token_hex(8)
        created = int(datetime.now(timezone.utc).timestamp())

        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO api_keys (key_id, hashed_key, org_id, created_at, "
                    "revoked_at) VALUES (?, ?, ?, ?, NULL)",
                    (identifier, hashed, org_id, created),
                )
                self._connection.commit()
            except sqlite3.Error as exc:
                raise ControlPlaneError(f"could not issue a key: {exc}") from exc

        return raw, ApiKeyRecord(
            key_id=identifier,
            hashed_key=hashed,
            org_id=org_id,
            created_at=datetime.fromtimestamp(created, timezone.utc),
        )

    def revoke(self, key_id: str, *, when: datetime | None = None) -> bool:
        """Mark a key revoked. Returns whether a live key was actually revoked.

        Revocation sets a time rather than deleting the row: a deleted key and
        a key that never existed are indistinguishable afterwards, and the
        question "was this revoked, and when" is exactly the one asked after an
        incident.
        """
        moment = int((when or datetime.now(timezone.utc)).timestamp())
        with self._lock:
            try:
                cursor = self._connection.execute(
                    "UPDATE api_keys SET revoked_at = ? "
                    "WHERE key_id = ? AND revoked_at IS NULL",
                    (moment, key_id),
                )
                self._connection.commit()
            except sqlite3.Error as exc:
                raise ControlPlaneError(f"could not revoke: {exc}") from exc
        return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            try:
                self._connection.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> "ControlPlane":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _record(row: sqlite3.Row) -> ApiKeyRecord:
    return ApiKeyRecord(
        key_id=row["key_id"],
        hashed_key=row["hashed_key"],
        org_id=row["org_id"],
        created_at=_moment(row["created_at"]),
        revoked_at=_moment(row["revoked_at"]),
    )


def _moment(seconds: int | None) -> datetime | None:
    """Unknown stays ``None`` rather than becoming 1970."""
    if seconds is None:
        return None
    return datetime.fromtimestamp(int(seconds), timezone.utc)


def verify_key(record: ApiKeyRecord | None, hashed_key: str) -> bool:
    """Whether ``record`` really is the credential for ``hashed_key``, and live.

    Constant-time on the digest comparison, and revocation checked here rather
    than at the call site so no caller can verify a key and forget to ask.
    """
    if record is None:
        return False
    if not hmac.compare_digest(record.hashed_key, hashed_key):
        return False
    return not record.is_revoked


def open_control_plane(path: str | Path, *, initialize: bool = True) -> ControlPlane:
    """Open the control plane, ready to use.

    Mirrors ``open_context_graph``: construct, bring the schema up, and close
    on any failure rather than handing back a half-open handle.
    """
    plane = ControlPlane(path)
    try:
        if initialize:
            plane.initialize_schema()
    except Exception:
        plane.close()
        raise
    return plane
