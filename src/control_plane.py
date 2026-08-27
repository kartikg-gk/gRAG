"""Issuing an API key, and finding out which organisation one belongs to.

The table this reads is declared in ``src.models``; this is the small amount
of behaviour that surrounds it — hashing, issuing, revoking, and the
comparison that decides whether a presented key is live. It is the only code
that turns a raw key into anything stored, so issuing and lookup cannot drift
into hashing differently and silently failing every comparison.

What is stored
--------------

**Never the key.** Only its SHA-256 digest, so a copy of this database does not
let the holder authenticate as anyone. A raw key exists exactly once, in the
response that issues it, and is never written down here.

The digest is not a password hash and does not want to be one. Argon2 or
bcrypt defend a low-entropy secret a human chose against offline guessing. An
API key here is 32 random bytes from ``secrets``; there is nothing to guess,
and a deliberately slow hash would add its cost to every request while
defending against an attack that cannot succeed.

Lookup is by digest, then the digest is compared again with
``hmac.compare_digest``. The second comparison is not redundant paranoia about
the database's index — it is what keeps the *verification* free of the
early-exit that ``==`` on strings has, so the code does not have to argue about
whether the engine leaked timing on the way in.

Where it connects
-----------------

A server-backed database named by the environment, because several processes
on different machines share this control plane — which is what the pod and
assignment tables are for. A caller may hand a URL or a path instead, which is
what the tests do; nothing falls back to a local file on its own.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Engine, update
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import select

from .models.control_plane import (
    DEFAULT_SCOPES,
    ApiKey,
    create_control_plane_schema,
)
from .models.database import (
    ControlPlaneNotConfigured,
    control_plane_sessions,
    create_control_plane_engine,
)

#: Bytes of randomness in an issued key. 32 bytes is 256 bits; guessing is not
#: a threat model at that width, which is what lets the digest stay fast.
KEY_BYTES = 32

#: Characters of the raw key kept as a human-readable handle.
#:
#: A key is 32 random bytes rendered as roughly 43 URL-safe characters, so
#: eight leaves about 35 unknown -- far more than enough that the remainder
#: cannot be searched. It is a label for telling two keys apart in a list, not
#: a fragment of the secret in any useful sense.
PREFIX_LENGTH = 8


class ControlPlaneError(RuntimeError):
    """The control plane could not answer.

    Distinct from "the key is not valid". A caller must not turn a database
    failure into a rejection with a reason, because the two need different
    responses: one is a 401 the client can act on, the other is a server fault
    the operator has to see.
    """


class ControlPlaneUnconfigured(ControlPlaneError, ControlPlaneNotConfigured):
    """Nothing said where the control plane is.

    Both types on purpose. It is a control-plane failure, so every caller
    already handling one handles this; and it is the configuration error the
    database module raises, so the message naming the two variables survives.
    """


def hash_api_key(raw: str) -> str:
    """The stored form of an API key.

    SHA-256 over the UTF-8 bytes, hex. The only function that decides what a
    stored key looks like — issuing and lookup both call it.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def new_api_key() -> str:
    """A fresh key. Returned once and never stored in this form."""
    return secrets.token_urlsafe(KEY_BYTES)


@dataclass(frozen=True)
class ApiKeyRecord:
    """One stored credential, detached from the session that read it.

    A plain frozen object rather than the model instance. Callers hold this
    across the request and must not be able to change a credential by
    assigning to it, and it must stay readable after the session closes.
    """

    key_id: str
    hashed_key: str
    org_id: str
    #: What this key may do. Recorded at issuance and not yet enforced
    #: anywhere -- see ``DEFAULT_SCOPES``.
    scopes: str = DEFAULT_SCOPES
    #: The first few characters of the raw key. Safe to store and safe to
    #: show: it identifies which key this is without being the key.
    prefix: str | None = None
    created_at: datetime | None = None
    revoked_at: datetime | None = None

    @property
    def is_revoked(self) -> bool:
        return self.revoked_at is not None


class ControlPlane:
    """Credential lookup, and the issuing side used to provision one.

    Holds an engine, not a connection. The pool hands one out per operation
    and checks it is alive first, so a process that has been idle since before
    the database's connection timeout does not fail its next request.
    """

    def __init__(self, target: str | Path | None = None, **options) -> None:
        try:
            self._engine = create_control_plane_engine(target, **options)
        except ControlPlaneNotConfigured as exc:
            raise ControlPlaneUnconfigured(str(exc)) from exc
        except SQLAlchemyError as exc:
            raise ControlPlaneError(f"could not open the control plane: {exc}") from exc
        self._sessions = control_plane_sessions(self._engine)

    @property
    def engine(self) -> Engine:
        """The engine, for code that reads the other control-plane tables."""
        return self._engine

    def initialize_schema(self) -> None:
        try:
            create_control_plane_schema(self._engine)
        except SQLAlchemyError as exc:
            raise ControlPlaneError(f"could not initialise: {exc}") from exc

    # -- reading ------------------------------------------------------------

    def record_for_hash(self, hashed_key: str) -> ApiKeyRecord | None:
        """The record with this digest, or ``None``.

        ``None`` means no such key. It does not mean "revoked" and it does not
        mean "the database is down" — the second raises, and the caller is
        responsible for making all three look identical to the client while
        staying distinguishable in the log.
        """
        try:
            with self._sessions() as session:
                row = session.exec(
                    select(ApiKey).where(ApiKey.hashed_key == hashed_key)
                ).first()
        except SQLAlchemyError as exc:
            raise ControlPlaneError(f"lookup failed: {exc}") from exc

        return _record(row) if row is not None else None

    # -- writing ------------------------------------------------------------

    def issue(
        self,
        org_id: str,
        *,
        key_id: str | None = None,
        scopes: str = DEFAULT_SCOPES,
    ) -> tuple[str, ApiKeyRecord]:
        """Mint a key for ``org_id``. Returns ``(raw_key, record)``.

        The raw key is returned and not kept. This is the only moment it
        exists, which is why the caller has to hand it to its owner now or
        issue another one.

        ``org_id`` is a declared reference, so an organisation that does not
        exist is a failure here rather than a working credential nobody can
        resolve. Provision the tenant first.
        """
        raw = new_api_key()
        hashed = hash_api_key(raw)
        identifier = key_id or secrets.token_hex(8)
        created = int(datetime.now(timezone.utc).timestamp())
        # Stored beside the digest so a list of keys can be told apart. This
        # is the only part of the raw key that outlives this call.
        prefix = raw[:PREFIX_LENGTH]

        try:
            with self._sessions() as session:
                session.add(
                    ApiKey(
                        key_id=identifier,
                        hashed_key=hashed,
                        org_id=org_id,
                        scopes=scopes,
                        prefix=prefix,
                        created_at=created,
                        revoked_at=None,
                    )
                )
                session.commit()
        except SQLAlchemyError as exc:
            raise ControlPlaneError(f"could not issue a key: {exc}") from exc

        return raw, ApiKeyRecord(
            key_id=identifier,
            hashed_key=hashed,
            org_id=org_id,
            scopes=scopes,
            prefix=prefix,
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
        # One statement rather than a read followed by a write: two callers
        # revoking the same key at once must not both be told they were the
        # one who did it.
        try:
            with self._sessions() as session:
                result = session.execute(
                    update(ApiKey)
                    .where(ApiKey.key_id == key_id, ApiKey.revoked_at.is_(None))
                    .values(revoked_at=moment)
                )
                session.commit()
        except SQLAlchemyError as exc:
            raise ControlPlaneError(f"could not revoke: {exc}") from exc
        return result.rowcount > 0

    def close(self) -> None:
        try:
            self._engine.dispose()
        except SQLAlchemyError:
            pass

    def __enter__(self) -> "ControlPlane":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _record(row: ApiKey) -> ApiKeyRecord:
    return ApiKeyRecord(
        key_id=row.key_id,
        hashed_key=row.hashed_key,
        org_id=row.org_id,
        scopes=row.scopes or DEFAULT_SCOPES,
        prefix=row.prefix,
        created_at=_moment(row.created_at),
        revoked_at=_moment(row.revoked_at),
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


def open_control_plane(
    target: str | Path | None = None, *, initialize: bool = True
) -> ControlPlane:
    """Open the control plane, ready to use.

    ``target`` may be a database URL or a path; omitted, the environment says
    where, and an environment that says nothing is an error rather than a
    local file.

    Mirrors ``open_context_graph``: construct, bring the schema up, and close
    on any failure rather than handing back a half-open handle.
    """
    plane = ControlPlane(target)
    try:
        if initialize:
            plane.initialize_schema()
    except Exception:
        plane.close()
        raise
    return plane
