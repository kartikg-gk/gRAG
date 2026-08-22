"""Who is asking, and whose data they are asking about.

Two dependencies, two credentials, two stores, and no path between them.

    Authorization: Bearer <session JWT>   ->  get_current_user      -> user_id
    X-API-Key: <api key>                  ->  get_current_tenant_org -> org_id

A verified user id never implies an organisation and an organisation never
implies a user. That is the whole point: the first says a person is who they
claim to be, the second says a request is entitled to a particular tenant's
data. A system that derives one from the other has one check wearing two hats,
and the day the hats disagree is a cross-tenant read.

Why the API key is not in the Authorization header
--------------------------------------------------

Both credentials want to be a bearer token, and a request carries one
``Authorization`` header. A route asking for both — which is the normal case —
would have to guess which credential it received and try the other on failure,
and "try it as a session token, then as an API key" is a downgrade attack
waiting for a token that parses as both.

So the key has its own header. ``Authorization: Bearer <key>`` is still
accepted for it, but **only when session verification is off**, where nothing
else is competing for the header — that keeps the single-credential
development setup working without creating the ambiguity in the deployed one.

Failures
--------

Every rejection is the same 401 with the same body. A client learns that its
credential was not accepted and nothing else — not whether a key exists, not
whether it was revoked, not whether the issuer was wrong. The distinctions all
exist, and they all go to the log, where the operator can see them and the
caller cannot.

Server faults are the exception worth naming: a JWKS endpoint that will not
answer and a control plane that will not read are not the client's fault, and
they are logged as errors rather than as rejected credentials.
"""

from __future__ import annotations

import hmac
import logging
import threading
from typing import Any

from fastapi import Depends, HTTPException, Request, status

from ..common.config import (
    CLERK_ALGORITHM,
    CLERK_AUTHORIZED_PARTIES,
    CLERK_ENABLED,
    CLERK_ISSUER,
    CLERK_JWKS_URL,
    CLERK_LEEWAY_SECONDS,
    CONTROL_PLANE_PATH,
    DEFAULT_TENANT_ORG_ID,
    DEV_USER_ID,
    JWKS_CACHE_SECONDS,
    MULTI_TENANCY_ENABLED,
)
from .control_plane import ControlPlaneError, hash_api_key, open_control_plane, verify_key
from .tenancy import reset_current_org, set_current_org

#: The first logging in this project. Authentication is the one place where
#: "it worked" and "it was rejected, here is why" have to be visible on the
#: server without being visible to the caller, and stderr prints from a
#: request path are not that.
logger = logging.getLogger("graphrag.api.auth")

#: One body for every rejection. Built once so no branch can accidentally
#: return a more helpful one.
INVALID_CREDENTIALS = "invalid authentication credentials"

API_KEY_HEADER = "X-API-Key"

_jwks_lock = threading.Lock()
_jwks_clients: dict[str, Any] = {}

_control_plane_lock = threading.Lock()
_control_plane = None


def _unauthorized() -> HTTPException:
    """The only rejection this module produces."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=INVALID_CREDENTIALS,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _report(message: str, exc: BaseException | None = None) -> None:
    """Log, and hand the failure to telemetry if any is installed.

    No telemetry package is a dependency of this project. If a deployment has
    installed one, it gets the event; if the report itself fails, that failure
    is swallowed, because an authentication path that breaks when its
    monitoring breaks is worse than one that is unmonitored.
    """
    logger.error(message, exc_info=exc)
    try:  # pragma: no cover - depends on what the deployment installed
        import sentry_sdk
    except Exception:
        return
    try:  # pragma: no cover
        if exc is not None:
            sentry_sdk.capture_exception(exc)
        else:
            sentry_sdk.capture_message(message)
    except Exception:
        pass


# --------------------------------------------------------------------------
# credentials off the wire
# --------------------------------------------------------------------------


def bearer_token(request: Request) -> str | None:
    """The token in ``Authorization: Bearer <token>``, or ``None``.

    The scheme is compared case-insensitively because the standard says it is
    case-insensitive, and exactly one space is required because a header with
    two is malformed rather than generous.
    """
    header = request.headers.get("Authorization")
    if not header:
        return None

    scheme, separator, token = header.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return None

    token = token.strip()
    return token or None


def api_key_from(request: Request) -> str | None:
    """The API key, from its own header or — only in single-credential mode —
    from ``Authorization``. See the module docstring for why that is
    conditional."""
    key = request.headers.get(API_KEY_HEADER)
    if key and key.strip():
        return key.strip()
    if not CLERK_ENABLED:
        return bearer_token(request)
    return None


# --------------------------------------------------------------------------
# session tokens
# --------------------------------------------------------------------------


def _jwks_client(url: str):
    """The cached key client for ``url``.

    Cached because the alternative is fetching a JWKS document on every
    request, which turns the identity provider into a hard dependency of every
    call and its latency into a floor under all of them. The client caches
    individual keys by ``kid``, so rotation costs one fetch rather than a
    restart: an unrecognised ``kid`` misses the cache and is looked up.
    """
    with _jwks_lock:
        client = _jwks_clients.get(url)
        if client is None:
            from jwt import PyJWKClient

            client = PyJWKClient(
                url,
                cache_keys=True,
                lifespan=int(JWKS_CACHE_SECONDS),
            )
            _jwks_clients[url] = client
        return client


def reset_jwks_cache() -> None:
    """Drop cached key clients. For tests, and for a forced re-fetch."""
    with _jwks_lock:
        _jwks_clients.clear()


def verify_session_token(token: str) -> dict:
    """Verified claims, or ``HTTPException(401)``.

    Nothing in the token is trusted before the signature is checked. The
    ``kid`` is read from the header to *find* a key, which is the one piece of
    unverified input this function acts on, and acting on it can only select
    which public key to try — a forged ``kid`` matching nothing fails, and one
    matching a real key still has to survive that key's signature check.

    The algorithm is fixed at the call, not read from the token. A token
    claiming ``alg: none`` or an HMAC algorithm is rejected before any key is
    involved, because the accepted list has exactly one entry.
    """
    import jwt

    if not CLERK_ISSUER or not CLERK_JWKS_URL:
        # Refuses to build rather than verifying against a trust anchor nobody
        # configured. Same rule as the model judges: no default issuer.
        _report(
            "session verification is enabled but GRAPHRAG_CLERK_ISSUER or "
            "GRAPHRAG_CLERK_JWKS_URL is unset; refusing to verify"
        )
        raise _unauthorized()

    try:
        signing_key = _jwks_client(CLERK_JWKS_URL).get_signing_key_from_jwt(token)
    except jwt.PyJWTError as exc:
        # Covers a malformed token whose header cannot be read, and a kid that
        # resolves to nothing.
        logger.warning("could not resolve a signing key: %s", exc)
        raise _unauthorized() from exc
    except Exception as exc:
        # The endpoint is unreachable or answered with something unusable.
        # A server fault, not a bad credential, and logged as one.
        _report(f"JWKS resolution failed for {CLERK_JWKS_URL}", exc)
        raise _unauthorized() from exc

    try:
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=[CLERK_ALGORITHM],
            issuer=CLERK_ISSUER,
            leeway=CLERK_LEEWAY_SECONDS,
            options={
                "require": ["exp", "iat", "sub"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                # No audience is configured for this surface. Verifying against
                # an unset audience would reject everything; saying so here
                # keeps it a decision rather than a default.
                "verify_aud": False,
            },
        )
    except jwt.PyJWTError as exc:
        logger.warning("token rejected: %s", type(exc).__name__)
        raise _unauthorized() from exc

    if CLERK_AUTHORIZED_PARTIES:
        party = claims.get("azp")
        if not party or not any(
            hmac.compare_digest(party, allowed) for allowed in CLERK_AUTHORIZED_PARTIES
        ):
            logger.warning("token rejected: azp not in the authorized parties")
            raise _unauthorized()

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        # `require` already caught an absent sub; this catches a present one
        # that is empty or not a string, which would otherwise become a
        # falsy user id downstream.
        logger.warning("token rejected: unusable sub claim")
        raise _unauthorized()

    return claims


async def get_current_user(request: Request) -> str:
    """The verified user id for this request.

    The id comes from the token's ``sub`` and from nowhere else. No header,
    query parameter or body field can supply it, so a client cannot name a
    user it is not.
    """
    if not CLERK_ENABLED:
        # Loud on every request, deliberately. A single startup line scrolls
        # away; this cannot be mistaken for an authenticated deployment while
        # reading a log.
        logger.warning(
            "session verification is DISABLED: request runs as %s and no token "
            "was checked",
            DEV_USER_ID,
        )
        request.state.user_id = DEV_USER_ID
        return DEV_USER_ID

    token = bearer_token(request)
    if token is None:
        raise _unauthorized()

    claims = verify_session_token(token)
    user_id = claims["sub"]
    request.state.user_id = user_id
    return user_id


# --------------------------------------------------------------------------
# tenants
# --------------------------------------------------------------------------


def control_plane():
    """The process-wide control plane handle, opened on first use."""
    global _control_plane
    with _control_plane_lock:
        if _control_plane is None:
            _control_plane = open_control_plane(CONTROL_PLANE_PATH)
        return _control_plane


def set_control_plane(plane) -> None:
    """Replace the handle. For tests, and for an application that owns its own."""
    global _control_plane
    with _control_plane_lock:
        _control_plane = plane


def resolve_org(api_key: str) -> str:
    """The organisation this key belongs to, or ``HTTPException(401)``.

    Missing, unknown, mismatched and revoked all leave by the same door with
    the same message. They are separated only in the log.
    """
    hashed = hash_api_key(api_key)

    try:
        record = control_plane().record_for_hash(hashed)
    except ControlPlaneError as exc:
        # Not a bad key — the store could not answer. Still a 401 to the
        # client, because handing back a 500 tells an attacker their key
        # reached a real lookup.
        _report("control plane lookup failed", exc)
        raise _unauthorized() from exc

    if record is None:
        logger.warning("api key rejected: no record for the presented key")
        raise _unauthorized()

    if record.is_revoked:
        # Worth its own line: a revoked key still in use means a credential
        # was not rotated out of a running client, or was stolen.
        logger.warning(
            "api key rejected: key_id=%s for org_id=%s was revoked at %s",
            record.key_id,
            record.org_id,
            record.revoked_at,
        )
        raise _unauthorized()

    if not verify_key(record, hashed):
        logger.warning("api key rejected: key_id=%s failed verification", record.key_id)
        raise _unauthorized()

    return record.org_id


async def get_current_tenant_org(request: Request):
    """The organisation whose data this request may touch.

    Yields rather than returns so the context variable is unbound when the
    request ends. A task reused by the next request must not start with this
    one's organisation still set.
    """
    if not MULTI_TENANCY_ENABLED:
        org_id = DEFAULT_TENANT_ORG_ID
    else:
        api_key = api_key_from(request)
        if api_key is None:
            raise _unauthorized()
        org_id = resolve_org(api_key)

    request.state.org_id = org_id
    token = set_current_org(org_id)
    try:
        yield org_id
    finally:
        reset_current_org(token)


__all__ = [
    "API_KEY_HEADER",
    "Depends",
    "api_key_from",
    "bearer_token",
    "control_plane",
    "get_current_tenant_org",
    "get_current_user",
    "reset_jwks_cache",
    "resolve_org",
    "set_control_plane",
    "verify_session_token",
]
