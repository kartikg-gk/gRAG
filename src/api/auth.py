"""Who is asking, and whose data they are asking about.

Two dependencies, two credentials, two stores, and no path between them.

    Authorization: Bearer <session JWT>   ->  get_current_user      -> user_id
    Authorization: Bearer <api key>       ->  get_current_tenant_org -> org_id

A verified user id never implies an organisation and an organisation never
implies a user. That is the whole point: the first says a person is who they
claim to be, the second says a request is entitled to a particular tenant's
data. A system that derives one from the other has one check wearing two hats,
and the day the hats disagree is a cross-tenant read.

Both credentials travel as a bearer token
-----------------------------------------

The session token and the API key each arrive as ``Authorization: Bearer``.
Each dependency reads the header for its own credential; neither falls back to
trying the other's.

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
    DEFAULT_TENANT_ORG_ID,
    DEV_USER_ID,
    JWKS_CACHE_SECONDS,
    MULTI_TENANCY_ENABLED,
)
from ..control_plane import (
    ControlPlaneError,
    hash_api_key,
    open_control_plane,
    verify_key,
)
from .tenancy import reset_current_org, set_current_org

#: The first logging in this project. Authentication is the one place where
#: "it worked" and "it was rejected, here is why" have to be visible on the
#: server without being visible to the caller, and stderr prints from a
#: request path are not that.
logger = logging.getLogger("graphrag.api.auth")

#: One body for every rejection. Built once so no branch can accidentally
#: return a more helpful one.
INVALID_CREDENTIALS = "invalid authentication credentials"

#: Seconds of clock skew tolerated on ``exp`` and ``iat``.
#:
#: **Fixed in code, deliberately not configurable.** Leeway is how long an
#: expired token keeps working, so it is a security property rather than a
#: deployment preference — and a value read from the environment is a value
#: that can be widened by whoever sets the environment, without the change
#: appearing in any diff. Five seconds covers ordinary clock drift between two
#: machines and is short enough that an expired token stays expired.
LEEWAY_SECONDS = 5

#: Whether the fail-open warning has already been emitted in this process.
#: The warning has to be impossible to miss and impossible to drown in, and
#: those pull opposite ways at one line per request.
_unconfigured_warning_emitted = False

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


def _warn_unconfigured_once() -> None:
    """Say once, loudly, that this process is not verifying anything.

    Once rather than per request. A line on every call is a line that scrolls
    a busy log until nobody sees it, and the thing being reported is a
    property of the configuration rather than of any one request — it does not
    become more true by being repeated.

    Startup is where this lands in practice, because the first request is
    usually a probe.
    """
    global _unconfigured_warning_emitted
    if _unconfigured_warning_emitted:
        return
    _unconfigured_warning_emitted = True
    logger.warning(
        "SESSION VERIFICATION IS OFF: no issuer is configured, so every "
        "request runs as %s and no token is checked. Set %s to verify.",
        DEV_USER_ID,
        "GRAPHRAG_CLERK_ISSUER",
    )


def reset_unconfigured_warning() -> None:
    """Let the warning fire again. For tests, and for a re-read of config."""
    global _unconfigured_warning_emitted
    _unconfigured_warning_emitted = False


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
    """The API key, from ``Authorization: Bearer <key>``, or ``None``."""
    return bearer_token(request)


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
            leeway=LEEWAY_SECONDS,
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
        # **A deliberate change of posture, not a bug fix.** This used to
        # refuse to verify against a trust anchor nobody configured, and fail
        # closed. It now skips verification entirely and runs as the
        # development user.
        #
        # What that buys: a checkout with nothing configured serves requests,
        # which is the ordinary state of working on this locally.
        #
        # What it costs, stated plainly because it is the whole risk: a
        # deployment that *intends* to verify and has a missing or misspelled
        # issuer variable will serve every request unauthenticated, and the
        # only signal is the warning below. Fail-closed made that
        # misconfiguration loud; this makes it quiet. There is no flag to pick
        # between the two -- this replaces the old behaviour rather than
        # joining it.
        _warn_unconfigured_once()
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
    """The process-wide control plane handle, opened on first use.

    Where it connects comes from the environment and has no default, so a
    process nobody told about a database raises here rather than quietly
    opening a private one of its own. That failure is a control-plane error
    like any other, and ``resolve_org`` already treats those as a rejection
    the client cannot distinguish from a bad key.
    """
    global _control_plane
    with _control_plane_lock:
        if _control_plane is None:
            _control_plane = open_control_plane()
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

    verified = verify_key(record, hashed)
    if verified is None and record.is_revoked:
        # Worth its own line: a revoked key still in use means a credential
        # was not rotated out of a running client, or was stolen.
        logger.warning(
            "api key rejected: key_id=%s for org_id=%s was revoked at %s",
            record.key_id,
            record.org_id,
            record.revoked_at,
        )
        raise _unauthorized()

    if verified is None:
        logger.warning("api key rejected: key_id=%s failed verification", record.key_id)
        raise _unauthorized()

    return verified.org_id


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
    "Depends",
    "api_key_from",
    "bearer_token",
    "control_plane",
    "get_current_tenant_org",
    "get_current_user",
    "reset_jwks_cache",
    "reset_unconfigured_warning",
    "resolve_org",
    "set_control_plane",
    "verify_session_token",
]
