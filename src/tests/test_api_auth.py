"""Tests for the two checks in front of the HTTP surface.

**No test here reaches a network or a real identity provider.** A keypair is
generated in-process, tokens are signed with it, and the JWKS client is
replaced by one that hands back the matching public key — so signature
verification, `kid` resolution and every claim rule run for real against
fabricated keys rather than being stubbed out.

The control plane is a real SQLite database in ``tmp_path``. It is small
enough that faking it would test the fake, and using it means the "a raw key
is never written" claim can be checked against the actual file on disk.

The two layers are tested apart and together, because the property that
matters most is that they stay apart: a verified user must not produce a
tenant, and a valid API key must not produce a user.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from src.api import auth as auth_module
from src.api import tenancy as tenancy_module
from src.api.app import create_app as _create_app


def create_app(*args, **kwargs):
    """The application, plus one probe route that reports what the two
    dependencies resolved. Test-only: it reads no graph data, which is what
    makes it a check on the credentials rather than a query path."""
    from fastapi import Depends, Request

    app = _create_app(*args, **kwargs)

    @app.get("/api/probe")
    def probe(
        request: Request,
        user_id: str = Depends(auth_module.get_current_user),
        org_id: str = Depends(auth_module.get_current_tenant_org),
    ) -> dict:
        return {
            "user_id": user_id,
            "org_id": org_id,
            "state_user_id": request.state.user_id,
            "state_org_id": request.state.org_id,
        }

    return app
from src.control_plane import (
    ControlPlaneError,
    hash_api_key,
    open_control_plane,
    verify_key,
)

def provisioned(path, org_ids=("org_alpha", "org_beta")):
    """A control plane with those tenants already provisioned.

    A credential refers to the organisation it names, so the row has to exist
    before a key can be issued against it. Tests below are about credentials
    rather than about provisioning, so they get their tenants from here.
    """
    from src.models import Organization, control_plane_sessions

    plane = open_control_plane(path)
    with control_plane_sessions(plane.engine)() as session:
        for org_id in org_ids:
            session.add(
                Organization(
                    org_id=org_id,
                    name=org_id,
                    plan="team",
                    status="active",
                    created_at=0,
                    updated_at=0,
                )
            )
        session.commit()
    return plane


ISSUER = "https://identity.test.invalid"
JWKS_URL = "https://identity.test.invalid/.well-known/jwks.json"
KID = "test-key-1"
SUBJECT = "user_abc123"


# --------------------------------------------------------------------------
# keys and tokens
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def keypair():
    """One RSA key for the module. Generating per test costs seconds, not ms."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key()
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = public.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem.decode(), public_pem.decode()


@pytest.fixture(scope="module")
def other_key():
    """A second key, for signing something the first key must reject."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


class FakeSigningKey:
    def __init__(self, key):
        self.key = key


class FakeJWKSClient:
    """Resolves by ``kid``, the way a real one does, and records the lookups."""

    def __init__(self, keys: dict[str, str], error: Exception | None = None):
        self.keys = keys
        self.error = error
        self.lookups: list[str] = []

    def get_signing_key_from_jwt(self, token: str):
        if self.error is not None:
            raise self.error
        kid = jwt.get_unverified_header(token).get("kid")
        self.lookups.append(kid)
        if kid not in self.keys:
            raise jwt.PyJWTError(f"no key for kid {kid!r}")
        return FakeSigningKey(self.keys[kid])


def token_for(private_pem: str, **overrides) -> str:
    """A well-formed token, with any claim replaced or removed.

    Passing ``exp=None`` removes the claim rather than setting it to null,
    which is how the "missing exp" cases are written without a second builder.
    """
    now = datetime.now(timezone.utc)
    claims = {
        "sub": SUBJECT,
        "iss": ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
    }
    claims.update(overrides)
    claims = {name: value for name, value in claims.items() if value is not None}
    return jwt.encode(claims, private_pem, algorithm="RS256", headers={"kid": KID})


@pytest.fixture
def clerk(monkeypatch, keypair):
    """Session verification on, pointed at the in-process keypair."""
    private_pem, public_pem = keypair
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", True)
    monkeypatch.setattr(auth_module, "CLERK_ISSUER", ISSUER)
    monkeypatch.setattr(auth_module, "CLERK_JWKS_URL", JWKS_URL)
    monkeypatch.setattr(auth_module, "CLERK_AUTHORIZED_PARTIES", ())
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)
    client = FakeJWKSClient({KID: public_pem})
    monkeypatch.setattr(auth_module, "_jwks_client", lambda url: client)
    return client


@pytest.fixture
def app_client(clerk):
    return TestClient(create_app(), raise_server_exceptions=False)


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ==========================================================================
# session tokens
# ==========================================================================


def test_a_valid_token_authenticates_as_its_subject(app_client, keypair):
    private_pem, _ = keypair
    response = app_client.get("/api/probe", headers=bearer(token_for(private_pem)))

    assert response.status_code == 200
    assert response.json()["user_id"] == SUBJECT


def test_a_missing_authorization_header_is_rejected(app_client):
    assert app_client.get("/api/probe").status_code == 401


@pytest.mark.parametrize(
    "header",
    [
        "",
        "Bearer",
        "Bearer ",
        "Basic abcdef",
        "Token abcdef",
        "abcdef",
        "Bearer  double-space",
    ],
)
def test_a_malformed_authorization_header_is_rejected(app_client, header, keypair):
    """A header that is not exactly one bearer token is not a credential."""
    response = app_client.get("/api/probe", headers={"Authorization": header})
    assert response.status_code == 401


@pytest.mark.parametrize("token", ["not-a-jwt", "a.b", "a.b.c", "..", "eyJhbGciOiJub25lIn0."])
def test_a_malformed_token_is_rejected(app_client, token):
    assert app_client.get("/api/probe", headers=bearer(token)).status_code == 401


def test_an_expired_token_is_rejected(app_client, keypair):
    private_pem, _ = keypair
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    token = token_for(private_pem, exp=int(past.timestamp()))

    assert app_client.get("/api/probe", headers=bearer(token)).status_code == 401


def test_a_token_signed_by_another_key_is_rejected(app_client, other_key):
    """The signature is checked against the key the kid resolved to."""
    assert app_client.get("/api/probe", headers=bearer(token_for(other_key))).status_code == 401


def test_a_token_from_another_issuer_is_rejected(app_client, keypair):
    private_pem, _ = keypair
    token = token_for(private_pem, iss="https://attacker.test.invalid")

    assert app_client.get("/api/probe", headers=bearer(token)).status_code == 401


@pytest.mark.parametrize("claim", ["sub", "exp", "iat"])
def test_a_token_missing_a_required_claim_is_rejected(app_client, keypair, claim):
    private_pem, _ = keypair
    assert app_client.get(
        "/api/probe", headers=bearer(token_for(private_pem, **{claim: None}))
    ).status_code == 401


def test_an_empty_subject_is_rejected(app_client, keypair):
    """Present but unusable. Would otherwise become a falsy user id."""
    private_pem, _ = keypair
    assert app_client.get(
        "/api/probe", headers=bearer(token_for(private_pem, sub=""))
    ).status_code == 401


def test_an_unsigned_token_is_rejected(app_client, keypair):
    """`alg: none` is the attack the fixed algorithm list exists to stop."""
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": SUBJECT,
            "iss": ISSUER,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
        },
        key=None,
        algorithm="none",
        headers={"kid": KID},
    )
    assert app_client.get("/api/probe", headers=bearer(token)).status_code == 401


def test_a_token_signed_with_hmac_over_the_public_key_is_rejected(app_client, keypair):
    """Algorithm confusion: the public key used as an HMAC secret.

    This is the attack that succeeds when the accepted algorithm is read from
    the token instead of fixed by the verifier.
    """
    _private_pem, public_pem = keypair
    now = datetime.now(timezone.utc)

    # Built by hand: PyJWT refuses to *sign* with a PEM key as an HMAC secret,
    # which is a guard on the issuing side. An attacker has no such library.
    def segment(payload: dict) -> bytes:
        return base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode()
        ).rstrip(b"=")

    signing_input = b".".join(
        [
            segment({"alg": "HS256", "typ": "JWT", "kid": KID}),
            segment(
                {
                    "sub": SUBJECT,
                    "iss": ISSUER,
                    "iat": int(now.timestamp()),
                    "exp": int((now + timedelta(minutes=5)).timestamp()),
                }
            ),
        ]
    )
    signature = base64.urlsafe_b64encode(
        hmac.new(public_pem.encode(), signing_input, hashlib.sha256).digest()
    ).rstrip(b"=")
    token = (signing_input + b"." + signature).decode()

    assert app_client.get("/api/probe", headers=bearer(token)).status_code == 401


def test_the_kid_selects_the_key_so_rotation_needs_no_restart(app_client, keypair, clerk):
    """A second key added to the set is used as soon as a token names it."""
    private_pem, public_pem = keypair
    clerk.keys["rotated-key"] = public_pem
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": SUBJECT,
            "iss": ISSUER,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
        },
        private_pem,
        algorithm="RS256",
        headers={"kid": "rotated-key"},
    )

    assert app_client.get("/api/probe", headers=bearer(token)).status_code == 200
    assert "rotated-key" in clerk.lookups


def test_an_unknown_kid_is_rejected(app_client, keypair):
    private_pem, _ = keypair
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": SUBJECT,
            "iss": ISSUER,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
        },
        private_pem,
        algorithm="RS256",
        headers={"kid": "a-key-that-does-not-exist"},
    )
    assert app_client.get("/api/probe", headers=bearer(token)).status_code == 401


def test_a_jwks_failure_rejects_rather_than_erroring(monkeypatch, keypair, clerk, caplog):
    """The endpoint being down is a server fault, logged as one, and still 401."""
    monkeypatch.setattr(
        auth_module,
        "_jwks_client",
        lambda url: FakeJWKSClient({}, error=ConnectionError("jwks unreachable")),
    )
    private_pem, _ = keypair
    client = TestClient(create_app(), raise_server_exceptions=False)

    with caplog.at_level("ERROR"):
        response = client.get("/api/probe", headers=bearer(token_for(private_pem)))

    assert response.status_code == 401
    assert any("JWKS" in record.message for record in caplog.records)


def test_an_unconfigured_issuer_runs_as_the_development_user(monkeypatch, clerk):
    """**A deliberate change of posture.**

    This used to refuse to verify against a trust anchor nobody configured and
    fail closed. It now skips verification and runs as the development user,
    so a checkout with nothing set serves requests. The cost is that a
    deployment meaning to verify, with a missing issuer variable, serves every
    request unauthenticated -- which is what the warning exists to surface.
    """
    monkeypatch.setattr(auth_module, "CLERK_ISSUER", "")
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    auth_module.reset_unconfigured_warning()
    client = TestClient(create_app(engine_factory=lambda: None), raise_server_exceptions=False)

    response = client.get("/api/probe")

    assert response.status_code == 200
    assert response.json()["user_id"] == auth_module.DEV_USER_ID


def test_the_unconfigured_warning_fires_once_not_once_per_request(
    monkeypatch, clerk, caplog
):
    """A line per request is a line nobody reads."""
    monkeypatch.setattr(auth_module, "CLERK_ISSUER", "")
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    auth_module.reset_unconfigured_warning()
    client = TestClient(create_app(engine_factory=lambda: None), raise_server_exceptions=False)

    with caplog.at_level("WARNING", logger="graphrag.api.auth"):
        for _ in range(5):
            assert client.get("/api/probe").status_code == 200

    fired = [
        record
        for record in caplog.records
        if "SESSION VERIFICATION IS OFF" in record.getMessage()
    ]
    assert len(fired) == 1, f"warning fired {len(fired)} times"


def test_the_unconfigured_warning_names_what_to_set(monkeypatch, clerk, caplog):
    monkeypatch.setattr(auth_module, "CLERK_ISSUER", "")
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    auth_module.reset_unconfigured_warning()
    client = TestClient(create_app(engine_factory=lambda: None), raise_server_exceptions=False)

    with caplog.at_level("WARNING", logger="graphrag.api.auth"):
        client.get("/api/probe")

    message = " ".join(record.getMessage() for record in caplog.records)
    assert "GRAPHRAG_CLERK_ISSUER" in message
    assert auth_module.DEV_USER_ID in message


def test_a_token_is_not_even_looked_at_when_no_issuer_is_configured(
    monkeypatch, clerk, other_key
):
    """Skipped entirely, not verified leniently: a bad token is not consulted."""
    monkeypatch.setattr(auth_module, "CLERK_ISSUER", "")
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    auth_module.reset_unconfigured_warning()
    client = TestClient(create_app(engine_factory=lambda: None), raise_server_exceptions=False)

    response = client.get("/api/probe", headers=bearer(token_for(other_key)))

    assert response.status_code == 200
    assert response.json()["user_id"] == auth_module.DEV_USER_ID


def test_the_leeway_is_fixed_in_code_and_not_read_from_the_environment(monkeypatch):
    """Leeway is how long an expired token keeps working. Not a deployment knob."""
    import src.common.config as config

    monkeypatch.setenv("GRAPHRAG_CLERK_LEEWAY_SECONDS", "86400")

    assert auth_module.LEEWAY_SECONDS == 5
    assert not hasattr(config, "CLERK_LEEWAY_SECONDS")


def test_a_permitted_authorized_party_is_accepted(monkeypatch, keypair, clerk):
    monkeypatch.setattr(auth_module, "CLERK_AUTHORIZED_PARTIES", ("https://app.test",))
    private_pem, _ = keypair
    client = TestClient(create_app(), raise_server_exceptions=False)
    token = token_for(private_pem, azp="https://app.test")

    assert client.get("/api/probe", headers=bearer(token)).status_code == 200


@pytest.mark.parametrize("azp", ["https://evil.test", "", None])
def test_a_disallowed_or_absent_authorized_party_is_rejected(
    monkeypatch, keypair, clerk, azp
):
    monkeypatch.setattr(auth_module, "CLERK_AUTHORIZED_PARTIES", ("https://app.test",))
    private_pem, _ = keypair
    client = TestClient(create_app(), raise_server_exceptions=False)
    token = token_for(private_pem, azp=azp)

    assert client.get("/api/probe", headers=bearer(token)).status_code == 401


def test_the_authorized_party_check_is_off_when_the_list_is_empty(app_client, keypair):
    """An empty allow-list means 'not checked', not 'permit nothing'."""
    private_pem, _ = keypair
    token = token_for(private_pem, azp="https://anything.test")

    assert app_client.get("/api/probe", headers=bearer(token)).status_code == 200


def test_a_rejection_says_nothing_about_why(app_client, keypair, other_key):
    """Expired, wrong issuer and wrong signature are one message to the client."""
    private_pem, _ = keypair
    past = int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp())
    bodies = {
        app_client.get("/api/probe", headers=bearer(token_for(private_pem, exp=past))).text,
        app_client.get(
            "/api/probe", headers=bearer(token_for(private_pem, iss="https://x.test"))
        ).text,
        app_client.get("/api/probe", headers=bearer(token_for(other_key))).text,
    }

    assert len(bodies) == 1


def test_a_client_supplied_user_id_is_ignored(app_client, keypair):
    """The id comes from the token's sub. Nothing else can name a user."""
    private_pem, _ = keypair
    response = app_client.get(
        "/api/probe?user_id=someone_else",
        headers={**bearer(token_for(private_pem)), "X-User-Id": "someone_else"},
    )

    assert response.status_code == 200
    assert response.json()["user_id"] == SUBJECT


def test_the_verified_user_is_on_request_state(app_client, keypair):
    private_pem, _ = keypair
    body = app_client.get("/api/probe", headers=bearer(token_for(private_pem))).json()

    assert body["state_user_id"] == SUBJECT


# ==========================================================================
# the development bypass
# ==========================================================================


def test_the_bypass_runs_as_the_development_user_and_warns(monkeypatch, caplog):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)
    auth_module.reset_unconfigured_warning()
    client = TestClient(create_app(), raise_server_exceptions=False)

    with caplog.at_level("WARNING"):
        response = client.get("/api/probe")

    assert response.status_code == 200
    assert response.json()["user_id"] == auth_module.DEV_USER_ID
    assert any("SESSION VERIFICATION IS OFF" in record.getMessage() for record in caplog.records)


def test_the_development_user_id_is_the_pinned_default():
    assert auth_module.DEV_USER_ID == "dev-user"


def test_verification_and_tenancy_default_off():
    """Reading configuration, not the patched module.

    Session verification is on exactly when an issuer is configured, and none
    is by default. Tenancy is off unless a deployment switches it on: a single
    process then needs no API key and no control plane.
    """
    from src.common import config

    assert config.CLERK_ENABLED is False
    assert config.MULTI_TENANCY_ENABLED is False


# ==========================================================================
# API keys and the control plane
# ==========================================================================


@pytest.fixture
def plane(tmp_path):
    store = provisioned(tmp_path / "control.db")
    yield store
    store.close()


@pytest.fixture
def tenant_client(monkeypatch, plane):
    """Tenancy on, session verification off, so one header carries the key."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    auth_module.set_control_plane(plane)
    yield TestClient(create_app(), raise_server_exceptions=False)
    auth_module.set_control_plane(None)


def test_a_valid_key_resolves_to_its_organisation(tenant_client, plane):
    raw, record = plane.issue("org_alpha")
    response = tenant_client.get("/api/probe", headers={"Authorization": f"Bearer {raw}"})

    assert response.status_code == 200
    assert response.json()["org_id"] == "org_alpha"
    assert record.org_id == "org_alpha"


def test_an_unknown_key_is_rejected(tenant_client, plane):
    plane.issue("org_alpha")
    assert tenant_client.get(
        "/api/probe", headers={"Authorization": f"Bearer {"not-a-real-key"}"}
    ).status_code == 401


def test_a_revoked_key_is_rejected_and_logged(tenant_client, plane, caplog):
    raw, record = plane.issue("org_alpha")
    assert plane.revoke(record.key_id) is True

    with caplog.at_level("WARNING"):
        response = tenant_client.get("/api/probe", headers={"Authorization": f"Bearer {raw}"})

    assert response.status_code == 401
    assert any("revoked" in entry.getMessage() for entry in caplog.records)


def test_a_missing_key_is_rejected_when_tenancy_is_on(tenant_client):
    assert tenant_client.get("/api/probe").status_code == 401


def test_a_control_plane_failure_rejects_and_is_logged(monkeypatch, tenant_client, caplog):
    """A store that cannot answer is a server fault, not a verdict on the key."""

    class Broken:
        def record_for_hash(self, hashed_key):
            raise ControlPlaneError("disk is on fire")

    auth_module.set_control_plane(Broken())

    with caplog.at_level("ERROR"):
        response = tenant_client.get("/api/probe", headers={"Authorization": f"Bearer {"anything"}"})

    assert response.status_code == 401
    assert any("control plane" in entry.getMessage() for entry in caplog.records)


def test_every_key_failure_looks_the_same_to_the_client(tenant_client, plane):
    raw, record = plane.issue("org_alpha")
    plane.revoke(record.key_id)
    live, _ = plane.issue("org_beta")

    bodies = {
        tenant_client.get("/api/probe", headers={"Authorization": f"Bearer {raw}"}).text,
        tenant_client.get("/api/probe", headers={"Authorization": f"Bearer {"unknown"}"}).text,
        tenant_client.get("/api/probe").text,
        tenant_client.get("/api/probe", headers={"Authorization": f"Bearer {live[:-1]}"}).text,
    }

    assert len(bodies) == 1


def test_the_raw_key_is_never_written_to_the_database(plane, tmp_path):
    """Checked against the bytes on disk, not against the API."""
    raw, _record = plane.issue("org_alpha")
    plane.close()

    blob = (tmp_path / "control.db").read_bytes()

    assert raw.encode() not in blob
    assert hash_api_key(raw).encode() in blob


def test_the_stored_form_is_sha256_of_the_key():
    import hashlib

    assert hash_api_key("abc") == hashlib.sha256(b"abc").hexdigest()


def test_verification_rejects_a_mismatched_digest(plane):
    _raw, record = plane.issue("org_alpha")

    assert verify_key(record, record.hashed_key) is record
    assert verify_key(record, hash_api_key("something else")) is None
    assert verify_key(None, record.hashed_key) is None


def test_verification_rejects_a_revoked_record(plane):
    raw, record = plane.issue("org_alpha")
    plane.revoke(record.key_id)

    stored = plane.record_for_hash(hash_api_key(raw))

    assert stored.is_revoked is True
    assert verify_key(stored, hash_api_key(raw)) is None


def test_revoking_twice_reports_that_nothing_changed(plane):
    _raw, record = plane.issue("org_alpha")

    assert plane.revoke(record.key_id) is True
    assert plane.revoke(record.key_id) is False


def test_two_keys_for_one_org_are_distinct_rows(plane):
    first, first_record = plane.issue("org_alpha")
    second, second_record = plane.issue("org_alpha")

    assert first != second
    assert first_record.key_id != second_record.key_id
    assert first_record.hashed_key != second_record.hashed_key


# ==========================================================================
# tenancy
# ==========================================================================


def test_single_tenant_mode_uses_the_default_and_needs_no_key(monkeypatch):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", False)
    client = TestClient(create_app(), raise_server_exceptions=False)

    response = client.get("/api/probe")

    assert response.status_code == 200
    assert response.json()["org_id"] == auth_module.DEFAULT_TENANT_ORG_ID


def test_different_keys_resolve_to_different_organisations(tenant_client, plane):
    alpha, _ = plane.issue("org_alpha")
    beta, _ = plane.issue("org_beta")

    first = tenant_client.get("/api/probe", headers={"Authorization": f"Bearer {alpha}"}).json()
    second = tenant_client.get("/api/probe", headers={"Authorization": f"Bearer {beta}"}).json()

    assert first["org_id"] == "org_alpha"
    assert second["org_id"] == "org_beta"


def test_the_resolved_tenant_is_on_request_state(tenant_client, plane):
    raw, _ = plane.issue("org_alpha")
    body = tenant_client.get("/api/probe", headers={"Authorization": f"Bearer {raw}"}).json()

    assert body["state_org_id"] == "org_alpha"


def test_current_org_reaches_code_that_never_sees_the_request(monkeypatch, plane):
    """The reason the ContextVar exists: a sync route, several frames deep."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    auth_module.set_control_plane(plane)

    def deep_call() -> str:
        """Takes no request, and still knows the tenant."""
        return tenancy_module.current_org()

    app = FastAPI()

    @app.get("/deep")
    def deep(org_id: str = Depends(auth_module.get_current_tenant_org)) -> dict:
        return {"from_dependency": org_id, "from_context": deep_call()}

    raw, _ = plane.issue("org_alpha")
    body = TestClient(app).get("/deep", headers={"Authorization": f"Bearer {raw}"}).json()

    assert body == {"from_dependency": "org_alpha", "from_context": "org_alpha"}
    auth_module.set_control_plane(None)


def test_current_org_reaches_an_async_route_too(monkeypatch, plane):
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    auth_module.set_control_plane(plane)

    app = FastAPI()

    @app.get("/deep")
    async def deep(org_id: str = Depends(auth_module.get_current_tenant_org)) -> dict:
        return {"from_context": tenancy_module.current_org()}

    raw, _ = plane.issue("org_beta")
    body = TestClient(app).get("/deep", headers={"Authorization": f"Bearer {raw}"}).json()

    assert body == {"from_context": "org_beta"}
    auth_module.set_control_plane(None)


def test_the_context_is_unbound_once_the_request_ends(monkeypatch, plane):
    """A reused task must not start with the previous request's tenant."""
    monkeypatch.setattr(auth_module, "CLERK_ENABLED", False)
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    auth_module.set_control_plane(plane)

    raw, _ = plane.issue("org_alpha")
    client = TestClient(create_app(), raise_server_exceptions=False)
    assert client.get("/api/probe", headers={"Authorization": f"Bearer {raw}"}).status_code == 200

    assert tenancy_module.current_org_or_none() is None
    auth_module.set_control_plane(None)


def test_reading_the_tenant_outside_a_request_raises():
    """Returning None would read as 'no filter', which is every tenant."""
    with pytest.raises(tenancy_module.NoCurrentOrg):
        tenancy_module.current_org()


def test_setting_and_resetting_restores_the_previous_value():
    token = tenancy_module.set_current_org("org_alpha")
    assert tenancy_module.current_org() == "org_alpha"
    tenancy_module.reset_current_org(token)
    assert tenancy_module.current_org_or_none() is None


# ==========================================================================
# the boundary between the two layers
# ==========================================================================


def test_a_verified_user_alone_does_not_produce_a_tenant(monkeypatch, keypair, clerk, plane):
    """Both checks are required. Passing one does not satisfy the other."""
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    auth_module.set_control_plane(plane)
    private_pem, _ = keypair
    client = TestClient(create_app(), raise_server_exceptions=False)

    response = client.get("/api/probe", headers=bearer(token_for(private_pem)))

    assert response.status_code == 401
    auth_module.set_control_plane(None)


def test_a_valid_key_alone_does_not_produce_a_user(monkeypatch, clerk, plane):
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    auth_module.set_control_plane(plane)
    raw, _ = plane.issue("org_alpha")
    client = TestClient(create_app(), raise_server_exceptions=False)

    response = client.get("/api/probe", headers={"Authorization": f"Bearer {raw}"})

    assert response.status_code == 401
    auth_module.set_control_plane(None)


def test_one_header_carries_one_credential(monkeypatch, keypair, clerk, plane):
    """Both credentials travel as Authorization: Bearer, so a request carries
    one of them. With sessions on, that one is the session token: sent there,
    it is not also read as a key, and the tenant check refuses the request."""
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    auth_module.set_control_plane(plane)
    private_pem, _ = keypair
    client = TestClient(create_app(), raise_server_exceptions=False)

    response = client.get("/api/probe", headers=bearer(token_for(private_pem)))

    assert response.status_code == 401
    auth_module.set_control_plane(None)


def test_the_api_key_is_not_read_from_authorization_when_sessions_are_on(
    monkeypatch, clerk, plane
):
    """With sessions on, the bearer is read as a session token first, and a
    key is not one."""
    monkeypatch.setattr(auth_module, "MULTI_TENANCY_ENABLED", True)
    auth_module.set_control_plane(plane)
    raw, _ = plane.issue("org_alpha")
    client = TestClient(create_app(), raise_server_exceptions=False)

    assert client.get("/api/probe", headers=bearer(raw)).status_code == 401
    auth_module.set_control_plane(None)


def test_the_health_route_needs_no_credential(clerk):
    class Store:
        @staticmethod
        def count_nodes():
            return 0

    class Engine:
        path = "fake.lbug"
        embedder = extractor = judge = now = None
        store = Store()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def close(self):
            pass

    with TestClient(create_app(engine_factory=Engine)) as client:
        assert client.get("/api/health").status_code == 200


# ==========================================================================
# the fields a stored credential carries
# ==========================================================================


def test_issuing_a_key_populates_the_recorded_fields(tmp_path):
    from src.control_plane import PREFIX_LENGTH
    from src.models import DEFAULT_SCOPES

    with provisioned(tmp_path / "control.db") as plane:
        raw, record = plane.issue("org_alpha")

    assert record.scopes == DEFAULT_SCOPES
    assert record.prefix == raw[:PREFIX_LENGTH]
    assert record.created_at is not None
    assert record.org_id == "org_alpha"


def test_scopes_default_to_read_and_can_be_set(tmp_path):
    from src.models import DEFAULT_SCOPES

    with provisioned(tmp_path / "control.db") as plane:
        _raw, default = plane.issue("org_alpha")
        _raw2, explicit = plane.issue("org_beta", scopes="read:write")

    assert DEFAULT_SCOPES == "read"
    assert default.scopes == "read"
    assert explicit.scopes == "read:write"


def test_the_prefix_is_far_too_short_to_reconstruct_the_key(tmp_path):
    """The whole point of a prefix is that storing and showing it is safe."""
    from src.control_plane import PREFIX_LENGTH

    with provisioned(tmp_path / "control.db") as plane:
        raw, record = plane.issue("org_alpha")

    assert len(record.prefix) == PREFIX_LENGTH == 8
    assert raw.startswith(record.prefix)
    # Most of the key is withheld: 32 random bytes render to about 43
    # characters, and eight of them leave thirty-five unknown.
    assert len(raw) - len(record.prefix) >= 30
    assert record.prefix != raw
    assert raw not in record.prefix


def test_the_prefix_and_scopes_survive_a_round_trip(tmp_path):
    with provisioned(tmp_path / "control.db") as plane:
        raw, issued = plane.issue("org_alpha", scopes="read:write")
        stored = plane.record_for_hash(hash_api_key(raw))

    assert stored.prefix == issued.prefix
    assert stored.scopes == "read:write"
    assert stored.created_at is not None


def test_the_raw_key_is_still_never_written(tmp_path):
    """The new column must not have become a place the secret leaks to."""
    path = tmp_path / "control.db"
    plane = provisioned(path)
    raw, _record = plane.issue("org_alpha")
    plane.close()

    blob = path.read_bytes()

    assert raw.encode() not in blob
    assert hash_api_key(raw).encode() in blob


def test_the_hash_column_is_unique_and_indexed(tmp_path):
    """Already true before the tables around it; confirmed rather than changed.

    Read from the database that was actually created, not from the model, so
    a declaration that never reaches the schema fails here.
    """
    from sqlalchemy import inspect

    plane = provisioned(tmp_path / "control.db")
    try:
        inspector = inspect(plane.engine)
        unique = inspector.get_unique_constraints("apikey")
        indexes = inspector.get_indexes("apikey")
        hashed = next(
            column
            for column in inspector.get_columns("apikey")
            if column["name"] == "hashed_key"
        )
    finally:
        plane.close()

    covered = [constraint["column_names"] for constraint in unique] + [
        index["column_names"] for index in indexes if index["unique"]
    ]

    assert ["hashed_key"] in covered
    assert hashed["nullable"] is False


def test_a_credential_refers_to_the_organisation_it_names(tmp_path):
    """The column is text, and it refers to a tenant that has to exist.

    Text because an organisation identifier is a string a provider chose, and
    a reference because the alternative was a credential that authenticated
    successfully as an organisation with no row anywhere.
    """
    from sqlalchemy import inspect

    from src.control_plane import ControlPlaneError

    plane = provisioned(tmp_path / "control.db")
    try:
        inspector = inspect(plane.engine)
        references = inspector.get_foreign_keys("apikey")
        stored_type = str(
            next(
                column
                for column in inspector.get_columns("apikey")
                if column["name"] == "org_id"
            )["type"]
        )

        with pytest.raises(ControlPlaneError):
            plane.issue("org_that_was_never_provisioned")
    finally:
        plane.close()

    assert [
        (reference["constrained_columns"], reference["referred_table"])
        for reference in references
    ] == [(["org_id"], "organization")]
    # The stored type as the database sees it, not the model's declaration:
    # a text column, refusing to be anything narrower.
    assert stored_type.startswith("VARCHAR") or stored_type == "TEXT"
