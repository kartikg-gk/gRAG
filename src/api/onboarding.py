"""Setting up a whole tenant in one admin call.

One request creates everything a tenant needs before its first build: the
organisation, a key to reach it with, the repository it tracks, and the pod it
is placed on. Then the first build is armed.

One transaction
---------------

The four rows land together or not at all. A tenant with a key and no
organisation is a credential nobody can resolve; an organisation with no
assignment is a tenant no pod will ever load. Either is worse than a request
that failed cleanly and can simply be sent again.

That is also why the key is built here rather than through the control plane's
issuing call: that call commits on its own, and a commit in the middle would
leave the first half of a tenant behind whenever the second half failed.

The pod is chosen **before** the transaction opens. Placement reads the fleet
and never raises, so there is nothing to roll back if it is slow or degraded,
and a read inside the transaction would hold it open for no benefit.

Arming happens after the commit
-------------------------------

The tenant exists once the commit returns. A queue that cannot be reached at
that point is a build that has not been scheduled yet, not a tenant that failed
to be created, so it is reported in the response rather than failing it. The
key in that response is shown exactly once; refusing the request would lose it.

The guard
---------

A shared secret in a header, and nothing else. This route creates tenants, so
it cannot depend on a tenant key or on a user's session. With no secret
configured the route is off, rather than open.
"""

from __future__ import annotations

import hmac
import logging
import secrets
import time

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict

from ..common.config import ADMIN_SECRET_KEY, POD_ADDRESS
from ..control_plane import PREFIX_LENGTH, hash_api_key, new_api_key
from ..models import (
    LOAD_PULLING,
    ApiKey,
    Organization,
    Pod,
    PodAssignment,
    Repository,
    control_plane_sessions,
)
from ..models.control_plane import DEFAULT_SCOPES, POD_BOOTING
from ..placement import choose_pod
from ..worker.tasks import arm_organization
from . import auth as auth_module

logger = logging.getLogger("graphrag.api.onboarding")

router = APIRouter(prefix="/api/admin/onboarding")

#: Shown with every successful response, because the key beside it cannot be
#: recovered from anything that is stored.
PROVISION_WARNING = "save this api_key now: it is shown once and cannot be recovered"

DISABLED_DETAIL = (
    "onboarding is disabled: GRAPHRAG_ADMIN_SECRET_KEY is not configured"
)

#: Deliberately says nothing about what failed. The cause is in the log.
FAILED_DETAIL = "tenant provisioning failed"


class ProvisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_name: str
    #: ``owner/repo``.
    repo_name: str


class ProvisionResponse(BaseModel):
    org_id: str
    api_key: str
    pod_id: str
    repository_id: str
    reconcile_armed: bool
    warning: str


def require_admin(x_admin_secret: str | None = Header(default=None)) -> None:
    """Let the request through only with the configured admin secret.

    Unconfigured is 503, not 401: nothing the caller sends can make it work,
    and a 401 would suggest a secret exists to be guessed. The comparison is
    constant-time, so how much of a guess matched cannot be timed.
    """
    expected = ADMIN_SECRET_KEY
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=DISABLED_DETAIL
        )

    if x_admin_secret is None or not hmac.compare_digest(
        x_admin_secret.encode("utf-8"), expected.encode("utf-8")
    ):
        logger.warning("onboarding refused: missing or wrong admin secret")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=auth_module.INVALID_CREDENTIALS,
        )


@router.post(
    "/provision",
    response_model=ProvisionResponse,
    dependencies=[Depends(require_admin)],
)
def provision(payload: ProvisionRequest) -> ProvisionResponse:
    """Create a tenant's four rows in one transaction, then arm its first build."""
    engine = auth_module.control_plane().engine

    org_id = "org_" + secrets.token_hex(8)
    pod_id = choose_pod(engine=engine)

    raw_key = new_api_key()
    repository_id = secrets.token_hex(8)
    now = int(time.time())

    try:
        with control_plane_sessions(engine)() as session:
            session.add(
                Organization(
                    org_id=org_id,
                    name=payload.tenant_name,
                    plan="free",
                    status="active",
                    created_at=now,
                    updated_at=now,
                )
            )
            # Written ahead of the rows that name it. The organisation table
            # sits in a cycle of foreign keys, so a single flush does not
            # promise to insert it first. This writes inside the transaction
            # and commits nothing: a later failure still rolls it back.
            session.flush()
            session.add(
                ApiKey(
                    key_id=secrets.token_hex(8),
                    hashed_key=hash_api_key(raw_key),
                    org_id=org_id,
                    scopes=DEFAULT_SCOPES,
                    prefix=raw_key[:PREFIX_LENGTH],
                    created_at=now,
                    revoked_at=None,
                )
            )
            session.add(
                Repository(
                    repo_id=repository_id,
                    org_id=org_id,
                    provider="github",
                    provider_repo_id=payload.repo_name,
                    name=payload.repo_name,
                    default_branch="main",
                    last_synced_cursor=None,
                    status="active",
                    created_at=now,
                )
            )
            if session.get(Pod, pod_id) is None:
                # Placement falls back to this process's pod when no pod has
                # registered, and the assignment below names it by foreign
                # key. Without a row, onboarding fails until the pod agent has
                # run once. Written as booting, not ready: it exists but has
                # not said it serves; its agent marks it ready at boot.
                session.add(
                    Pod(
                        pod_id=pod_id,
                        address=POD_ADDRESS,
                        status=POD_BOOTING,
                        last_heartbeat_at=None,
                        created_at=now,
                    )
                )
                session.flush()
            session.add(
                PodAssignment(
                    pod_id=pod_id,
                    org_id=org_id,
                    load_status=LOAD_PULLING,
                    assigned_at=now,
                )
            )
            session.commit()
    except Exception as exc:  # noqa: BLE001 - any failure leaves no tenant
        # The session closes on the way out and rolls back what it held, so
        # nothing above this line survives. The cause stays in the log.
        auth_module._report(f"provisioning {org_id} failed", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=FAILED_DETAIL
        ) from exc

    reconcile_armed = True
    try:
        arm_organization(org_id)
    except Exception:  # noqa: BLE001 - the tenant exists; say so and continue
        logger.warning(
            "%s: provisioned, but its first build could not be armed",
            org_id,
            exc_info=True,
        )
        reconcile_armed = False

    return ProvisionResponse(
        org_id=org_id,
        api_key=raw_key,
        pod_id=pod_id,
        repository_id=repository_id,
        reconcile_armed=reconcile_armed,
        warning=PROVISION_WARNING,
    )
