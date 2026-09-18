"""Only answering a tenant's request from the pod assigned to it.

With more than one serving process, a tenant request can land on a process
that is not the one holding that tenant's graph. Before this, such a process
either had the graph by accident or returned 503, and nothing told the caller
whether to wait, go elsewhere, or give up. This middleware says which.

The lookup
----------

``assignment_status`` reads every assignment for a tenant and classifies it
relative to this pod. It is a plain function over the control plane, so each
verdict is testable without an application.

A tenant assigned to another pod counts as elsewhere even if that pod is still
loading: the caller still needs to be pointed there, and a pod that is ready is
preferred over one that is not when there is a choice.

The gate
--------

It runs only for routes that read a tenant's graph, only with tenancy on, and
never for preflight requests. The route dependencies still run after it lets a
request through: the tenant is resolved again there, and ``engine_for`` still
answers 503 when the control plane says this pod is ready but the graph has not
reached this process's registry.

Every refusal other than a bad key is also reported, because each one means the
fleet and the traffic reaching it disagree.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import HTTPException
from sqlmodel import select
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..common.config import POD_ID
from ..models import LOAD_READY, PodAssignment, control_plane_sessions
from . import auth as auth_module

logger = logging.getLogger("graphrag.api.routing")

#: The routes that read a tenant's graph. Everything else passes untouched.
TENANT_SCOPED_PREFIXES = ("/api/trace", "/api/subgraph", "/api/suggestions")

READY_HERE = "ready_here"
PULLING_HERE = "pulling_here"
READY_ELSEWHERE = "ready_elsewhere"
UNASSIGNED = "unassigned"

PULLING_DETAIL = "this tenant's graph is still loading on this pod; retry shortly"
ELSEWHERE_DETAIL = "this tenant is served by a different pod"
UNASSIGNED_DETAIL = "no pod is serving this tenant yet"
UNAVAILABLE_DETAIL = "routing unavailable"

RETRY_AFTER_SECONDS = "5"
ASSIGNED_POD_HEADER = "X-Graphrag-Assigned-Pod"


def assignment_status(org_id: str, *, pod_id=None, engine=None) -> tuple[str, dict]:
    """How ``org_id`` is placed relative to ``pod_id``, and the detail to report.

    ``pod_id`` defaults to this process's pod and ``engine`` to the process's
    control plane.
    """
    pod_id = pod_id if pod_id is not None else POD_ID
    engine = engine if engine is not None else auth_module.control_plane().engine

    with control_plane_sessions(engine)() as session:
        rows = session.exec(
            select(PodAssignment).where(PodAssignment.org_id == org_id)
        ).all()

    mine = next((row for row in rows if row.pod_id == pod_id), None)
    if mine is not None:
        if mine.load_status == LOAD_READY:
            return READY_HERE, {"pod_id": pod_id}
        return PULLING_HERE, {"pod_id": pod_id, "load_status": mine.load_status}

    target = next((row for row in rows if row.load_status == LOAD_READY), None)
    if target is None and rows:
        target = rows[0]
    if target is not None:
        return READY_ELSEWHERE, {
            "pod_id": target.pod_id,
            "load_status": target.load_status,
        }

    return UNASSIGNED, {}


def _unauthorized() -> JSONResponse:
    """The project's one uniform credential refusal, as a response.

    A middleware cannot raise ``HTTPException``, so this is what the dependency
    would have produced, built directly.
    """
    return JSONResponse(
        {"detail": auth_module.INVALID_CREDENTIALS},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


class TenantRoutingMiddleware(BaseHTTPMiddleware):
    """Let a tenant request through only on the pod serving that tenant."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if not auth_module.MULTI_TENANCY_ENABLED:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        if not request.url.path.startswith(TENANT_SCOPED_PREFIXES):
            return await call_next(request)

        api_key = auth_module.api_key_from(request)
        if api_key is None:
            return _unauthorized()
        try:
            # A database read, off the event loop.
            org_id = await asyncio.to_thread(auth_module.resolve_org, api_key)
        except HTTPException:
            return _unauthorized()

        try:
            verdict, detail = await asyncio.to_thread(assignment_status, org_id)
        except Exception as exc:  # noqa: BLE001 - any lookup failure is unavailability
            auth_module._report(f"routing lookup failed for {org_id}", exc)
            return JSONResponse({"detail": UNAVAILABLE_DETAIL}, status_code=503)

        if verdict == READY_HERE:
            request.state.org_id = org_id
            return await call_next(request)

        if verdict == PULLING_HERE:
            auth_module._report(f"{org_id}: request arrived while its graph is loading here")
            return JSONResponse(
                {"detail": PULLING_DETAIL, "org_id": org_id},
                status_code=503,
                headers={"Retry-After": RETRY_AFTER_SECONDS},
            )

        if verdict == READY_ELSEWHERE:
            assigned = detail["pod_id"]
            auth_module._report(f"{org_id}: request reached {POD_ID}, served by {assigned}")
            return JSONResponse(
                {"detail": ELSEWHERE_DETAIL, "org_id": org_id, "assigned_pod": assigned},
                status_code=421,
                headers={ASSIGNED_POD_HEADER: str(assigned)},
            )

        auth_module._report(f"{org_id}: request arrived with no pod assigned")
        return JSONResponse(
            {"detail": UNASSIGNED_DETAIL, "org_id": org_id},
            status_code=409,
        )
