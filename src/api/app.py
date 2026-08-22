"""The HTTP surface.

Deliberately thin. Everything a route needs is already a callable elsewhere —
retrieval is ``src.retrieval.retrieve``, the store is ``ContextGraph`` — so
this layer's whole job is to establish *who* and *which tenant* before any of
it runs, and to keep those two answers out of the request body.

The dependency pair is the interface:

    def route(user_id: str = Depends(get_current_user),
              org_id: str = Depends(get_current_tenant_org)):

Both are resolved before the route body executes, and both raise 401 rather
than returning a sentinel, so a body that runs has a verified user and a
verified tenant with no branch of its own.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Request

from .auth import get_current_tenant_org, get_current_user


def create_app() -> FastAPI:
    """Build the application.

    A factory rather than a module-level instance so tests can build one per
    configuration without reimporting the module, which matters here because
    the dependencies read configuration that tests need to vary.
    """
    app = FastAPI(title="graphrag")

    @app.get("/health")
    def health() -> dict:
        """Unauthenticated on purpose: a readiness probe that needs a
        credential cannot report that credentials are misconfigured."""
        return {"status": "ok"}

    @app.get("/projects")
    def projects(
        request: Request,
        user_id: str = Depends(get_current_user),
        org_id: str = Depends(get_current_tenant_org),
    ) -> dict:
        """The example the two dependencies exist for.

        ``user_id`` is the verified ``sub`` of a session token. ``org_id`` is
        the organisation an API key resolved to. Neither is read from the
        request body, and there is no parameter through which a client could
        offer its own.
        """
        return {
            "user_id": user_id,
            "org_id": org_id,
            # Both are also on request.state, for middleware and for anything
            # that has the request but not the dependency's return value.
            "state_user_id": request.state.user_id,
            "state_org_id": request.state.org_id,
        }

    return app
