"""The HTTP surface, and the two checks in front of it.

    from src.api import get_current_user, get_current_tenant_org

    @app.get("/projects")
    def projects(user_id: str = Depends(get_current_user),
                 org_id: str = Depends(get_current_tenant_org)):
        ...

Two separate questions, answered from two separate credentials:

    session JWT  -> get_current_user       -> a trusted user identity
    API key      -> get_current_tenant_org -> a trusted tenant identity

Neither is derived from the other. See ``auth`` for why that boundary is the
one that matters, and ``tenancy`` for reading the resolved organisation from
code that never sees a request.

**Nothing here is imported by the rest of the project.** The web framework and
the token library are an optional dependency group, so importing ``src.api``
is the thing that requires them — which is what keeps the two runtime
dependencies at two.
"""

from .auth import (
    API_KEY_HEADER,
    get_current_tenant_org,
    get_current_user,
    reset_jwks_cache,
    resolve_org,
    set_control_plane,
    verify_session_token,
)
from .control_plane import (
    ApiKeyRecord,
    ControlPlane,
    ControlPlaneError,
    hash_api_key,
    new_api_key,
    open_control_plane,
    verify_key,
)
from .tenancy import (
    NoCurrentOrg,
    current_org,
    current_org_or_none,
    reset_current_org,
    set_current_org,
)

__all__ = [
    "API_KEY_HEADER",
    "ApiKeyRecord",
    "ControlPlane",
    "ControlPlaneError",
    "NoCurrentOrg",
    "current_org",
    "current_org_or_none",
    "get_current_tenant_org",
    "get_current_user",
    "hash_api_key",
    "new_api_key",
    "open_control_plane",
    "reset_current_org",
    "reset_jwks_cache",
    "resolve_org",
    "set_control_plane",
    "set_current_org",
    "verify_key",
    "verify_session_token",
]
