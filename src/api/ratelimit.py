"""Per-user limits on the routes that call a paid model.

A fixed-window counter held in memory: one count per key per window. The key
is the signed-in user, not the address, so everyone behind one NAT does not
share a single allowance. The session dependency puts the user on the request
before the limit is checked; the address is the fallback for a route without
one.

In memory means per process. Several workers each keep their own counter, so
the effective limit is that many times the configured rate.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def _user_key(request: Request) -> str:
    """The signed-in user, or the client address when there is none."""
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=_user_key)

#: The one rate every model-calling route shares.
LLM_RATE_LIMIT = "10/minute"
