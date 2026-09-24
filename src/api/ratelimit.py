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

import os

from fastapi import HTTPException
from limits import parse
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

#: Traces per key. A trace calls no answer model, but it embeds the question,
#: walks the graph and may ask the intent model, so a flood of them fills the
#: CPU. **Chosen**: well above one person's pace, low enough to stop a script.
TRACE_RATE_LIMIT = os.environ.get("GRAPHRAG_TRACE_RATE_LIMIT", "60/minute")
_TRACE_LIMIT = parse(TRACE_RATE_LIMIT)


def trace_rate_guard(request: Request) -> None:
    """Refuse a trace past the per-key rate with 429.

    A dependency rather than a decorator: the trace route is declared inside
    the application factory, and a decorator there would register the limit
    again for every application built. It counts in the limiter's own store,
    so ``limiter.reset()`` and ``limiter.enabled`` govern it too.
    """
    if not limiter.enabled:
        return
    if not limiter.limiter.hit(_TRACE_LIMIT, "trace", _user_key(request)):
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {TRACE_RATE_LIMIT}")
