"""Which organisation the current request belongs to.

``request.state.org_id`` is the record, and this is the way to read it from
code that never sees a ``Request``. The store, the retrieval arms and the
graph builder are all called several frames below a route and none of them
take a request; threading one through every signature to carry a single string
would put a web framework's type into modules that have no other reason to
know one exists.

Reading it is deliberately loud
-------------------------------

``current_org`` raises when nothing is set rather than returning ``None``. A
tenant-scoped query with no tenant is not a query with a missing filter — it
is a query across every tenant, and a caller that treats ``None`` as "no
filter" reads everyone's data. The failure has to happen at the read, not at
whatever the caller does with an absent value.

Async and threads
-----------------

A ``ContextVar`` is per-task, and a task's context is *copied* into the worker
thread when a synchronous route is run in the threadpool. So a value set in an
async dependency is visible to an ``async def`` route on the same task and to a
plain ``def`` route running in a thread. It is not visible to a thread started
by hand, and a value set inside a threadpool route does not travel back out —
both are properties of the copy, and neither matters here because the only
writer is the dependency.

The token is reset when the request ends. Without that, a task reused for the
next request would start with the previous request's organisation still set,
which is the exact failure this module exists to prevent.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

#: Private, so nothing outside this module can set it without going through
#: the function that says what setting it means.
_current_org: ContextVar[str | None] = ContextVar("graphrag_current_org", default=None)


class NoCurrentOrg(RuntimeError):
    """Read outside a request, or inside one that resolved no tenant."""


def set_current_org(org_id: str) -> Token:
    """Bind the organisation for this task. Returns the token to reset with."""
    return _current_org.set(org_id)


def reset_current_org(token: Token) -> None:
    """Unbind, restoring whatever was bound before."""
    _current_org.reset(token)


def current_org() -> str:
    """The organisation this request belongs to.

    Raises rather than returning ``None``: see the module docstring. A caller
    that genuinely wants "tenant or nothing" should ask for
    ``current_org_or_none`` and handle the absence explicitly.
    """
    org_id = _current_org.get()
    if org_id is None:
        raise NoCurrentOrg(
            "no organisation is bound to this context; a tenant-scoped call "
            "was made outside a request or before tenant resolution ran"
        )
    return org_id


def current_org_or_none() -> str | None:
    """The organisation, or ``None``. For code that has a real answer for both."""
    return _current_org.get()
