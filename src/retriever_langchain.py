"""This project's retrieval, as a retriever an agent framework can hold.

A thin conversion at a boundary: run the query the facade already runs, and
turn each result into the container the framework expects. Nothing here
retrieves, scores, or ranks — every number in a document's metadata was
computed by the module that already computed it.

**This module lives outside every package deliberately.** It needs
``langchain-core``, and the packages under ``src/`` do not; keeping it a
sibling means installing this adapter is opt-in and nothing else grows a
dependency because it exists. That is the same arrangement the tracing adapter
uses, for the same reason.

Why the score breakdown survives the boundary
---------------------------------------------

``metadata`` carries the total, each arm's contribution and the decay factor
separately. Collapsing them into one number here would be the single change
that makes this adapter useless for the thing it exposes: a node at 0.42 says
nothing on its own — it could be a strong vector match that recency cut, a
strong traversal the vector arm never saw, or both arms agreeing weakly.
Anything built on this retriever would be unable to tell those apart, and the
two-arm design would be invisible from outside.

An arm that did not find a node reports **0.0 rather than being absent**.
Absent and zero are the same number and different facts, and a consumer forced
to distinguish "this key is missing" from "this arm scored nothing" will get
it wrong.

Where the text comes from
-------------------------

A ranked result is an entity, and an entity is not prose. The text is fetched
for the whole result set in **one** call rather than one per result — the
store already answers that question in a batch, and asking per node would turn
one query into as many as there are results.

A result whose text cannot be reached still produces a document. Its label, or
failing that its type and id, is better than an empty one: an empty document
is indistinguishable from a retrieval that found nothing, and a caller
counting evidence would count it.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

#: What an arm reports when it did not contribute. Named rather than written
#: as a bare zero, because the decision it encodes — report the zero, never
#: drop the key — is the point.
NO_CONTRIBUTION = 0.0


def _labels_from(run) -> dict[str, str]:
    """Any labels the run happened to carry, by id.

    The fused hit carries a type and no label; only the vector arm's hits
    carry one. So this reads the arm results the run already returns rather
    than asking the store again — a label is a nicety for the fallback text,
    not worth a second query.
    """
    labels: dict[str, str] = {}
    for hit in getattr(getattr(run, "vector", None), "hits", ()) or ():
        label = getattr(hit, "label", None)
        if label:
            labels[hit.id] = label
    return labels


def _content_for(hit, chunks, labels: dict[str, str]) -> str:
    """The text of a result, or the best stand-in for it.

    Chunks first, joined in the order the store returned them. Then the label.
    Then the type and id, which is always available and always says something.
    """
    text = "\n\n".join(
        (chunk.get("content") or "").strip()
        for chunk in chunks
        if (chunk.get("content") or "").strip()
    )
    if text:
        return text

    label = labels.get(hit.id)
    if label:
        return label

    node_type = getattr(hit, "node_type", None)
    return f"{node_type} {hit.id}" if node_type else hit.id


def _metadata_for(hit, chunks) -> dict[str, Any]:
    """Everything that produced this result's position, kept apart."""
    return {
        "id": hit.id,
        "score": float(hit.score),
        "vector_score": float(getattr(hit, "vector_score", NO_CONTRIBUTION)),
        "graph_score": float(getattr(hit, "graph_score", NO_CONTRIBUTION)),
        "decay": float(getattr(hit, "decay", 1.0)),
        "node_type": getattr(hit, "node_type", None),
        "age_days": getattr(hit, "age_days", None),
        "found_by_both": bool(getattr(hit, "found_by_both", False)),
        "chunk_ids": [chunk["id"] for chunk in chunks if chunk.get("id")],
        "sources": [chunk["path"] for chunk in chunks if chunk.get("path")],
    }


def documents_from(run, texts: dict[str, list[dict[str, Any]]] | None = None):
    """A finished run as documents, with no framework call involved.

    Separate from the retriever so the conversion can be exercised on a run
    built by hand, and so a caller holding a run for another reason can reuse
    it without going through a retriever it does not want.
    """
    texts = texts or {}
    labels = _labels_from(run)

    documents = []
    for hit in getattr(run, "hits", ()) or ():
        chunks = list(texts.get(hit.id) or ())
        documents.append(
            Document(
                page_content=_content_for(hit, chunks, labels),
                metadata=_metadata_for(hit, chunks),
            )
        )
    return documents


def routed_documents(response) -> list[Document]:
    """Convert the native router response using its shared source-text rules."""
    from .retrieval.response import format_page_content

    seen: set[str] = set()
    return [
        Document(
            page_content=format_page_content(node, seen),
            metadata={
                "id": node.id,
                "score_total": node.score_total,
                "score_vector": node.score_vector,
                "score_graph": node.score_graph,
                "trace_log": response.trace_log,
            },
        )
        for node in response.results
    ]


class GraphRetriever(BaseRetriever):
    """This project's two-arm retrieval, behind the framework's interface.

    Constructed from the facade, because that is what owns a query: it holds
    the open store, the embedder, and the optional extractor and judge.

    **Those optional pieces are configured on the facade, not here.** The
    retrieval path takes them when the engine is built rather than per query,
    so accepting them again at this boundary would be a second place they
    live and a second thing to keep in step. They keep degrading exactly as
    they already do — without an extractor the first seed tier cannot run and
    selection falls to the vector tiers; without a judge an unmatched query
    takes the conceptual fallback — and none of that is re-implemented here.

    ``k`` is optional and forwarded only when set, so an unset one keeps
    whatever default the sequence applies rather than this holding a copy.
    """

    engine: Any
    k: int | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _texts_for(self, run) -> dict[str, list[dict[str, Any]]]:
        """Source text for every ranked id, in one call.

        The store answers this for a set of ids in a single query. Asking per
        result would turn one query into as many as there are results, on a
        path that already ran two arms and a fusion.

        A store that cannot answer yields no text rather than failing the
        retrieval: the ranking is still a real answer, and a document falls
        back to its label.
        """
        try:
            return self.engine.store.documents_for_entities(run.ids)
        except Exception:
            return {}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if hasattr(self.engine, "route"):
            return routed_documents(self.engine.route(query, self.k))
        run = self.engine.retrieve(query, self.k)
        return documents_from(run, self._texts_for(run))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> list[Document]:
        """The same call, off the calling thread.

        The facade already has an asynchronous entry point that hands the
        blocking work to an executor, so this uses it rather than inventing a
        second arrangement. Nothing here adds a timeout or a retry: what the
        wrapped call has an opinion about is the opinion that holds.
        """
        if hasattr(self.engine, "route_async"):
            return routed_documents(await self.engine.route_async(query, self.k))
        run = await self.engine.retrieve_async(query, self.k)
        return documents_from(run, self._texts_for(run))
