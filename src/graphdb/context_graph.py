"""The graph store: entities, relationships, documents and their embeddings.

Everything database-specific lives behind this class. Queries are Cypher,
results come back as ordinary lists of dicts, and no caller ever sees a driver
object. Business logic stays out — resolution decides *what* is the same thing,
this decides how it is written down.

Concurrency
-----------

The engine serialises writes and scales reads across connections, so this holds
**one write connection behind a lock** and **a bounded pool of read
connections**. A read leases a connection, materialises its rows, and returns
the connection — the lease is a context manager so it comes back even when the
query raises. Nothing opens a connection per query.

Timestamps
----------

The ``ts`` columns are ``INT64``. The conversion happens once, at the edge, in
``_epoch`` and ``_moment`` — the column stays integral and callers keep passing
``datetime``. Unknown times are stored **NULL rather than 0**: this project's
rule is that an absent timestamp stays absent, because ``0`` reads as 1970 to a
recency scorer and buries the row rather than leaving it alone.

Embeddings
----------

The column is ``FLOAT[dim]``, which is float32. A vector written and read back
is therefore close to what went in, not bit-identical — ``0.1`` returns as
``0.10000000149011612``. Anything comparing stored vectors must use a
tolerance.
"""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..common.config import (
    CONFIDENCE,
    DOC_TABLE,
    EMBEDDING_DIMENSION,
    MENTIONS_TABLE,
    NODE_TABLE,
    POOL_TIMEOUT_SECONDS,
    READ_POOL_SIZE,
    REL_TABLE,
    VECTOR_INDEX_NAME,
    VECTOR_METRIC,
)

#: Columns every table is expected to have, so a database opened from an older
#: build can be brought forward rather than rebuilt. Kept beside the DDL that
#: creates them: two lists that can disagree is how a migration silently stops
#: covering a column.
_EXPECTED_COLUMNS: dict[str, dict[str, str]] = {
    NODE_TABLE: {
        "id": "STRING",
        "label": "STRING",
        "type": "STRING",
        "ts": "INT64",
        "embedding": f"FLOAT[{EMBEDDING_DIMENSION}]",
    },
    DOC_TABLE: {"id": "STRING", "path": "STRING", "content": "STRING"},
    REL_TABLE: {"confidence": "DOUBLE", "relation": "STRING", "ts": "INT64"},
    MENTIONS_TABLE: {},
}


class ContextGraph:
    """Persistence and retrieval for the knowledge graph.

    Open it, use it, close it — or use it as a context manager, which closes it
    for you.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        read_pool_size: int = READ_POOL_SIZE,
        pool_timeout: float = POOL_TIMEOUT_SECONDS,
        read_only: bool = False,
    ) -> None:
        import ladybug

        self._ladybug = ladybug
        self.path = str(path)
        self._pool_timeout = pool_timeout
        self._closed = False
        #: Whether this handle may write. Exposed because a caller that has to
        #: be able to state it did not write needs to be able to check it.
        self.read_only = read_only

        # The engine replays its write-ahead log and checkpoints on connect, so
        # an ordinary open changes the file before any statement runs. A tool
        # that only reads cannot otherwise show that it only read.
        self._database = ladybug.Database(self.path, read_only=read_only)

        # One writer, serialised. The lock covers schema changes and index
        # rebuilds too: those mutate the same catalogue the writes touch.
        self._write_lock = threading.Lock()
        self._write_connection = self._new_connection()

        self._read_pool: queue.Queue = queue.Queue(maxsize=read_pool_size)
        self._read_connections = [
            self._new_connection() for _ in range(read_pool_size)
        ]
        for connection in self._read_connections:
            self._read_pool.put(connection)

    # -- connections -------------------------------------------------------

    def _new_connection(self):
        """A connection with the vector extension available.

        Loaded per connection rather than once per database: the extension is a
        property of the session, and a read connection that lacks it fails only
        when someone runs a vector query on it.
        """
        connection = self._ladybug.Connection(self._database)
        for statement in ("LOAD EXTENSION vector", "INSTALL vector"):
            try:
                connection.execute(statement)
                if statement.startswith("LOAD"):
                    return connection
            except Exception:
                continue
        # Installed but not yet loaded in this session.
        try:
            connection.execute("LOAD EXTENSION vector")
        except Exception:
            pass
        return connection

    @contextmanager
    def _read(self):
        """Lease a read connection and return it, whatever happens."""
        self._refuse_if_closed()
        try:
            connection = self._read_pool.get(timeout=self._pool_timeout)
        except queue.Empty as exc:
            raise RuntimeError(
                f"no read connection free after {self._pool_timeout}s; "
                f"the pool holds {len(self._read_connections)}"
            ) from exc
        try:
            yield connection
        finally:
            self._read_pool.put(connection)

    def _refuse_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("this ContextGraph is closed")

    # -- running queries ---------------------------------------------------

    @staticmethod
    def _rows(result) -> list[list[Any]]:
        """Drain a driver result into plain lists, before the lease ends."""
        drained = []
        while result.has_next():
            drained.append(result.get_next())
        return drained

    def query(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[list[Any]]:
        """Run a read query on a leased connection. Rows are materialised."""
        with self._read() as connection:
            result = connection.execute(cypher, parameters=parameters or {})
            return self._rows(result)

    def execute(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[list[Any]]:
        """Run a write, schema change or index operation. Serialised."""
        self._refuse_if_closed()
        with self._write_lock:
            result = self._write_connection.execute(
                cypher, parameters=parameters or {}
            )
            return self._rows(result)

    # -- schema ------------------------------------------------------------

    def initialize_schema(self) -> None:
        """Create the tables and the vector index. Safe to call repeatedly."""
        self.execute(
            f"CREATE NODE TABLE IF NOT EXISTS {NODE_TABLE}("
            f"  id STRING PRIMARY KEY,"
            f"  label STRING,"
            f"  type STRING,"
            f"  ts INT64,"
            f"  embedding FLOAT[{EMBEDDING_DIMENSION}]"
            f")"
        )
        self.execute(
            f"CREATE NODE TABLE IF NOT EXISTS {DOC_TABLE}("
            f"  id STRING PRIMARY KEY, path STRING, content STRING"
            f")"
        )
        self.execute(
            f"CREATE REL TABLE IF NOT EXISTS {REL_TABLE}("
            f"  FROM {NODE_TABLE} TO {NODE_TABLE},"
            f"  confidence DOUBLE, relation STRING, ts INT64"
            f")"
        )
        self.execute(
            f"CREATE REL TABLE IF NOT EXISTS {MENTIONS_TABLE}("
            f"  FROM {DOC_TABLE} TO {NODE_TABLE}"
            f")"
        )
        self.build_vector_index()

    def table_columns(self, table: str) -> dict[str, str]:
        """Column name to declared type, as the database currently has it."""
        rows = self.query(f"CALL TABLE_INFO('{table}') RETURN *")
        return {row[1]: row[2] for row in rows}

    def migrate(self) -> list[str]:
        """Add any expected column the database does not have yet.

        Idempotent, and it preserves rows: ``ALTER TABLE ADD`` leaves existing
        values alone and fills the new column with NULL, which is what an
        unknown value should be. ``CREATE TABLE IF NOT EXISTS`` will not do
        this — it sees the table, does nothing, and leaves the column missing.

        Returns the columns it added, as ``"table.column"``, so a caller can
        report what changed rather than guess.
        """
        existing_tables = {row[1] for row in self.query("CALL SHOW_TABLES() RETURN *")}
        added: list[str] = []

        for table, columns in _EXPECTED_COLUMNS.items():
            if table not in existing_tables:
                continue
            present = self.table_columns(table)
            for column, declared in columns.items():
                if column in present:
                    continue
                self.execute(f"ALTER TABLE {table} ADD {column} {declared}")
                added.append(f"{table}.{column}")

        return added

    # -- entities ----------------------------------------------------------

    def upsert_entity(
        self,
        entity_id: str,
        label: str,
        entity_type: str,
        timestamp: datetime | None = None,
        embedding: Sequence[float] | None = None,
    ) -> None:
        """Write an entity, or fold a re-observation into an existing one.

        One statement carries both guarantees, which is the point of writing it
        this way rather than reading, deciding in Python, and writing back:

        * **label and type are write-once.** ``ON CREATE SET`` assigns them and
          ``ON MATCH SET`` does not touch them, so a noisier surface form
          arriving later cannot overwrite a canonical name.
        * **the timestamp only moves forward.** A stale document processed last
          must not drag recency backward.
        """
        vector = self._checked_embedding(embedding, entity_id)
        self.execute(
            f"MERGE (e:{NODE_TABLE} {{id: $id}}) "
            f"ON CREATE SET e.label = $label, e.type = $type, "
            f"              e.ts = CAST($ts AS INT64), e.embedding = $embedding "
            f"ON MATCH  SET e.ts = CASE "
            f"                WHEN CAST($ts AS INT64) IS NULL THEN e.ts "
            f"                WHEN e.ts IS NULL THEN CAST($ts AS INT64) "
            f"                WHEN CAST($ts AS INT64) > e.ts "
            f"                     THEN CAST($ts AS INT64) ELSE e.ts END",
            {
                "id": entity_id,
                "label": label,
                "type": entity_type,
                "ts": _epoch(timestamp),
                "embedding": vector,
            },
        )

    def set_embedding(self, entity_id: str, embedding: Sequence[float]) -> None:
        """Attach or replace an entity's vector, without touching anything else."""
        vector = self._checked_embedding(embedding, entity_id)
        self.execute(
            f"MATCH (e:{NODE_TABLE} {{id: $id}}) SET e.embedding = $embedding",
            {"id": entity_id, "embedding": vector},
        )

    @staticmethod
    def _checked_embedding(
        embedding: Sequence[float] | None, subject: str
    ) -> list[float] | None:
        """Refuse a vector of the wrong width at the point it is written.

        The column is fixed-width, so a wrong-width vector is not a value the
        database can hold. Raising here names the entity; letting the driver
        raise names only the column.
        """
        if embedding is None:
            return None
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"embedding for {subject!r} has {len(embedding)} dimensions, "
                f"but the {NODE_TABLE}.embedding column is "
                f"{EMBEDDING_DIMENSION}-wide"
            )
        return [float(value) for value in embedding]

    def find_by_label(self, label: str) -> list[dict[str, Any]]:
        """Entities whose label matches, ignoring case.

        Exact matching, not fuzzy: ``OpenAI``, ``openai`` and ``OPENAI`` are the
        same label written three ways, and that is a normalisation question.
        Anything looser belongs in ``vector_search``.
        """
        rows = self.query(
            f"MATCH (e:{NODE_TABLE}) WHERE lower(e.label) = lower($label) "
            f"RETURN e.id, e.label, e.type, e.ts ORDER BY e.id",
            {"label": label},
        )
        return [_entity(row) for row in rows]

    def entity_exists(self, entity_id: str) -> bool:
        rows = self.query(
            f"MATCH (e:{NODE_TABLE} {{id: $id}}) RETURN count(*)", {"id": entity_id}
        )
        return bool(rows and rows[0][0])

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        rows = self.query(
            f"MATCH (e:{NODE_TABLE} {{id: $id}}) RETURN e.id, e.label, e.type, e.ts",
            {"id": entity_id},
        )
        return _entity(rows[0]) if rows else None

    def get_embedding(self, entity_id: str) -> list[float] | None:
        rows = self.query(
            f"MATCH (e:{NODE_TABLE} {{id: $id}}) RETURN e.embedding",
            {"id": entity_id},
        )
        if not rows or rows[0][0] is None:
            return None
        return [float(value) for value in rows[0][0]]

    # -- relationships -----------------------------------------------------

    def upsert_relationship(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        confidence: float | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Write a relationship, keeping whichever evidence is stronger.

        ``confidence`` defaults to the configured weight for ``relation``, so a
        caller naming a relation gets that relation's price without repeating
        it.

        **Order matters and is deliberate.** Every branch compares against
        ``r.confidence``, and confidence is assigned *last*. Assigning it first
        would leave the later branches comparing the new value against itself,
        so every write would look like an upgrade and a weak proximity edge
        would overwrite an authorship fact.
        """
        weight = CONFIDENCE[relation] if confidence is None else float(confidence)
        self.execute(
            f"MATCH (a:{NODE_TABLE} {{id: $source}}), (b:{NODE_TABLE} {{id: $target}}) "
            f"MERGE (a)-[r:{REL_TABLE}]->(b) "
            f"ON CREATE SET r.confidence = $confidence, r.relation = $relation, "
            f"              r.ts = CAST($ts AS INT64) "
            f"ON MATCH  SET "
            f"  r.relation = CASE WHEN $confidence > r.confidence "
            f"                    THEN $relation ELSE r.relation END, "
            f"  r.ts = CASE WHEN $confidence > r.confidence "
            f"              THEN CAST($ts AS INT64) ELSE r.ts END, "
            f"  r.confidence = CASE WHEN $confidence > r.confidence "
            f"                      THEN $confidence ELSE r.confidence END",
            {
                "source": source_id,
                "target": target_id,
                "relation": relation,
                "confidence": weight,
                "ts": _epoch(timestamp),
            },
        )

    def get_relationship(
        self, source_id: str, target_id: str
    ) -> dict[str, Any] | None:
        rows = self.query(
            f"MATCH (a:{NODE_TABLE} {{id: $source}})-[r:{REL_TABLE}]->"
            f"(b:{NODE_TABLE} {{id: $target}}) "
            f"RETURN r.relation, r.confidence, r.ts",
            {"source": source_id, "target": target_id},
        )
        if not rows:
            return None
        relation, confidence, ts = rows[0]
        return {
            "relation": relation,
            "confidence": confidence,
            "timestamp": _moment(ts),
        }

    # -- documents ---------------------------------------------------------

    def upsert_document(self, document_id: str, path: str, content: str) -> None:
        """Store a source document, refreshing its path and text if re-ingested.

        A document's content is **not** write-once, where an entity's label is:
        the file changed, and the newer text is what is there. A label is a
        name chosen once; content is a fact about the file right now.

        **OPEN: whether the content column should exist at all.**

        Measured. Storing the text costs 1.005 stored bytes per raw byte —
        real source text does not compress in this store — which is 33.2% of
        the store across 67 project files, projecting to about 65 MB at 10,000
        documents. Not free, and not alarming either.

        Counted. Production call sites for the document half of this class:
        zero. Nothing writes a document, because graph construction still
        emits JSON and never reaches the store. Every exercise of these
        methods is a test.

        That is not enough to remove the column. Storage with no consumer
        looks like pure cost, but the reason there is no consumer is that the
        thing that would write documents has not been connected yet — absence
        of a reader today says nothing about whether retrieval will want the
        text tomorrow. Deleting the column now would be deciding a question on
        the strength of unfinished work.

        What would settle it. A production caller appearing. At that moment
        the question is answerable by reading one thing: does the caller read
        ``content`` back, or only ``path`` and the mention edges? If only the
        latter, the column goes.

        ``test_no_production_code_calls_the_document_api`` is what surfaces
        that moment. It is **supposed to fail** when the store is wired up.
        Re-pinning it would throw away the signal it exists to give; the
        correct response is to answer the question above and replace the test
        with one that pins the answer.
        """
        self.execute(
            f"MERGE (d:{DOC_TABLE} {{id: $id}}) "
            f"ON CREATE SET d.path = $path, d.content = $content "
            f"ON MATCH  SET d.path = $path, d.content = $content",
            {"id": document_id, "path": path, "content": content},
        )

    def add_mention(self, document_id: str, entity_id: str) -> None:
        """Record that a document mentions an entity. Idempotent via MERGE."""
        self.execute(
            f"MATCH (d:{DOC_TABLE} {{id: $doc}}), (e:{NODE_TABLE} {{id: $entity}}) "
            f"MERGE (d)-[:{MENTIONS_TABLE}]->(e)",
            {"doc": document_id, "entity": entity_id},
        )

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        rows = self.query(
            f"MATCH (d:{DOC_TABLE} {{id: $id}}) RETURN d.id, d.path, d.content",
            {"id": document_id},
        )
        if not rows:
            return None
        return {"id": rows[0][0], "path": rows[0][1], "content": rows[0][2]}

    def documents_for_entities(
        self, entity_ids: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Source documents mentioning each entity, grouped by entity.

        One batched query rather than one per entity. Every requested id appears
        in the result, including those nothing mentions — an empty list is an
        answer, a missing key is a hole the caller has to guess about.
        """
        wanted = list(dict.fromkeys(entity_ids))
        grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in wanted}
        if not wanted:
            return grouped

        rows = self.query(
            f"UNWIND $ids AS wanted "
            f"MATCH (d:{DOC_TABLE})-[:{MENTIONS_TABLE}]->"
            f"(e:{NODE_TABLE} {{id: wanted}}) "
            f"RETURN wanted, d.id, d.path, d.content ORDER BY wanted, d.id",
            {"ids": wanted},
        )
        for entity_id, doc_id, path, content in rows:
            grouped[entity_id].append(
                {"id": doc_id, "path": path, "content": content}
            )
        return grouped

    # -- vector search -----------------------------------------------------

    def build_vector_index(self, *, rebuild: bool = False) -> None:
        """Create the vector index, or rebuild it over current embeddings.

        **Writes after the build are searchable without a rebuild.** Measured
        on ladybug 0.19.1: three entities indexed at build time, three more
        written afterwards with no rebuild, and a query matching a post-build
        entity returns it top-ranked at similarity 1.000, with all six rows
        present. There is no staleness window to work around and nothing here
        tracks one.

        That is a property of the engine, not of this class, which is why
        ``test_a_write_after_the_index_build_is_searchable`` asserts it rather
        than trusting this paragraph. A future version could index only the
        rows present at creation, and then a caller would silently miss recent
        entities — no error, just absent rows.

        Creating an index that already exists is an error rather than a no-op,
        so an existing one is dropped first when rebuilding and tolerated
        otherwise.
        """
        if rebuild:
            try:
                self.execute(
                    f"CALL DROP_VECTOR_INDEX('{NODE_TABLE}', '{VECTOR_INDEX_NAME}')"
                )
            except Exception:
                pass  # nothing to drop

        try:
            self.execute(
                f"CALL CREATE_VECTOR_INDEX("
                f"'{NODE_TABLE}', '{VECTOR_INDEX_NAME}', 'embedding', "
                f"metric := '{VECTOR_METRIC}')"
            )
        except Exception as exc:
            if "already exists" not in str(exc):
                raise

    def vector_search(
        self, embedding: Sequence[float], k: int = 10
    ) -> list[dict[str, Any]]:
        """The k nearest entities to a query vector.

        Returns distance *and* similarity. The metric is cosine over
        L2-normalised vectors, so ``similarity = 1 - distance`` — reported
        rather than left for each caller to derive, because deriving it wrongly
        is silent.
        """
        vector = self._checked_embedding(embedding, "query")
        rows = self.query(
            f"CALL QUERY_VECTOR_INDEX("
            f"'{NODE_TABLE}', '{VECTOR_INDEX_NAME}', $query, $k) "
            f"RETURN node.id, node.label, node.type, node.ts, distance "
            f"ORDER BY distance ASC, node.id ASC",
            {"query": vector, "k": int(k)},
        )
        return [
            {
                "id": row[0],
                "label": row[1],
                "type": row[2],
                "timestamp": _moment(row[3]),
                "distance": float(row[4]),
                "similarity": 1.0 - float(row[4]),
            }
            for row in rows
        ]

    # -- graph retrieval ---------------------------------------------------

    def neighbors(self, entity_id: str, k: int = 10) -> list[dict[str, Any]]:
        """One hop out, strongest first.

        Ties break on the neighbour id so the order is the same on every run —
        an unordered tie makes a diff of two runs look like a change.
        """
        rows = self.query(
            f"MATCH (a:{NODE_TABLE} {{id: $id}})-[r:{REL_TABLE}]-(b:{NODE_TABLE}) "
            f"RETURN b.id, b.label, b.type, r.relation, r.confidence, r.ts "
            f"ORDER BY r.confidence DESC, b.id ASC LIMIT $k",
            {"id": entity_id, "k": int(k)},
        )
        return [_neighbor(row) for row in rows]

    def node_degree(self, entity_id: str) -> int:
        """How many distinct entities this one touches.

        Distinct neighbours rather than relationship rows: two edges to the same
        node is one connection, and counting rows would make a pair of parallel
        edges look like a hub.
        """
        rows = self.query(
            f"MATCH (a:{NODE_TABLE} {{id: $id}})-[:{REL_TABLE}]-(b:{NODE_TABLE}) "
            f"RETURN count(DISTINCT b.id)",
            {"id": entity_id},
        )
        return int(rows[0][0]) if rows else 0

    def expand_frontier(
        self,
        node_ids: Iterable[str],
        k: int = 10,
        max_degree: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Expand many nodes at once, grouped by the node they came from.

        **One query for the whole frontier**, not one per node. A traversal that
        issues a query per frontier node spends its time in round trips.

        ``max_degree`` filters hubs **per relation type**, not per node. A
        repository with 10,000 ``TOUCHES`` edges and 3 ``AUTHORED`` edges should
        lose the file list and keep the authorship: dropping the node entirely
        because one of its relations is broad throws away the useful half.
        """
        wanted = list(dict.fromkeys(node_ids))
        grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in wanted}
        if not wanted:
            return grouped

        rows = self.query(
            f"UNWIND $ids AS origin "
            f"MATCH (a:{NODE_TABLE} {{id: origin}})-[r:{REL_TABLE}]-"
            f"(b:{NODE_TABLE}) "
            f"RETURN origin, b.id, b.label, b.type, r.relation, r.confidence, r.ts "
            f"ORDER BY origin, r.confidence DESC, b.id ASC",
            {"ids": wanted},
        )

        # Degree per (origin, relation) — the unit the hub filter works on.
        per_relation: dict[tuple[str, str], set[str]] = {}
        for origin, neighbor_id, _, _, relation, _, _ in rows:
            per_relation.setdefault((origin, relation), set()).add(neighbor_id)

        for row in rows:
            origin, relation = row[0], row[4]
            if (
                max_degree is not None
                and len(per_relation[(origin, relation)]) > max_degree
            ):
                continue
            if len(grouped[origin]) >= k:
                continue
            grouped[origin].append(_neighbor(row[1:]))

        return grouped

    def subgraph(self, entity_ids: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        """The requested nodes, their one-hop neighbours, and the edges between.

        A requested node with no neighbours still appears. Returning only nodes
        that happen to have edges would silently drop exactly the isolated
        entities a caller most needs to see.
        """
        wanted = list(dict.fromkeys(entity_ids))
        if not wanted:
            return {"nodes": [], "edges": []}

        requested_rows = self.query(
            f"UNWIND $ids AS wanted "
            f"MATCH (e:{NODE_TABLE} {{id: wanted}}) "
            f"RETURN e.id, e.label, e.type, e.ts ORDER BY e.id",
            {"ids": wanted},
        )
        neighbor_rows = self.query(
            f"UNWIND $ids AS wanted "
            f"MATCH (a:{NODE_TABLE} {{id: wanted}})-[:{REL_TABLE}]-(b:{NODE_TABLE}) "
            f"RETURN DISTINCT b.id, b.label, b.type, b.ts ORDER BY b.id",
            {"ids": wanted},
        )

        nodes: dict[str, dict[str, Any]] = {}
        for row in requested_rows:
            nodes[row[0]] = {**_entity(row), "requested": True}
        for row in neighbor_rows:
            if row[0] not in nodes:
                nodes[row[0]] = {**_entity(row), "requested": False}

        visible = list(nodes)
        edge_rows = self.query(
            f"MATCH (a:{NODE_TABLE})-[r:{REL_TABLE}]->(b:{NODE_TABLE}) "
            f"WHERE list_contains($visible, a.id) AND list_contains($visible, b.id) "
            f"RETURN a.id, b.id, r.relation, r.confidence, r.ts "
            f"ORDER BY a.id, b.id",
            {"visible": visible},
        )
        edges = [
            {
                "source": row[0],
                "target": row[1],
                "relation": row[2],
                "confidence": row[3],
                "timestamp": _moment(row[4]),
            }
            for row in edge_rows
        ]

        return {"nodes": list(nodes.values()), "edges": edges}

    # -- statistics --------------------------------------------------------

    def count_nodes(self) -> int:
        rows = self.query(f"MATCH (e:{NODE_TABLE}) RETURN count(*)")
        return int(rows[0][0]) if rows else 0

    def count_documents(self) -> int:
        rows = self.query(f"MATCH (d:{DOC_TABLE}) RETURN count(*)")
        return int(rows[0][0]) if rows else 0

    def most_connected(self, limit: int = 10) -> list[dict[str, Any]]:
        """The hubs, by distinct neighbours rather than by edge rows."""
        rows = self.query(
            f"MATCH (a:{NODE_TABLE})-[:{REL_TABLE}]-(b:{NODE_TABLE}) "
            f"RETURN a.id, a.label, a.type, count(DISTINCT b.id) AS degree "
            f"ORDER BY degree DESC, a.id ASC LIMIT $limit",
            {"limit": int(limit)},
        )
        return [
            {"id": row[0], "label": row[1], "type": row[2], "degree": int(row[3])}
            for row in rows
        ]

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close every connection and the database. Safe to call twice."""
        if self._closed:
            return
        self._closed = True

        for connection in self._read_connections:
            _close_quietly(connection)
        self._read_connections = []
        while not self._read_pool.empty():
            try:
                self._read_pool.get_nowait()
            except queue.Empty:
                break

        _close_quietly(self._write_connection)
        _close_quietly(self._database)

    def __enter__(self) -> "ContextGraph":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# --------------------------------------------------------------------------
# the edge where datetimes become integers, and driver rows become dicts
# --------------------------------------------------------------------------


def _epoch(moment: datetime | None) -> int | None:
    """A datetime as whole seconds, or NULL.

    The column is INT64 and this is the only place the conversion happens.

    **``None`` stays ``None`` rather than becoming ``0``.** Zero is not a
    neutral filler, it is the first second of 1970 — a real date, and one a
    recency calculation will happily rank. Collapsing unknown into zero makes
    every undated entity the oldest thing in the graph, buried under
    everything with a real timestamp, and nothing downstream can tell that
    apart from a genuine 1970 record. Absent is not a date and sorts nowhere.

    Nothing scores recency yet, which is why this is written down and pinned
    by test rather than left to be rediscovered: between the decision and its
    first consumer there is nothing that would notice the rule breaking.
    ``test_a_stored_zero_is_distinct_from_absent`` holds the two apart, and a
    reopen test confirms the property belongs to the column rather than to a
    live handle.

    A naive datetime is read as UTC. Guessing local time would make the same
    document import differently on two machines.

    Every query binding this value wraps it in ``CAST(... AS INT64)``. The
    driver infers a parameter's type from the first value it is bound to and
    caches the prepared statement, so a ``None`` seen first makes the parameter
    boolean and every later integer fails with a type-change error. The cast
    fixes the type at prepare time instead of at first use.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp())


def _moment(seconds: int | None) -> datetime | None:
    """The inverse of ``_epoch``. NULL stays unknown."""
    if seconds is None:
        return None
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc)


def _entity(row: Sequence[Any]) -> dict[str, Any]:
    return {
        "id": row[0],
        "label": row[1],
        "type": row[2],
        "timestamp": _moment(row[3]),
    }


def _neighbor(row: Sequence[Any]) -> dict[str, Any]:
    return {
        "id": row[0],
        "label": row[1],
        "type": row[2],
        "relation": row[3],
        "confidence": row[4],
        "timestamp": _moment(row[5]),
    }


def _close_quietly(handle) -> None:
    """Close what can be closed. A double close must not raise."""
    closer = getattr(handle, "close", None)
    if closer is None:
        return
    try:
        closer()
    except Exception:
        pass


def open_context_graph(
    path: str | Path,
    *,
    initialize: bool = True,
    migrate: bool = True,
    read_only: bool = False,
) -> ContextGraph:
    """Open a graph store, ready to use.

    Creates the tables when asked, then brings an older database forward. Both
    are idempotent, so this is the ordinary way to open a store whether or not
    one already exists at the path.

    ``read_only`` opens a handle that cannot write, and forces both of those
    steps off: each writes to the catalogue, so neither can run against such a
    handle, and leaving them to fail would make the flag depend on argument
    order. A store opened this way leaves the file's bytes alone, which is what
    lets a tool over a store say it did not write to it.
    """
    graph = ContextGraph(path, read_only=read_only)
    try:
        if initialize and not read_only:
            graph.initialize_schema()
        if migrate and not read_only:
            graph.migrate()
    except Exception:
        graph.close()
        raise
    return graph
