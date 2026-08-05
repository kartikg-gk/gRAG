"""Query tracing — record what retrieval returned, and see what was used.

A self-contained unit: standard library only, and it imports nothing from the
rest of the pipeline. Copy this directory elsewhere and it still works.

Four concerns, four modules, and the seams between them are deliberate:

* ``schema``   — the data shape. No behaviour.
* ``capture``  — accumulation. Never learns an output path.
* ``store``    — persistence. Never learns how recording works.
* ``classify`` — measurement, and separately the verdict.
* ``viewer``   — applies a threshold at render time.

The trace stores ``overlap`` and no verdict. Whether an item counts as used is
computed when a trace is read, so changing the threshold reclassifies every
archived trace rather than freezing each at the cutoff current when it was
written.

    from src.tracing import Recorder, save, render

    recorder = Recorder(producer="my-pipeline")
    with recorder.span("retrieve", kind="retriever"):
        recorder.record_query(question)
        recorder.record_items(my_retriever(question))
    recorder.record_answer(answer)

    trace = recorder.finish()
    save(trace)
    print(render(trace))
"""

from .capture import (
    KIND_CHAIN,
    KIND_LLM,
    KIND_RETRIEVER,
    KIND_TOOL,
    UNKNOWN_SOURCE,
    Recorder,
    capture,
)
from .classify import (
    DEFAULT_THRESHOLD,
    MIN_TOKEN_LENGTH,
    is_used,
    overlap_score,
    score_overlaps,
    used_count,
)
from .schema import (
    ARM_GRAPH,
    ARM_HYBRID,
    ARM_UNKNOWN,
    ARM_VECTOR,
    ARMS,
    PRODUCER_UNKNOWN,
    SCHEMA_VERSION,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_RUNNING,
    Span,
    Trace,
    TraceEdge,
    TraceItem,
    to_dict,
    trace_from_dict,
)
from .store import (
    DEFAULT_TRACE_DIR,
    EMPTY_SLUG,
    TRACE_DIR_ENV,
    default_path,
    finish_and_save,
    load,
    save,
    slugify,
    trace_dir,
    trace_filename,
)
from .viewer import render, render_file

__all__ = [
    # schema
    "SCHEMA_VERSION",
    "Trace",
    "TraceItem",
    "TraceEdge",
    "Span",
    "to_dict",
    "trace_from_dict",
    "ARM_VECTOR",
    "ARM_GRAPH",
    "ARM_HYBRID",
    "ARM_UNKNOWN",
    "ARMS",
    "PRODUCER_UNKNOWN",
    "STATUS_RUNNING",
    "STATUS_OK",
    "STATUS_ERROR",
    # accumulation
    "capture",
    "Recorder",
    "UNKNOWN_SOURCE",
    "KIND_CHAIN",
    "KIND_RETRIEVER",
    "KIND_LLM",
    "KIND_TOOL",
    # persistence
    "save",
    "load",
    "finish_and_save",
    "default_path",
    "trace_dir",
    "trace_filename",
    "slugify",
    "TRACE_DIR_ENV",
    "DEFAULT_TRACE_DIR",
    "EMPTY_SLUG",
    # measurement and verdict
    "overlap_score",
    "score_overlaps",
    "is_used",
    "used_count",
    "DEFAULT_THRESHOLD",
    "MIN_TOKEN_LENGTH",
    # display
    "render",
    "render_file",
]
