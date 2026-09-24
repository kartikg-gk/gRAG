"""What is wired to ``ContextGraph``, and what is not.

A source scan, deliberately. It imports neither ``ladybug`` nor
``src.graphdb.context_graph``, so it runs on an interpreter where the graph
store cannot be opened at all. That matters: ``test_context_graph.py`` guards
itself with ``importorskip``, so on a machine without ladybug its whole suite
disappears silently. A fact that only holds while those tests run is not
pinned — it is merely usually checked.

What is pinned here has moved once already, and the move is the point.

It used to be that the document half of ``ContextGraph`` had no production
caller at all. That assertion was written to fail the moment a writer
appeared, and it did: ``src/knowledge/persist.py`` now writes documents and
mention edges during ingest. The failure named both call sites, which is what
it was for.

What that settled is narrower than it looks. Documents are written. Nothing
reads them back. The ``content`` column costs about one stored byte per raw
content byte and roughly a third of the store on real files, and that cost is
now being paid for a column with a writer and no reader — which is a worse
position than having neither, not a better one.

So the open question moved rather than closed, and
``test_nothing_in_production_reads_a_document_back`` now holds it. That one is
supposed to fail in turn, when retrieval starts reading document rows, and the
question to answer at that moment is whether the reader takes ``content`` or
only ``path`` and the mention edges. Writing stays pinned to exactly one
module so a second writer cannot appear unnoticed while the question is open.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1]
TESTS = SOURCE_ROOT / "tests"
EXAMPLES = SOURCE_ROOT.parent / "examples"

#: The document half of the store's API.
DOCUMENT_API = frozenset(
    {"upsert_document", "get_document", "documents_for_entities", "add_mention"}
)


def call_sites(root: Path, *, skip: Path | None = None) -> dict[str, list[str]]:
    """Every call to a ``DOCUMENT_API`` name under ``root``, by method name.

    Parsed rather than grepped. A grep matches the name in a docstring, in a
    comment and in this test's own ``DOCUMENT_API`` literal; only a call is a
    caller, and only calls decide whether the column has a reader.
    """
    found: dict[str, list[str]] = {name: [] for name in DOCUMENT_API}

    for path in sorted(root.rglob("*.py")):
        if skip is not None and skip in path.parents:
            continue
        if "__pycache__" in path.parts:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else None
            )
            if name in DOCUMENT_API:
                try:
                    where = path.relative_to(SOURCE_ROOT.parent)
                except ValueError:
                    # A scanned tree outside the project — the scanner's own
                    # test uses tmp_path. Reporting the absolute path is fine;
                    # failing to report a call site would not be.
                    where = path
                # as_posix(): the assertions below compare against
                # forward-slash paths, and a native separator makes this
                # test pass on one OS and fail on another.
                found[name].append(f"{Path(where).as_posix()}:{node.lineno}")

    return found


# --------------------------------------------------------------------------
# the pin
# --------------------------------------------------------------------------


def test_documents_are_written_only_by_the_two_chosen_modules():
    """A writer now exists, and there is exactly one of it.

    This replaces an assertion that no production code called the document API
    at all. That assertion existed to fail at this moment, and it did — naming
    ``persist.upsert_document`` and ``persist.add_mention`` — so it has done
    its job and is rewritten rather than re-pinned.

    **What the failure settled, and what it did not.** It established that
    documents are written. It did not establish that anything reads them back,
    and that was always the question. The content column costs about one
    stored byte per raw byte and roughly a third of the store on real files;
    that cost is now being paid, and it is paid for a column with a writer and
    no reader.

    So the column's fate turns on a **reader** appearing, not a writer. The
    test below is the one holding that open now. Writing stays pinned here to
    one module so a second writer cannot appear unnoticed and make the
    question harder to answer.
    """
    production = call_sites(SOURCE_ROOT, skip=TESTS)
    writers = {
        name: sites
        for name, sites in production.items()
        if name in {"upsert_document", "add_mention"} and sites
    }

    # Two writers, each chosen: the local build, and the tenant compile, which
    # writes each item's text so a tenant answer has something behind a name.
    # A third module writing documents is still a failure here.
    writing_modules = ("src/knowledge/persist.py", "src/worker/compiler.py")

    assert set(writers) == {"upsert_document", "add_mention"}
    for name, sites in writers.items():
        for site in sites:
            assert site.startswith(writing_modules), f"{name} written from {site}"
        assert any(site.startswith(writing_modules[0]) for site in sites), sites
        assert any(site.startswith(writing_modules[1]) for site in sites), sites


def test_the_content_column_now_has_a_production_reader_that_reads_content():
    """The open question, answered rather than re-pinned.

    Its predecessor asserted that nothing in production read a document back,
    and said what to do when a reader appeared: look at what it reads. If it
    takes ``content``, the column has earned its cost and this test should pin
    that; if it takes only ``path`` and the mentions, the column is storage
    with no consumer and should go.

    **A reader appeared and it takes content.** ``Engine.context`` reads
    document rows and puts their text into the string it returns, so the
    column is load-bearing for the facade's prompt output. It is no longer
    written and read by nothing.

    Worth recording about the scanner rather than the column: the HTTP query
    route reached ``documents_for_entities`` before the facade did, by handing
    it to an executor rather than calling it. Passed as an object it is an
    Attribute and not a Call, so the predecessor never saw it — a reader can
    hide from this scan by being passed instead of called.
    """
    production = call_sites(SOURCE_ROOT, skip=TESTS)
    readers = {
        name: sites
        for name, sites in production.items()
        if name in {"get_document", "documents_for_entities"} and sites
    }

    assert readers, "the document reader disappeared; the column is undecidable again"

    # And what it does with the row is the part that decides the column.
    engine_source = (SOURCE_ROOT / "engine.py").read_text(encoding="utf-8")
    results_source = (SOURCE_ROOT / "results.py").read_text(encoding="utf-8")

    assert "documents_for_entities" in engine_source
    assert 'get("content")' in results_source, (
        "the reader stopped reading content; if it only needs path and "
        "mentions the column should go rather than this being relaxed"
    )


def test_the_examples_do_not_call_the_document_api_either():
    """The demos write JSON through ``GraphBuilder``, not to the store.

    Kept separate from the production scan because the examples are the most
    likely place a first caller appears — ``examples/graph_demo.py`` is where
    someone would naturally reach for persistence.
    """
    offenders = {
        name: sites for name, sites in call_sites(EXAMPLES).items() if sites
    }

    assert offenders == {}


def test_the_tests_exercise_the_document_api_thoroughly():
    """Every document method is driven by some test, several times over.

    This exists so the assertions above cannot pass for the wrong reason. If
    the API were deleted outright, "nothing reads a document back" would still
    hold and would mean the opposite of what it means today.

    A floor rather than an exact count. The exact number was pinned while the
    only call sites were in one file; it now moves whenever a test is added,
    and a number that has to be bumped on unrelated work stops being read and
    starts being edited reflexively. The floor still fails if the API is
    removed, which is the failure worth catching.
    """
    in_tests = call_sites(TESTS)
    total = sum(len(sites) for sites in in_tests.values())

    assert total >= 19, (
        f"document API test coverage dropped to {total} call sites: {in_tests}"
    )
    for name in DOCUMENT_API:
        assert in_tests[name], f"{name} is no longer exercised by any test"


# --------------------------------------------------------------------------
# the scanner itself, so a silently-broken scan cannot pass everything
# --------------------------------------------------------------------------


def test_the_scanner_finds_calls_and_ignores_mentions(tmp_path):
    """A scan that finds nothing must be distinguishable from a clean tree."""
    (tmp_path / "module.py").write_text(
        '"""A docstring naming get_document, which is not a call."""\n'
        "# add_mention in a comment is not a call either\n"
        "NAMES = ['upsert_document']\n"
        "def f(graph):\n"
        "    graph.get_document('d1')\n"
        "    return graph.documents_for_entities(['e'])\n",
        encoding="utf-8",
    )

    found = call_sites(tmp_path)

    assert len(found["get_document"]) == 1
    assert len(found["documents_for_entities"]) == 1
    assert found["upsert_document"] == []
    assert found["add_mention"] == []


@pytest.mark.parametrize("name", sorted(DOCUMENT_API))
def test_every_pinned_name_still_exists_on_the_class(name):
    """The scan is meaningless if it is watching for methods that are gone."""
    source = (SOURCE_ROOT / "graphdb" / "context_graph.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    methods = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert name in methods
