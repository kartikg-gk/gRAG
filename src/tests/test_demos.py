"""Every example script runs, and says something.

``resolve_demo`` broke twice while the whole suite stayed green, both times
caught only by running it by hand. This file is the control that was missing.

Running to completion is the floor, not the bar. A demo that runs and reports
nothing useful is a failure this project has hit twice — a trace reporting
"used 3 of 3", and a template answer that quoted its own input — and a smoke
test asserting only "did not crash" would have passed both. So each demo here
is checked for the property that makes it worth having:

* the extraction demo must find entities, of more than one type
* the resolution demo must both create and merge — a run that merged nothing
  and a run that merged everything are the two ways it can be useless
* the retrieval demos must produce a used set that is neither empty nor the
  whole retrieved set

Output is captured and asserted rather than only produced, because "it printed
something" and "it printed the right number" are different claims.
"""

from __future__ import annotations

import json
import re

import pytest

from examples import entity_demo, offline_demo, resolve_demo
from src.analysis import Extractor, Resolver
from src.tracing import is_used, load


def numbers_after(output: str, label: str) -> list[int]:
    """Every integer on the line introduced by ``label``."""
    for line in output.splitlines():
        if line.strip().startswith(label):
            return [int(n) for n in re.findall(r"\d+", line)]
    raise AssertionError(f"no line starting {label!r} in:\n{output}")


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def test_the_entity_demo_runs_to_completion(capsys):
    assert entity_demo.main([]) == 0
    assert capsys.readouterr().out


def test_the_entity_demo_actually_extracts_entities(capsys):
    entity_demo.main([])

    found = numbers_after(capsys.readouterr().out, "23 entities")
    assert found[0] > 0


def test_the_entity_demo_finds_more_than_one_type():
    """A run that only ever found tickets would be a broken rule stage."""
    extractor = Extractor("none")
    types = {
        entity.type
        for document in entity_demo.DOCUMENTS
        for entity in extractor.extract(document)
    }

    assert len(types) >= 3


def test_the_entity_demo_reports_the_backend_it_used(capsys):
    """"Rules only" and "the model found nothing" must not look alike."""
    entity_demo.main([])

    assert "backend in use:" in capsys.readouterr().out


def test_the_entity_demo_corpus_still_contains_the_hard_cases():
    """The adversarial documents are the point. Losing them guts the check."""
    corpus = " ".join(entity_demo.DOCUMENTS)

    assert "self-service" in corpus
    assert "microservice" in corpus
    assert "acme/repo#118" in corpus


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def test_the_resolve_demo_runs_to_completion(capsys):
    assert resolve_demo.main([]) == 0
    assert capsys.readouterr().out


def test_the_resolve_demo_both_creates_and_merges(capsys):
    """The two ways it can be useless: merge nothing, or merge everything."""
    resolve_demo.main([])
    output = capsys.readouterr().out

    created = numbers_after(output, "created")[0]
    merges = numbers_after(output, "variant merges")[0]

    assert created > 0
    assert merges > 0


def test_the_resolve_demo_reduces_the_entity_count(capsys):
    """Resolution that leaves the count unchanged did nothing."""
    resolve_demo.main([])
    output = capsys.readouterr().out

    seen = numbers_after(output, "entities seen")[0]
    canonical = numbers_after(output, "canonical entities")[0]

    assert 0 < canonical < seen


def test_the_resolve_demo_reports_no_cross_type_merge(capsys):
    resolve_demo.main([])

    assert numbers_after(capsys.readouterr().out, "cross-type merges")[0] == 0


def test_the_resolve_demo_query_probes_all_pass(capsys):
    """Every probe is marked ok, not just printed."""
    resolve_demo.main([])
    output = capsys.readouterr().out

    probe_lines = [l for l in output.splitlines() if l.startswith("  ok ") or l.startswith("  !! ")]
    assert probe_lines
    assert all(line.startswith("  ok ") for line in probe_lines)


def test_the_resolve_demo_entity_set_is_order_independent(capsys):
    """The property the demo claims. Asserted, not read off the screen."""
    resolve_demo.main([])
    output = capsys.readouterr().out

    # The section heading also starts with "order", so match the numbered rows.
    order_lines = [
        l for l in output.splitlines() if re.match(r"\s+order \d+:", l)
    ]
    assert len(order_lines) == 3
    assert all("set identical" in line for line in order_lines)


def test_the_resolve_demo_reports_a_score_distribution(capsys):
    """The evidence behind the band fraction, without which it means nothing."""
    resolve_demo.main([])
    output = capsys.readouterr().out

    assert "score distribution" in output
    assert numbers_after(output, ">= 0.92 fast")[-1] > 0


def test_the_resolve_demo_corpus_still_contains_real_drift():
    """Three known variant pairs. Without them the demo proves nothing."""
    extractor = Extractor("none")
    surfaces = {
        entity.text
        for document in entity_demo.DOCUMENTS
        for entity in extractor.extract(document)
    }

    for left, right in [
        ("notification-service", "notification_service"),
        ("PR #1290", "PR#1290"),
        ("order_service", "ORDER_SERVICE"),
    ]:
        assert left in surfaces and right in surfaces


# --------------------------------------------------------------------------
# retrieval and answering
# --------------------------------------------------------------------------


def test_the_offline_demo_runs_to_completion(capsys, tmp_path):
    assert offline_demo.main(["--output", str(tmp_path / "trace.json")]) == 0
    assert capsys.readouterr().out


def test_the_offline_demo_used_set_is_neither_empty_nor_everything(tmp_path):
    """The failure this assertion exists for is a trace reading "used 3 of 3"."""
    path = tmp_path / "trace.json"
    offline_demo.main(["--output", str(path)])

    trace = load(path)
    used = [item for item in trace.items if is_used(item.overlap)]

    assert trace.items, "the demo retrieved nothing"
    assert 0 < len(used) < len(trace.items)


def test_the_offline_demo_used_set_is_not_the_top_ranked_items(tmp_path):
    """Otherwise overlap is just reproducing the ranking and measures nothing."""
    path = tmp_path / "trace.json"
    offline_demo.main(["--output", str(path)])

    trace = load(path)
    used = [index for index, item in enumerate(trace.items) if is_used(item.overlap)]

    assert used != list(range(len(used)))


def test_the_offline_demo_answer_is_not_empty(tmp_path):
    path = tmp_path / "trace.json"
    offline_demo.main(["--output", str(path)])

    assert load(path).answer


def test_the_offline_demo_writes_a_readable_trace(tmp_path):
    path = tmp_path / "trace.json"
    offline_demo.main(["--output", str(path)])

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 3
    assert payload["retrievals"]


# --------------------------------------------------------------------------
# the graph demo needs langgraph, which is an optional group
# --------------------------------------------------------------------------


def test_the_graph_demo_runs_to_completion(capsys, tmp_path):
    pytest.importorskip("langgraph")
    from examples import graph_demo

    assert graph_demo.main(["--output", str(tmp_path / "trace.json")]) == 0
    assert capsys.readouterr().out


def test_the_graph_demo_records_spans_for_its_nodes(tmp_path):
    """A traced graph that recorded no spans traced nothing."""
    pytest.importorskip("langgraph")
    from examples import graph_demo

    path = tmp_path / "trace.json"
    graph_demo.main(["--output", str(path)])

    trace = load(path)
    assert trace.spans
    assert trace.items
