"""Tests for entity extraction.

A fake backend stands in for spaCy almost everywhere. That is not a
convenience: spaCy is an optional dependency, so a suite that needed it would
either be unrunnable on a clean checkout or would quietly skip the behaviour it
was written to protect. The one test that does exercise the real model skips
explicitly and says so.

The fake also does what the real model cannot be made to do on demand: return a
chosen span with a chosen score, so window-boundary handling and score-based
deduplication can be asserted exactly instead of approximately.
"""

from __future__ import annotations

import pytest

from src.analysis import (
    DEFAULT_BACKEND,
    Entity,
    Extractor,
    LABEL_MAP,
    NullBackend,
    SOURCE_NONE,
    SOURCE_RULES,
    clean,
    dedupe,
    load_backend,
)
from src.common.config import (
    ENTITY_ORG,
    ENTITY_PERSON,
    ENTITY_PR,
    ENTITY_SERVICE,
    ENTITY_TICKET,
    MIN_ENTITY_LENGTH,
    RULE_SCORE,
    STATISTICAL_SCORE,
)


class FakeBackend:
    """Reports whatever it is told to, at whatever score.

    Offsets given to it are relative to the text it receives — the same
    contract spaCy has — so a test can hand the extractor a phrase and check
    that the offset coming out is absolute.
    """

    name = "fake"

    def __init__(self, findings=(), *, score_by_text=None):
        self._findings = list(findings)
        self._score_by_text = score_by_text or {}

    def entities(self, text: str):
        for surface, label in self._findings:
            start = text.find(surface)
            if start != -1:
                yield surface, label, start, start + len(surface)


def extractor_with(backend, **kwargs) -> Extractor:
    """An extractor with the statistical stage replaced."""
    made = Extractor(SOURCE_NONE, **kwargs)
    made.backend = backend
    return made


# --------------------------------------------------------------------------
# the entity contract
# --------------------------------------------------------------------------


def test_an_entity_can_be_found_at_its_own_offsets():
    """document[start:end] == text. The contract the offsets exist for."""
    text = "The fix for #412 landed in payment_service last week."

    for entity in Extractor(SOURCE_NONE).extract(text):
        assert text[entity.start : entity.end] == entity.text


def test_offsets_are_exclusive_at_the_end():
    text = "see #412 now"

    entity = Extractor(SOURCE_NONE).extract(text)[0]

    assert (entity.start, entity.end) == (4, 8)
    assert text[entity.start : entity.end] == "#412"


def test_every_entity_carries_all_six_fields():
    entity = Extractor(SOURCE_NONE).extract("fixes #412")[0]

    assert entity.text and entity.type and entity.source
    assert 0.0 <= entity.score <= 1.0
    assert entity.start < entity.end


def test_extracting_from_empty_text_returns_nothing():
    assert Extractor(SOURCE_NONE).extract("") == []


# --------------------------------------------------------------------------
# stage one: rules, not statistics
# --------------------------------------------------------------------------


def test_a_ticket_reference_is_labelled_by_the_rule_stage():
    entity = Extractor(SOURCE_NONE).extract("Reopened #412 this morning.")[0]

    assert entity.type == ENTITY_TICKET
    assert entity.source == SOURCE_RULES
    assert entity.score == RULE_SCORE


def test_a_pull_request_reference_is_labelled_by_the_rule_stage():
    entity = Extractor(SOURCE_NONE).extract("Superseded by PR #1347 yesterday.")[0]

    assert entity.type == ENTITY_PR
    assert entity.source == SOURCE_RULES
    assert entity.score == RULE_SCORE


def test_a_service_name_is_labelled_by_the_rule_stage():
    entity = Extractor(SOURCE_NONE).extract("Deployed payment_service to staging.")[0]

    assert entity.type == ENTITY_SERVICE
    assert entity.source == SOURCE_RULES


@pytest.mark.parametrize(
    "text,expected",
    [
        ("PR #1347", ENTITY_PR),
        ("pull request #1347", ENTITY_PR),
        ("pull-request #1347", ENTITY_PR),
        ("pr 1347", ENTITY_PR),
        ("#42", ENTITY_TICKET),
        ("issue #42", ENTITY_TICKET),
        ("payment_service", ENTITY_SERVICE),
        ("auth-service", ENTITY_SERVICE),
        ("billing_svc", ENTITY_SERVICE),
    ],
)
def test_the_known_formats_are_recognised(text, expected):
    entities = Extractor(SOURCE_NONE).extract(f"context {text} context")

    assert [entity.type for entity in entities] == [expected]


def test_the_rule_stage_needs_no_backend_at_all():
    """The whole reason spaCy can stay optional."""
    extractor = Extractor(SOURCE_NONE)

    assert isinstance(extractor.backend, NullBackend)
    assert extractor.extract("fixes #412")[0].type == ENTITY_TICKET


def test_a_rule_match_beats_an_overlapping_statistical_match():
    """The precise stage wins the span, which is why it runs first."""
    backend = FakeBackend([("payment_service", "ORG")])
    extractor = extractor_with(backend)

    entities = extractor.extract("We deployed payment_service today.")

    assert [(e.type, e.source) for e in entities] == [(ENTITY_SERVICE, SOURCE_RULES)]


@pytest.mark.parametrize(
    "phrase", ["self-service", "customer-service", "microservice", "in-service"]
)
def test_an_english_compound_is_not_a_service(phrase):
    """Found by reading the output, not by a failing test. See examples/entity_demo.py."""
    assert Extractor(SOURCE_NONE).extract(f"the {phrase} portal") == []


def test_a_multi_word_service_name_is_not_mistaken_for_a_compound():
    """customer-service is prose; customer_billing_service is a real name."""
    entities = Extractor(SOURCE_NONE).extract("calls customer_billing_service now")

    assert [entity.text for entity in entities] == ["customer_billing_service"]


def test_a_pull_request_reference_is_not_also_read_as_a_ticket():
    entities = Extractor(SOURCE_NONE).extract("See PR #1347 for the fix.")

    assert [entity.type for entity in entities] == [ENTITY_PR]


# --------------------------------------------------------------------------
# stage two: the statistical backend, and its one translation table
# --------------------------------------------------------------------------


def test_a_statistical_finding_is_kept_with_the_backend_as_its_source():
    extractor = extractor_with(FakeBackend([("Alice Mbeki", "PERSON")]))

    entities = extractor.extract("Reviewed by Alice Mbeki on Tuesday.")

    assert [(e.text, e.type, e.source) for e in entities] == [
        ("Alice Mbeki", ENTITY_PERSON, "fake")
    ]
    assert entities[0].score == STATISTICAL_SCORE


def test_labels_are_translated_through_the_mapping_table():
    extractor = extractor_with(FakeBackend([("Acme Corp", "ORG")]))

    assert extractor.extract("Acme Corp ships it.")[0].type == ENTITY_ORG


def test_an_unmapped_label_is_dropped():
    """spaCy emits DATE, CARDINAL and more that say nothing about a repo."""
    extractor = extractor_with(FakeBackend([("Tuesday", "DATE")]))

    assert extractor.extract("It shipped Tuesday.") == []


def test_the_mapping_table_is_the_only_translation():
    """Every label the code accepts is reachable from the table alone."""
    assert set(LABEL_MAP.values()) == {ENTITY_PERSON, ENTITY_ORG, "Product"}


# --------------------------------------------------------------------------
# window boundaries
# --------------------------------------------------------------------------


def test_an_entity_spanning_a_window_boundary_is_found_once_at_absolute_offsets():
    """The reason overlap exists, asserted end to end."""
    filler = " ".join(f"word{index}" for index in range(18))
    text = f"{filler} Alice Mbeki {filler}"
    expected_start = text.index("Alice Mbeki")

    # Windows of 10 words stepping 5: "Alice Mbeki" straddles a boundary and is
    # whole only in a window that exists because of the overlap.
    extractor = extractor_with(
        FakeBackend([("Alice Mbeki", "PERSON")]), window=10, overlap=5
    )
    entities = extractor.extract(text)

    assert len(entities) == 1
    assert entities[0].text == "Alice Mbeki"
    assert entities[0].start == expected_start
    assert text[entities[0].start : entities[0].end] == "Alice Mbeki"


def test_an_entity_seen_in_two_windows_is_reported_once():
    filler = " ".join(f"word{index}" for index in range(30))
    text = f"{filler} Acme Corp {filler}"

    extractor = extractor_with(FakeBackend([("Acme Corp", "ORG")]), window=10, overlap=5)

    assert len(extractor.extract(text)) == 1


def test_a_bad_window_geometry_is_refused_at_construction():
    """Not on whichever document happens to be long enough to need a window."""
    with pytest.raises(ValueError, match="smaller than the window"):
        Extractor(SOURCE_NONE, window=10, overlap=10)


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  payment_service  ", "payment_service"),
        ("* payment_service", "payment_service"),
        ('"payment_service"', "payment_service"),
        ("payment_service,", "payment_service"),
        ("(payment_service)", "payment_service"),
        ("- payment_service.", "payment_service"),
        ("“payment_service”", "payment_service"),
        ("• payment_service", "payment_service"),
    ],
)
def test_junk_is_trimmed_from_both_ends(raw, expected):
    cleaned = clean(raw, 0, len(raw))

    assert cleaned is not None
    assert cleaned[0] == expected


def test_cleaning_keeps_the_offsets_pointing_at_the_cleaned_text():
    document = 'the "payment_service" broke'
    raw_start = document.index('"payment_service"')
    raw = document[raw_start : raw_start + len('"payment_service"')]

    surface, start, end = clean(raw, raw_start, raw_start + len(raw))

    assert document[start:end] == surface == "payment_service"


def test_internal_punctuation_is_never_touched():
    assert clean("payment_service", 0, 15)[0] == "payment_service"
    assert clean("auth-service", 0, 12)[0] == "auth-service"
    assert clean("#412", 0, 4)[0] == "#412"


def test_an_entity_below_the_minimum_length_is_rejected():
    assert clean("ab", 0, 2) is None


def test_an_entity_at_the_minimum_length_is_kept():
    surface = "a" * MIN_ENTITY_LENGTH

    assert clean(surface, 0, len(surface))[0] == surface


def test_a_form_that_is_all_junk_is_rejected_rather_than_emptied():
    """None, not "". A falsy string still gets inserted by a careless caller."""
    assert clean("***", 0, 3) is None
    assert clean("   ", 0, 3) is None


def test_a_statistical_finding_that_cleans_to_nothing_is_dropped():
    extractor = extractor_with(FakeBackend([("--", "ORG")]))

    assert extractor.extract("Acme -- Corp") == []


# --------------------------------------------------------------------------
# deduplication
# --------------------------------------------------------------------------


def test_the_same_entity_twice_keeps_the_higher_score():
    entities = [
        Entity("Acme", ENTITY_ORG, 0.4, 0, 4, "fake"),
        Entity("Acme", ENTITY_ORG, 0.9, 50, 54, "fake"),
    ]

    assert [entity.score for entity in dedupe(entities)] == [0.9]


def test_the_same_surface_form_with_a_different_type_is_kept_separately():
    entities = [
        Entity("Acme", ENTITY_ORG, 0.4, 0, 4, "fake"),
        Entity("Acme", ENTITY_PERSON, 0.4, 0, 4, "fake"),
    ]

    assert len(dedupe(entities)) == 2


def test_a_tie_keeps_the_earlier_occurrence():
    """So the output is stable across runs rather than iteration-order luck."""
    entities = [
        Entity("Acme", ENTITY_ORG, 0.5, 90, 94, "fake"),
        Entity("Acme", ENTITY_ORG, 0.5, 10, 14, "fake"),
    ]

    assert dedupe(entities)[0].start == 10


def test_results_come_back_in_document_order():
    text = "payment_service broke, see #412, then PR #99 fixed it"

    starts = [entity.start for entity in Extractor(SOURCE_NONE).extract(text)]

    assert starts == sorted(starts)


# --------------------------------------------------------------------------
# the pluggable backend
# --------------------------------------------------------------------------


def test_an_unavailable_backend_degrades_instead_of_raising():
    assert load_backend("no-such-backend") is not None


def test_asking_for_no_backend_gets_the_null_one():
    assert isinstance(load_backend(SOURCE_NONE), NullBackend)


def test_the_null_backend_records_itself_as_the_source():
    """"Rules only" and "the model found nothing" must not look alike."""
    assert NullBackend().name == SOURCE_NONE


def test_swapping_the_backend_changes_no_caller():
    text = "Reviewed by Alice Mbeki."

    without = Extractor(SOURCE_NONE).extract(text)
    with_fake = extractor_with(FakeBackend([("Alice Mbeki", "PERSON")])).extract(text)

    assert without == []
    assert [entity.text for entity in with_fake] == ["Alice Mbeki"]


def test_the_default_backend_preference_is_spacy():
    assert DEFAULT_BACKEND == "spacy"


def test_the_real_spacy_backend_extracts_a_person():
    """The one test that needs the optional dependency. Skips without it."""
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm is not downloaded")

    entities = Extractor("spacy").extract("Barack Obama reviewed the change.")

    assert any(entity.type == ENTITY_PERSON for entity in entities)
    assert all(entity.source == "spacy" for entity in entities if entity.source != SOURCE_RULES)
