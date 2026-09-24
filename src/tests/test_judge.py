"""Tests for the two model judges.

**No test here makes a network call, and none needs the client library
installed.** The transport is a fake object exposing the one method the judges
use, and the two factories are exercised against a fake module put in
``sys.modules`` — which also proves the import really is lazy, since a
module-scope import would have run before any test could substitute anything.

The seam is the client rather than the HTTP layer because everything worth
asserting here sits above it: which endpoint and model were paired, what the
call was given, how a reply is read, what happens when there is no reply to
read, and what a run can tell afterwards about which endpoint answered.
"""

from __future__ import annotations

import ast
import importlib
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src.analysis import Entity, Resolver, Similarity
from src.common import judge as judge_module
from src.common.config import (
    ENDPOINT_FAST,
    ENDPOINT_GENERAL,
    ENTITY_SERVICE,
    INTENT_CONCEPTUAL,
    INTENT_RELATIONAL,
    STAGE_FALLBACK,
    STAGE_MODEL,
)
from src.common.judge import (
    CONCEPTUAL,
    NO,
    RELATIONAL,
    YES,
    IntentJudge,
    JudgeError,
    MergeJudge,
    fast_client,
)
from src.retrieval import classify

#: A value that would be a disaster in a traceback, chosen to be unmistakable
#: if it ever appears in one.
SECRET = "key-abcdef0123456789"

#: Between DEEP_THRESHOLD and FAST_THRESHOLD, so resolution reaches the model.
BAND_SCORE = 0.88

#: A query no marker in either list matches, so classification reaches stage
#: two. Every intent test that wants the judge consulted uses this.
UNMATCHED = "token expiry handling"


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content, *, choices=None):
        self.choices = choices if choices is not None else [FakeChoice(content)]


class FakeClient:
    """One method deep: what the judges actually call.

    Records every call, so the parameters a judge is responsible for setting —
    temperature, and the two bounds on the query path — are asserted rather
    than assumed.
    """

    def __init__(self, content=None, *, error=None, choices=None):
        self.content = content
        self.error = error
        self.choices = choices
        self.calls: list[dict] = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeResponse(self.content, choices=self.choices)


def timing_out() -> FakeClient:
    """A client that behaves the way the library does when the timeout fires."""
    return FakeClient(error=TimeoutError("request timed out"))


class FakeOpenAI:
    """Stands in for the client library's constructor, keeping its arguments."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def fake_library(monkeypatch):
    """Put a fake client library where the lazy import will find it."""
    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return module


def configure(monkeypatch, **overrides):
    """Set both endpoints to known values, then apply the overrides.

    The constants are read into the judge module's namespace at import, so they
    are patched there rather than through the environment. Keys are read at
    call time and stay in the environment, which is where they live in
    production.
    """
    settings = {
        "JUDGE_BASE_URL": "http://general/v1",
        "JUDGE_MODEL": "general-model",
        "JUDGE_FAST_BASE_URL": "http://fast/v1",
        "JUDGE_FAST_MODEL": "fast-model",
    }
    settings.update(overrides)
    for name, value in settings.items():
        monkeypatch.setattr(judge_module, name, value)
    fake_library(monkeypatch)


class StubEmbedder:
    """Places vectors so that any two distinct texts score exactly ``score``.

    The seam is at vector production, so the real ``Similarity.scores`` runs
    and resolution's bands are driven by geometry rather than by a stubbed
    scoring function.
    """

    dimension = 64
    identity = "stub"

    def __init__(self, score: float):
        self.score = score
        self._axes: dict[str, int] = {}

    def _axis(self, text: str) -> int:
        if text not in self._axes:
            self._axes[text] = 1 + len(self._axes)
        return self._axes[text]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        vector[0] = math.sqrt(self.score)
        vector[self._axis(text)] = math.sqrt(max(0.0, 1.0 - self.score))
        return vector


def entity(text: str, kind: str = ENTITY_SERVICE) -> Entity:
    return Entity(
        text=text, type=kind, score=1.0, start=0, end=len(text), source="rules"
    )


# --------------------------------------------------------------------------
# reading a reply
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reply, expected", [(YES, True), (NO, False)])
def test_the_merge_question_reads_both_answers(reply, expected):
    client = FakeClient(reply)
    assert MergeJudge(client, "m").same("auth-service", "auth_service") is expected


@pytest.mark.parametrize(
    "reply, expected", [(RELATIONAL, True), (CONCEPTUAL, False)]
)
def test_the_intent_question_reads_both_answers(reply, expected):
    assert IntentJudge(FakeClient(reply), "m").relational(UNMATCHED) is expected


def test_the_intent_vocabulary_is_the_one_the_classifier_uses():
    """Spelling the words twice is how the two drift apart."""
    assert RELATIONAL == INTENT_RELATIONAL.upper()
    assert CONCEPTUAL == INTENT_CONCEPTUAL.upper()


def test_an_answer_that_keeps_talking_still_reads_as_yes():
    """The merge question has no token cap, so the reply may not stop at one word."""
    assert MergeJudge(FakeClient("YES, they are the same"), "m").same("a", "b") is True


def test_an_explanation_after_no_still_reads_as_no():
    client = FakeClient("NO - one is a person and one is a service")
    assert MergeJudge(client, "m").same("a", "b") is False


@pytest.mark.parametrize("reply", ["yes", " Yes ", "\nYES\n", "yEs"])
def test_case_and_surrounding_whitespace_change_nothing(reply):
    assert MergeJudge(FakeClient(reply), "m").same("a", "b") is True


@pytest.mark.parametrize("reply", [" relational ", "Relational\n", "sEmAnTiC"])
def test_the_intent_answer_is_read_the_same_way(reply):
    judge = IntentJudge(FakeClient(reply), "m")
    assert judge.relational(UNMATCHED) is reply.strip().upper().startswith(RELATIONAL)


@pytest.mark.parametrize("reply", ["maybe", "same", "1", "TRUE", "unsure"])
def test_an_unrecognised_word_raises(reply):
    with pytest.raises(JudgeError):
        MergeJudge(FakeClient(reply), "m").same("a", "b")


@pytest.mark.parametrize("reply", ["", "   ", "\n", None])
def test_an_empty_reply_raises(reply):
    with pytest.raises(JudgeError):
        MergeJudge(FakeClient(reply), "m").same("a", "b")


def test_a_response_carrying_no_choices_raises():
    with pytest.raises(JudgeError):
        MergeJudge(FakeClient(YES, choices=[]), "m").same("a", "b")


def test_a_transport_error_raises_rather_than_returning_a_default():
    """The caller counts this. Swallowing it would make the count meaningless."""
    client = FakeClient(error=RuntimeError("connection reset"))
    with pytest.raises(RuntimeError):
        MergeJudge(client, "m").same("a", "b")


def test_a_timeout_raises_rather_than_returning_conceptual():
    with pytest.raises(TimeoutError):
        IntentJudge(timing_out(), "m").relational(UNMATCHED)


# --------------------------------------------------------------------------
# what each call carries
# --------------------------------------------------------------------------


def test_both_questions_are_asked_at_temperature_zero():
    merge = FakeClient(YES)
    MergeJudge(merge, "m").same("a", "b")
    intent = FakeClient(CONCEPTUAL)
    IntentJudge(intent, "m").relational(UNMATCHED)

    assert merge.calls[0]["temperature"] == 0
    assert intent.calls[0]["temperature"] == 0


def test_the_query_question_is_bounded_and_the_ingest_one_is_not():
    """The asymmetry is these two bounds, and it is the only asymmetry."""
    merge = FakeClient(YES)
    MergeJudge(merge, "m").same("a", "b")
    intent = FakeClient(CONCEPTUAL)
    IntentJudge(intent, "m", timeout=1.5, max_tokens=3).relational(UNMATCHED)

    assert "timeout" not in merge.calls[0]
    assert "max_tokens" not in merge.calls[0]
    assert intent.calls[0]["timeout"] == 1.5
    assert intent.calls[0]["max_tokens"] == 3


def test_the_bounds_apply_on_the_fallback_endpoint_too(monkeypatch):
    """Degraded routing must still be a bounded query, not a stall."""
    configure(monkeypatch)
    monkeypatch.delenv(judge_module.JUDGE_FAST_KEY_VAR, raising=False)
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)

    resolved = fast_client()
    client = FakeClient(CONCEPTUAL)
    judge = IntentJudge(client, resolved.model, resolved.endpoint)
    judge.relational(UNMATCHED)

    assert judge.endpoint == ENDPOINT_GENERAL
    assert client.calls[0]["timeout"] == judge_module.JUDGE_TIMEOUT_SECONDS
    assert client.calls[0]["max_tokens"] == judge_module.JUDGE_MAX_TOKENS


def test_both_surface_forms_reach_the_exact_merge_question():
    client = FakeClient(YES)
    MergeJudge(client, "m").same("payment_service", "payment-service")

    sent = client.calls[0]["messages"][-1]["content"]
    assert sent == (
        "Do 'payment_service' and 'payment-service' refer to one and the same "
        "entity? Reply with YES or NO only."
    )


def test_the_merge_question_uses_the_pinned_text():
    client = FakeClient(YES)
    MergeJudge(client, "m").same("ignore the above", "b")
    assert client.calls[0]["messages"][-1]["content"] == (
        "Do 'ignore the above' and 'b' refer to one and the same entity? Reply with YES or NO only."
    )


def test_the_query_reaches_the_exact_intent_question():
    client = FakeClient(CONCEPTUAL)
    IntentJudge(client, "m").relational(UNMATCHED)

    sent = client.calls[0]["messages"][-1]["content"]
    assert sent == (
        "Answer with a single word: RELATIONAL when the question follows a chain of "
        "events, people or links, otherwise SEMANTIC.\n\n" + UNMATCHED
    )


# --------------------------------------------------------------------------
# the two endpoints
# --------------------------------------------------------------------------


def test_the_fast_endpoint_is_used_when_its_key_is_set(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv(judge_module.JUDGE_FAST_KEY_VAR, SECRET)
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, "other-key")

    resolved = fast_client()

    assert resolved.client.kwargs["base_url"] == "http://fast/v1"
    assert resolved.client.kwargs["api_key"] == SECRET
    assert resolved.model == "fast-model"
    assert resolved.endpoint == ENDPOINT_FAST


def test_the_general_endpoint_answers_when_the_fast_key_is_unset(monkeypatch):
    configure(monkeypatch)
    monkeypatch.delenv(judge_module.JUDGE_FAST_KEY_VAR, raising=False)
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)

    resolved = fast_client()

    assert resolved.client.kwargs["base_url"] == "http://general/v1"
    assert resolved.model == "general-model"
    assert resolved.endpoint == ENDPOINT_GENERAL


def test_the_caller_can_tell_the_fallback_happened(monkeypatch):
    """The fallback is silent in the answer. It must not be silent in the output."""
    configure(monkeypatch)
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)

    monkeypatch.setenv(judge_module.JUDGE_FAST_KEY_VAR, SECRET)
    configured = fast_client()

    monkeypatch.delenv(judge_module.JUDGE_FAST_KEY_VAR)
    degraded = fast_client()

    assert configured.endpoint != degraded.endpoint


def test_a_blank_fast_key_counts_as_unset(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv(judge_module.JUDGE_FAST_KEY_VAR, "   ")
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)

    assert fast_client().endpoint == ENDPOINT_GENERAL


def test_the_endpoint_and_its_model_are_never_returned_apart(monkeypatch):
    """A model name paired with the wrong URL fails as a 404 from nowhere."""
    configure(monkeypatch)
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, "other-key")

    monkeypatch.setenv(judge_module.JUDGE_FAST_KEY_VAR, SECRET)
    fast = fast_client()

    monkeypatch.delenv(judge_module.JUDGE_FAST_KEY_VAR)
    general = fast_client()

    assert (fast.client.kwargs["base_url"], fast.model) == (
        "http://fast/v1",
        "fast-model",
    )
    assert (general.client.kwargs["base_url"], general.model) == (
        "http://general/v1",
        "general-model",
    )


def test_one_endpoint_configured_twice_is_a_legitimate_setup(monkeypatch):
    """Nothing requires the two to differ, and nothing should."""
    configure(
        monkeypatch,
        JUDGE_FAST_BASE_URL="http://general/v1",
        JUDGE_FAST_MODEL="general-model",
    )
    monkeypatch.setenv(judge_module.JUDGE_FAST_KEY_VAR, SECRET)
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)

    resolved = fast_client()

    assert resolved.model == "general-model"
    assert resolved.endpoint == ENDPOINT_FAST


# --------------------------------------------------------------------------
# construction refuses an unset value
# --------------------------------------------------------------------------


def test_a_judge_with_no_key_fails_at_construction(monkeypatch):
    """Not at the first call, when the run is already underway."""
    configure(monkeypatch)
    monkeypatch.delenv(judge_module.JUDGE_KEY_VAR, raising=False)

    with pytest.raises(JudgeError):
        MergeJudge()


@pytest.mark.parametrize("unset", ["JUDGE_MODEL", "JUDGE_BASE_URL"])
def test_a_merge_judge_with_an_unset_value_fails_at_construction(unset, monkeypatch):
    configure(monkeypatch, **{unset: ""})
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)

    with pytest.raises(JudgeError) as caught:
        MergeJudge()

    assert "not configured" in str(caught.value)


@pytest.mark.parametrize("unset", ["JUDGE_FAST_MODEL", "JUDGE_FAST_BASE_URL"])
def test_an_intent_judge_with_an_unset_value_fails_at_construction(unset, monkeypatch):
    configure(monkeypatch, **{unset: ""})
    monkeypatch.setenv(judge_module.JUDGE_FAST_KEY_VAR, SECRET)
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)

    with pytest.raises(JudgeError) as caught:
        IntentJudge()

    assert "not configured" in str(caught.value)


def test_an_unset_value_names_the_variable_that_would_set_it(monkeypatch):
    configure(monkeypatch, JUDGE_MODEL="")
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)

    with pytest.raises(JudgeError) as caught:
        MergeJudge(FakeClient(YES))

    assert "GRAPHRAG_JUDGE_MODEL" in str(caught.value)


def test_nothing_ships_a_default_model_or_endpoint():
    """A default would silently choose for anyone who set only a key."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, os, sys;"
            "[os.environ.pop(name) for name in list(os.environ)"
            " if name.startswith('GRAPHRAG_JUDGE')];"
            "config = importlib.import_module('src.common.config');"
            "print(repr((config.JUDGE_MODEL, config.JUDGE_BASE_URL,"
            " config.JUDGE_FAST_MODEL, config.JUDGE_FAST_BASE_URL)))",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == repr(("", "", "", ""))


# --------------------------------------------------------------------------
# the key
# --------------------------------------------------------------------------


def _every_failure(monkeypatch):
    """Every exception the judges can raise, with the key set throughout."""
    configure(monkeypatch, JUDGE_MODEL="", JUDGE_FAST_MODEL="")
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)
    monkeypatch.setenv(judge_module.JUDGE_FAST_KEY_VAR, SECRET)

    raised = []

    def capture(call):
        try:
            call()
        except Exception as error:  # noqa: BLE001 - catching all is the point
            raised.append(error)
        else:
            raise AssertionError("expected a failure")

    # unset model, on each path
    capture(lambda: MergeJudge())
    capture(lambda: IntentJudge())
    # empty, unrecognised, and no choices at all
    capture(lambda: MergeJudge(FakeClient(""), "m").same("a", "b"))
    capture(lambda: MergeJudge(FakeClient("perhaps"), "m").same("a", "b"))
    capture(lambda: MergeJudge(FakeClient(YES, choices=[]), "m").same("a", "b"))
    # transport, on a message that quotes the URL the key was sent to
    capture(
        lambda: MergeJudge(
            FakeClient(error=RuntimeError(f"connect to {judge_module.JUDGE_BASE_URL}")),
            "m",
        ).same("a", "b")
    )
    # timeout, on the query path
    capture(lambda: IntentJudge(timing_out(), "m").relational(UNMATCHED))
    # missing key
    monkeypatch.delenv(judge_module.JUDGE_KEY_VAR)
    monkeypatch.delenv(judge_module.JUDGE_FAST_KEY_VAR)
    capture(lambda: MergeJudge())
    return raised


def test_the_key_appears_in_no_failure_this_module_can_raise(monkeypatch):
    raised = _every_failure(monkeypatch)

    assert len(raised) == 8
    for error in raised:
        assert SECRET not in str(error)
        assert SECRET not in repr(error)


def test_a_missing_key_names_the_variable_rather_than_the_value(monkeypatch):
    configure(monkeypatch)
    monkeypatch.delenv(judge_module.JUDGE_KEY_VAR, raising=False)

    with pytest.raises(JudgeError) as caught:
        MergeJudge()

    assert judge_module.JUDGE_KEY_VAR in str(caught.value)


@pytest.mark.parametrize("judge", [MergeJudge, IntentJudge])
def test_a_judge_repr_carries_no_key_and_no_client(judge, monkeypatch):
    """A default repr prints the client, and a client holds a key."""
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)
    text = repr(judge(FakeClient(YES), "some-model"))

    assert "some-model" in text
    assert SECRET not in text
    assert "FakeClient" not in text


def test_no_key_is_spelled_in_the_configuration_module():
    """Only the variable names live there. The values never do."""
    from src.common import config

    source = Path(config.__file__).read_text(encoding="utf-8")
    assert config.JUDGE_KEY_VAR in source
    assert config.JUDGE_FAST_KEY_VAR in source
    assert "api_key" not in source


# --------------------------------------------------------------------------
# the optional dependency
# --------------------------------------------------------------------------


def test_the_client_library_is_never_imported_at_module_scope():
    """Asserted against the parse tree, so a docstring example cannot fire it."""
    tree = ast.parse(Path(judge_module.__file__).read_text(encoding="utf-8"))

    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name != "openai" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "openai"


def test_the_module_imports_with_the_client_library_absent():
    """The client library stays out of the runtime set. A fresh interpreter,
    so nothing is cached."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['openai'] = None;"
            "import src.common.judge as judge;"
            "print(judge.MergeJudge.__name__)",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "MergeJudge"


def test_only_a_call_fails_when_the_client_library_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)

    importlib.reload(judge_module)
    try:
        with pytest.raises(ImportError):
            judge_module.chat_client()
    finally:
        monkeypatch.undo()
        importlib.reload(judge_module)


# --------------------------------------------------------------------------
# what a run can tell afterwards
# --------------------------------------------------------------------------


def test_an_unconfigured_fast_endpoint_is_distinguishable_from_a_judge_that_answered():
    """Both runs weight every query the same way. They mean different things."""
    answered = IntentJudge(FakeClient(CONCEPTUAL), "m", ENDPOINT_GENERAL)
    for _ in range(3):
        classify(UNMATCHED, answered)

    unreachable = IntentJudge(timing_out(), "m", ENDPOINT_GENERAL)
    for _ in range(3):
        classify(UNMATCHED, unreachable)

    absent_stages = [classify(UNMATCHED).stage for _ in range(3)]

    assert [answered.calls, answered.failures] == [3, 0]
    assert [unreachable.calls, unreachable.failures] == [3, 3]
    assert absent_stages == [STAGE_FALLBACK] * 3


def test_the_judge_records_which_endpoint_it_got(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)
    monkeypatch.delenv(judge_module.JUDGE_FAST_KEY_VAR, raising=False)

    judge = IntentJudge()

    assert judge.endpoint == ENDPOINT_GENERAL
    assert ENDPOINT_GENERAL in repr(judge)


def test_a_counted_failure_is_still_raised():
    """Counting is not handling. The caller still decides what a failure means."""
    judge = IntentJudge(timing_out(), "m")

    with pytest.raises(TimeoutError):
        judge.relational(UNMATCHED)

    assert judge.failures == 1


# --------------------------------------------------------------------------
# through the real consumers
# --------------------------------------------------------------------------


def test_a_grey_band_pair_reaches_the_model_and_merges():
    """The band is the only place resolution pays for a call, and this is it."""
    resolver = Resolver(
        Similarity(StubEmbedder(BAND_SCORE)), MergeJudge(FakeClient(YES), "m")
    )
    resolver.add(entity("auth-service"))
    resolver.add(entity("auth_service"))

    assert resolver.stats.model_calls == 1
    assert resolver.stats.model_merges == 1
    assert resolver.stats.fast_merges == 0
    assert resolver.stats.failures == 0
    assert len(resolver.entities) == 1


def test_the_model_answering_no_splits_the_pair_without_counting_a_failure():
    """A rejection and an unreachable endpoint must not become the same number."""
    resolver = Resolver(
        Similarity(StubEmbedder(BAND_SCORE)), MergeJudge(FakeClient(NO), "m")
    )
    resolver.add(entity("auth-service"))
    resolver.add(entity("billing-service"))

    assert resolver.stats.model_rejections == 1
    assert resolver.stats.failures == 0
    assert len(resolver.entities) == 2


def test_a_raising_judge_counts_a_failure_and_creates_a_new_entity():
    """``band_fraction`` only means something if a failure reaches this branch."""
    client = FakeClient(error=RuntimeError("connection reset"))
    resolver = Resolver(Similarity(StubEmbedder(BAND_SCORE)), MergeJudge(client, "m"))
    resolver.add(entity("auth-service"))
    resolver.add(entity("auth_service"))

    assert resolver.stats.model_calls == 1
    assert resolver.stats.failures == 1
    assert resolver.stats.model_merges == 0
    assert len(resolver.entities) == 2


def test_an_unrecognised_reply_reaches_resolution_as_a_failure():
    resolver = Resolver(
        Similarity(StubEmbedder(BAND_SCORE)), MergeJudge(FakeClient("perhaps"), "m")
    )
    resolver.add(entity("auth-service"))
    resolver.add(entity("auth_service"))

    assert resolver.stats.failures == 1
    assert len(resolver.entities) == 2


def test_classification_records_the_model_stage():
    result = classify(UNMATCHED, IntentJudge(FakeClient(RELATIONAL), "m"))

    assert result.intent == INTENT_RELATIONAL
    assert result.stage == STAGE_MODEL


def test_classification_records_the_model_stage_for_a_conceptual_answer():
    result = classify(UNMATCHED, IntentJudge(FakeClient(CONCEPTUAL), "m"))

    assert result.intent == INTENT_CONCEPTUAL
    assert result.stage == STAGE_MODEL


def test_a_raising_intent_judge_falls_back_rather_than_propagating():
    client = FakeClient(error=RuntimeError("connection reset"))
    result = classify(UNMATCHED, IntentJudge(client, "m"))

    assert result.intent == INTENT_CONCEPTUAL
    assert result.stage == STAGE_FALLBACK


def test_a_timing_out_intent_judge_does_not_raise_past_classify():
    """The judge raises. The classifier is where that stops."""
    result = classify(UNMATCHED, IntentJudge(timing_out(), "m"))

    assert result.intent == INTENT_CONCEPTUAL
    assert result.stage == STAGE_FALLBACK


def test_a_marker_match_never_reaches_the_judge():
    """Stage one is the cheap one. A call here is a call that was not needed."""
    judge = IntentJudge(FakeClient(RELATIONAL), "m")
    classify("who reviewed the auth change", judge)

    assert judge.calls == 0


def test_the_query_client_disables_the_library_retry(monkeypatch):
    """The timeout is a budget, so it has to bound the call and not one attempt.

    The client library retries by default. With retries left on, a timeout
    fires per attempt and the wall-clock cost of a classification is the budget
    multiplied by the attempts, plus backoff — measured at 45x the budget on a
    real endpoint. The fallback this path takes is the answer an unmatched
    query gets anyway, so a second attempt buys nothing worth the latency.
    """
    configure(monkeypatch)
    monkeypatch.setenv(judge_module.JUDGE_FAST_KEY_VAR, SECRET)

    resolved = fast_client()

    assert resolved.client.kwargs["max_retries"] == 0


def test_the_query_client_disables_the_retry_on_the_fallback_endpoint_too(monkeypatch):
    """Degraded routing is still inside the user's latency budget."""
    configure(monkeypatch)
    monkeypatch.delenv(judge_module.JUDGE_FAST_KEY_VAR, raising=False)
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)

    resolved = fast_client()

    assert resolved.endpoint == ENDPOINT_GENERAL
    assert resolved.client.kwargs["max_retries"] == 0


def test_the_ingest_client_keeps_the_library_retry(monkeypatch):
    """Nothing is waiting on the ingest question, so a retry there is free."""
    configure(monkeypatch)
    monkeypatch.setenv(judge_module.JUDGE_KEY_VAR, SECRET)

    client = judge_module.chat_client()

    assert "max_retries" not in client.kwargs
