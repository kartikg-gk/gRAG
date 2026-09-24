from __future__ import annotations

import pytest

from src.analysis import Extractor, SOURCE_NONE, windows
from src.common import config
from src.ingestion.models import PullRequest
from src.knowledge.documents import source_documents


def word_text(count: int) -> str:
    return " ".join(f"word{index}" for index in range(count))


def pull_request(body: str, number: int = 1) -> PullRequest:
    return PullRequest.model_validate(
        {
            "id": 90_000 + number,
            "number": number,
            "title": "a title",
            "state": "open",
            "draft": False,
            "body": body,
            "html_url": f"https://example.invalid/pull/{number}",
            "user": {"login": "alice", "id": 1},
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )


def documents(body: str):
    return source_documents(pull_requests=[pull_request(body)])


def test_extraction_defaults_are_300_words_with_50_words_overlap():
    assert config.WINDOW_WORDS == 300
    assert config.WINDOW_OVERLAP_WORDS == 50


def test_short_extraction_input_is_one_exact_absolute_window():
    text = "  " + word_text(299) + "\n"
    produced = list(windows(text))

    assert len(produced) == 1
    assert produced[0].text == word_text(299)
    assert text[produced[0].start : produced[0].end] == produced[0].text


def test_extraction_windows_advance_250_words_and_share_exactly_50():
    text = word_text(551)
    produced = list(windows(text))
    first = produced[0].text.split()
    second = produced[1].text.split()

    assert second[0] == "word250"
    assert first[-50:] == second[:50]
    assert all(text[item.start : item.end] == item.text for item in produced)


def test_extraction_window_validation_is_unchanged():
    with pytest.raises(ValueError, match="at least 1 word"):
        list(windows("alpha", size=0, overlap=0))
    with pytest.raises(ValueError, match="negative"):
        list(windows("alpha", size=2, overlap=-1))
    with pytest.raises(ValueError, match="smaller than the window"):
        list(windows("alpha", size=2, overlap=2))


def test_overlap_duplicate_mentions_are_still_deduplicated():
    class CountingBackend:
        name = "counting"

        def __init__(self):
            self.matches = 0

        def entities(self, text):
            start = text.find("Acme Corp")
            if start >= 0:
                self.matches += 1
                yield "Acme Corp", "ORG", start, start + len("Acme Corp")

    text = f"{word_text(260)} Acme Corp {word_text(100)}"
    backend = CountingBackend()
    extractor = Extractor(SOURCE_NONE)
    extractor.backend = backend

    found = extractor.extract(text)

    assert backend.matches == 2
    assert len(found) == 1
    assert found[0].text == "Acme Corp"
    assert text[found[0].start : found[0].end] == "Acme Corp"


def test_document_defaults_are_character_based_1200_and_150():
    assert config.DOCUMENT_CHUNK_CHARACTERS == 1200
    assert config.DOCUMENT_CHUNK_OVERLAP_CHARACTERS == 150


def test_1200_characters_is_one_chunk_and_1201_is_two():
    assert [item.content for item in documents("x" * 1200)] == ["x" * 1200]
    split = documents("x" * 1201)
    assert len(split) == 2
    assert split[0].content == "x" * 1200
    assert split[1].content == "x" * 151


def test_full_document_chunks_advance_1050_and_overlap_original_150():
    body = "".join(chr(33 + index % 90) for index in range(2250))
    split = documents(body)

    assert len(split) == 2
    assert split[0].content == body[:1200]
    assert split[1].content == body[1050:2250]
    assert split[0].content[-150:] == split[1].content[:150] == body[1050:1200]


def test_document_chunks_are_bounded_and_reconstruct_exactly():
    body = "".join(chr(33 + index % 90) for index in range(5003))
    split = documents(body)

    assert all(len(item.content) <= 1200 for item in split)
    reconstructed = split[0].content + "".join(
        item.content[150:] for item in split[1:]
    )
    assert reconstructed == body


def test_document_chunking_counts_unicode_characters_not_bytes():
    split = documents("é" * 1201)

    assert len(split) == 2
    assert [len(item.content) for item in split] == [1200, 151]


def test_document_chunks_preserve_whitespace_and_exclude_blank_chunks():
    body = "start  \n\t middle   end"
    assert documents(body)[0].content == body

    split = documents("x" + " " * 2500)
    assert [item.id for item in split] == ["doc:pr:1:0"]
    assert split[0].content == ("x" + " " * 1199)


def test_document_chunk_ids_and_metadata_remain_deterministic():
    body = "x" * 1201
    first = documents(body)
    second = documents(body)

    assert first == second
    assert [item.id for item in first] == ["doc:pr:1:0", "doc:pr:1:1"]
    assert all(item.path == "https://example.invalid/pull/1" for item in first)
    assert all(item.origin == "pr:1" for item in first)
    assert all(item.field == "pull_request_body" for item in first)
