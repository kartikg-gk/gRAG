"""Tests for the sliding window.

The property that matters is that a window never lies about where it came
from. Most of these tests check that by slicing the original text with the
offsets a window reports and comparing — an assertion the code cannot satisfy
by accident.
"""

from __future__ import annotations

import pytest

from src.analysis import Window, windows
from src.common.config import WINDOW_OVERLAP_WORDS, WINDOW_WORDS


def words(count: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


# --------------------------------------------------------------------------
# offsets are absolute and true
# --------------------------------------------------------------------------


def test_a_window_slices_the_text_it_reports():
    text = words(500)

    for window in windows(text, size=100, overlap=20):
        assert text[window.start : window.end] == window.text


def test_the_first_window_starts_at_the_first_word_not_at_zero():
    """Leading whitespace is not part of any window."""
    text = "   alpha bravo charlie"

    first = next(iter(windows(text, size=2, overlap=1)))

    assert first.start == 3
    assert first.text == "alpha bravo"


def test_later_windows_carry_a_nonzero_offset():
    text = words(300)

    offsets = [window.start for window in windows(text, size=100, overlap=20)]

    assert offsets[0] == 0
    assert all(offset > 0 for offset in offsets[1:])


def test_absolute_translates_a_position_inside_the_window():
    window = Window(text="bravo charlie", start=6, end=19)

    assert window.absolute(0, 5) == (6, 11)


def test_every_window_covers_whole_words():
    text = words(120)

    for window in windows(text, size=25, overlap=5):
        assert not window.text.startswith(" ")
        assert not window.text.endswith(" ")
        assert window.text == window.text.strip()


# --------------------------------------------------------------------------
# overlap
# --------------------------------------------------------------------------


def test_consecutive_windows_share_the_configured_overlap():
    text = words(60)

    produced = list(windows(text, size=20, overlap=5))

    first_words = produced[0].text.split()
    second_words = produced[1].text.split()
    assert first_words[-5:] == second_words[:5]


def test_every_word_appears_in_at_least_one_window():
    text = words(97)

    seen = {word for window in windows(text, size=20, overlap=5) for word in window.text.split()}

    assert seen == set(text.split())


def test_an_overlap_equal_to_the_window_is_refused():
    with pytest.raises(ValueError, match="smaller than the window"):
        list(windows("alpha bravo", size=10, overlap=10))


def test_an_overlap_larger_than_the_window_is_refused():
    with pytest.raises(ValueError, match="smaller than the window"):
        list(windows("alpha bravo", size=10, overlap=11))


def test_a_negative_overlap_is_refused():
    with pytest.raises(ValueError, match="negative"):
        list(windows("alpha bravo", size=10, overlap=-1))


def test_a_window_smaller_than_one_word_is_refused():
    with pytest.raises(ValueError, match="at least 1 word"):
        list(windows("alpha bravo", size=0, overlap=0))


def test_zero_overlap_is_allowed():
    produced = list(windows(words(10), size=5, overlap=0))

    assert len(produced) == 2


# --------------------------------------------------------------------------
# edges of the input
# --------------------------------------------------------------------------


def test_text_shorter_than_one_window_yields_one_window():
    produced = list(windows("alpha bravo charlie", size=100, overlap=10))

    assert len(produced) == 1
    assert produced[0].text == "alpha bravo charlie"


def test_empty_text_yields_nothing():
    assert list(windows("", size=10, overlap=2)) == []


def test_whitespace_only_text_yields_nothing():
    assert list(windows("   \n\t  ", size=10, overlap=2)) == []


def test_the_tail_is_not_emitted_twice():
    """A window that reached the last word ends the run."""
    produced = list(windows(words(45), size=20, overlap=5))

    assert len({(window.start, window.end) for window in produced}) == len(produced)
    assert produced[-1].text.endswith("word44")


def test_the_configured_defaults_are_a_valid_geometry():
    """The shipped configuration must not be one that raises."""
    assert WINDOW_OVERLAP_WORDS < WINDOW_WORDS
    assert list(windows(words(5)))


def test_no_window_is_contained_in_another():
    """A window whose span sits inside another's is pure duplicate work.

    Found by mutation testing: deleting the guard that stops after the window
    reaching the final word passed every other test in this file, because the
    input sizes they used happened not to produce a redundant tail.
    """
    for count in (45, 46, 50, 52, 61, 77):
        produced = list(windows(words(count), size=20, overlap=5))
        spans = [(w.start, w.end) for w in produced]

        for index, (start, end) in enumerate(spans):
            for other_start, other_end in spans[index + 1 :]:
                assert not (other_start >= start and other_end <= end), (
                    f"{count} words: ({other_start},{other_end}) is inside "
                    f"({start},{end})"
                )
