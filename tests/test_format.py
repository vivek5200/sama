import pytest
from sama.format import (
    format_choice, format_noul, format_score, format_question,
    FORMAT_VERSION, check_format_compatibility, FormatVersionError,
)


def test_format_choice_exact():
    state = "Gold is a metal."
    q = {"type": "choice", "instructions": "Which?", "options": ["Au", "Ag", "Gd", "Go"]}
    expected = "Gold is a metal.\n\nQuestion: Which?\nType: choice\nOptions: (A) Au (B) Ag (C) Gd (D) Go\nAnswer:"
    assert format_choice(state, q) == expected


def test_format_noul_exact():
    state = "The Earth is flat."
    q = {"type": "noul", "instructions": "Is this true?"}
    expected = "The Earth is flat.\n\nQuestion: Is this true?\nType: noul\nAnswer:"
    assert format_noul(state, q) == expected


def test_format_question_dispatches():
    state = "x"
    q = {"type": "noul", "instructions": "y"}
    assert format_question(state, q) == format_noul(state, q)


def test_format_version_match():
    check_format_compatibility(FORMAT_VERSION)  # no error


def test_format_version_mismatch():
    with pytest.raises(FormatVersionError):
        check_format_compatibility("0.9.0")


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        format_question("x", {"type": "ranking", "instructions": "y"})
