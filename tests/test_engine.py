"""Test that _format_state produces the exact training format."""
from sama.engine import DecisionEngine, TypedDecisionHead


class FakeEngine:
    """Minimal stub to test _format_state without loading a real model."""
    _format_state = DecisionEngine._format_state


def test_format_state_basic():
    engine = FakeEngine()
    result = engine._format_state(
        "What is the chemical symbol for gold?",
        {
            "answer": {
                "type": "choice",
                "instructions": "Which of the following is the correct answer?",
                "options": ["Au", "Ag", "Gd", "Go"],
            }
        },
    )
    expected = (
        "What is the chemical symbol for gold?\n"
        "\n"
        "Question: Which of the following is the correct answer?\n"
        "Options: (A) Au (B) Ag (C) Gd (D) Go\n"
        "Answer:"
    )
    assert result == expected, f"Format mismatch:\n{result!r}\n!=\n{expected!r}"


def test_format_state_dict_options():
    """Options can be a dict — keys should be used."""
    engine = FakeEngine()
    result = engine._format_state(
        "Context here",
        {
            "q": {
                "type": "choice",
                "instructions": "Pick one",
                "options": {"Yes": 1, "No": 0},
            }
        },
    )
    assert "(A) Yes" in result
    assert "(B) No" in result
    # No pipe separators
    assert " | " not in result


def test_format_state_no_qid_in_output():
    """The question line must NOT include the question ID."""
    engine = FakeEngine()
    result = engine._format_state(
        "Some context",
        {
            "my_question_id": {
                "type": "choice",
                "instructions": "Which is correct?",
                "options": ["A", "B", "C", "D"],
            }
        },
    )
    assert "my_question_id" not in result
    assert "Question: Which is correct?" in result


def test_typed_decision_head_output_shape():
    """TypedDecisionHead should return choice_logits with max_options columns."""
    import torch
    head = TypedDecisionHead(hidden_dim=64, proj_dim=32, max_options=4)
    h = torch.randn(2, 64)
    out = head(h)
    assert "choice_logits" in out
    assert out["choice_logits"].shape == (2, 4)
