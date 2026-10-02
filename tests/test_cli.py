"""Test the CLI interface (mocking model loading)."""
import json
import subprocess
import sys
from unittest.mock import patch, MagicMock

import pytest


def test_decide_help():
    """The decide subcommand should accept --help."""
    result = subprocess.run(
        [sys.executable, "-m", "sama.cli", "decide", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--state" in result.stdout
    assert "--options" in result.stdout


def test_serve_help():
    """The serve subcommand should accept --help."""
    result = subprocess.run(
        [sys.executable, "-m", "sama.cli", "serve", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--port" in result.stdout


def test_no_command_exits_nonzero():
    """Running without a subcommand should exit with code 1."""
    result = subprocess.run(
        [sys.executable, "-m", "sama.cli"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_decide_produces_json():
    """Mocked decide should produce valid JSON output."""
    from sama.types import Decision, DecisionResponse

    mock_response = DecisionResponse(
        decisions={
            "q": Decision(
                answer="billing",
                probabilities=[0.8, 0.1, 0.05, 0.05],
                confidence=0.8,
                action="soft_review",
                status="confident",
            )
        }
    )

    mock_engine = MagicMock()
    mock_engine.decide.return_value = mock_response

    mock_cls = MagicMock(return_value=mock_engine)

    with patch("sama.cli.DecisionEngine", mock_cls) if hasattr(sys.modules.get("sama.cli", None), "DecisionEngine") else patch("sama.DecisionEngine.from_pretrained", return_value=mock_engine):
        # Test via subprocess with mocking is complex, so test the function directly
        from sama.cli import cmd_decide
        import argparse

        args = argparse.Namespace(
            model="./models/v0.2",
            state="Customer says invoice was double-charged.",
            instructions="Which team should handle this?",
            options=["billing", "technical", "sales", "support"],
        )

        # Patch DecisionEngine in the sama module used by cli
        with patch("sama.DecisionEngine.from_pretrained", return_value=mock_engine):
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                cmd_decide(args)
            output = f.getvalue()

        parsed = json.loads(output)
        assert "decisions" in parsed
        assert "q" in parsed["decisions"]
        assert parsed["decisions"]["q"]["answer"] == "billing"
