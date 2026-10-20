import pytest

def test_engine_loads_v0_1():
    from sama import DecisionEngine
    e = DecisionEngine.from_pretrained("./models/v0.1")
    assert "choice" in e.heads


def test_engine_loads_v0_2():
    from sama import DecisionEngine
    e = DecisionEngine.from_pretrained("./models/v0.2")
    assert "choice" in e.heads


def test_engine_loads_v0_3():
    from sama import DecisionEngine
    e = DecisionEngine.from_pretrained("./models/v0.3")
    assert "choice" in e.heads
    assert "noul" in e.heads
    assert "score" in e.heads


def test_missing_head_error():
    from sama import DecisionEngine
    from sama.engine import MissingHeadError
    e = DecisionEngine.from_pretrained("./models/v0.2")  # choice only
    with pytest.raises(MissingHeadError):
        e.decide("test", {"q": {"type": "noul", "instructions": "is this true?"}})


def test_format_version_mismatch_raises(tmp_path):
    # Create a fake config with wrong format version
    import json
    from sama import DecisionEngine
    from sama.format import FormatVersionError
    cfg = {"encoder": "Qwen/Qwen2.5-0.5B", "hidden_dim": 896,
           "format_version": "0.5.0", "heads": {}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(FormatVersionError):
        DecisionEngine(str(tmp_path))
