"""Test that Sama's public API is importable."""


def test_import_decision_engine():
    from sama import DecisionEngine
    assert DecisionEngine is not None


def test_import_types():
    from sama import Decision, DecisionResponse
    assert Decision is not None
    assert DecisionResponse is not None


def test_import_losses():
    from sama.losses import rlcd_choice_loss
    assert callable(rlcd_choice_loss)


def test_version():
    import sama
    assert sama.__version__ == "0.2.0"
