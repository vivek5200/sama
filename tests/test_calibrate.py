import numpy as np
from sama.calibrate import ece, apply_power_scaling


def test_ece_perfect():
    confs = np.array([0.9] * 100)
    corrects = np.array([1] * 90 + [0] * 10)
    assert ece(confs, corrects) < 0.01


def test_ece_overconfident():
    confs = np.array([0.9] * 100)
    corrects = np.array([1] * 50 + [0] * 50)  # only 50% correct
    assert ece(confs, corrects) > 0.3


def test_power_scaling_identity():
    assert apply_power_scaling(0.9, 1.0) == 0.9


def test_power_scaling_sharpens():
    # T < 1 sharpens (pushes confidence toward 1)
    assert apply_power_scaling(0.9, 0.5) > 0.9
