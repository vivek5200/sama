import torch
from sama.heads import ChoiceHead, NoulHead, ScoreHead, HEAD_CLASSES


def test_choice_head_shape():
    head = ChoiceHead(hidden_dim=1536)
    out = head(torch.randn(4, 1536))
    assert out.shape == (4, 8)


def test_noul_head_shape():
    head = NoulHead(hidden_dim=1536)
    out = head(torch.randn(4, 1536))
    assert out.shape == (4,)


def test_score_head_shape():
    head = ScoreHead(hidden_dim=1536)
    out = head(torch.randn(4, 1536))
    assert out.shape == (4, 8)


def test_head_types_registered():
    assert HEAD_CLASSES["choice"] is ChoiceHead
    assert HEAD_CLASSES["noul"] is NoulHead
    assert HEAD_CLASSES["score"] is ScoreHead
