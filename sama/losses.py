"""RLCD loss for calibrated decision heads.

The RLCD (Reinforcement Learning with Calibrated Decisions) loss combines
cross-entropy for accuracy with a Brier score penalty for calibration.
"""
import torch
import torch.nn.functional as F


def rlcd_choice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_options: int,
    brier_weight: float = 0.5,
) -> torch.Tensor:
    """RLCD loss: cross-entropy + brier_weight * Brier score.

    Args:
        logits: Raw logits of shape (batch, max_options).
        labels: Ground-truth class indices of shape (batch,).
        n_options: Number of valid options (logits are sliced to [:, :n_options]).
        brier_weight: Weight for the Brier score term (default 0.5).

    Returns:
        Scalar loss tensor.
    """
    logits = logits[:, :n_options]
    ce = F.cross_entropy(logits, labels)

    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(labels, num_classes=n_options).float()
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean()

    return ce + brier_weight * brier
