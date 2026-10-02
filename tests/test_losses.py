"""Test the RLCD loss function with hand-computed values."""
import torch
from sama.losses import rlcd_choice_loss


def test_rlcd_perfect_prediction():
    """When the model is perfectly confident and correct, loss should be low."""
    logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
    labels = torch.tensor([0])
    loss = rlcd_choice_loss(logits, labels, n_options=4)
    assert loss.item() < 0.1, f"Expected near-zero loss, got {loss.item()}"


def test_rlcd_uniform_prediction():
    """Uniform logits should give higher loss than a confident correct prediction."""
    uniform_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    confident_logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
    labels = torch.tensor([0])

    loss_uniform = rlcd_choice_loss(uniform_logits, labels, n_options=4)
    loss_confident = rlcd_choice_loss(confident_logits, labels, n_options=4)
    assert loss_uniform > loss_confident


def test_rlcd_wrong_prediction():
    """Confident wrong prediction should have high loss."""
    logits = torch.tensor([[-10.0, 10.0, -10.0, -10.0]])
    labels = torch.tensor([0])  # Correct answer is 0, model says 1
    loss = rlcd_choice_loss(logits, labels, n_options=4)
    assert loss.item() > 1.0


def test_rlcd_hand_computed():
    """Hand-compute the loss for a simple case.

    logits = [1, 0] with label 0, n_options=2
    softmax([1,0]) = [e/(e+1), 1/(e+1)] ≈ [0.7311, 0.2689]
    CE = -log(0.7311) ≈ 0.3133
    Brier = (0.7311-1)^2 + (0.2689-0)^2 = 0.0723 + 0.0723 = 0.1446
    RLCD = 0.3133 + 0.5 * 0.1446 = 0.3856
    """
    logits = torch.tensor([[1.0, 0.0, -999.0, -999.0]])  # only 2 valid
    labels = torch.tensor([0])
    loss = rlcd_choice_loss(logits, labels, n_options=2, brier_weight=0.5)
    assert abs(loss.item() - 0.3856) < 0.01, f"Expected ~0.3856, got {loss.item()}"


def test_rlcd_respects_n_options():
    """Loss should only use the first n_options logits."""
    logits = torch.tensor([[1.0, 0.0, 100.0, 100.0]])
    labels = torch.tensor([0])
    loss_2 = rlcd_choice_loss(logits, labels, n_options=2)
    loss_4 = rlcd_choice_loss(logits, labels, n_options=4)
    # With n_options=2, the huge logits at positions 2,3 are ignored
    assert loss_2 < 1.0
    assert loss_4 > loss_2


def test_rlcd_batch():
    """Loss should work with batched inputs."""
    logits = torch.tensor([[5.0, -5.0, -5.0, -5.0],
                           [5.0, -5.0, -5.0, -5.0]])
    labels = torch.tensor([0, 0])
    loss = rlcd_choice_loss(logits, labels, n_options=4)
    assert loss.dim() == 0  # scalar
    assert loss.item() < 0.1
