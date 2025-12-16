from otter.test_files import test_case
import torch
import torch.nn.functional as F
from unittest.mock import patch


OK_FORMAT = False

name = "Exercise 5.2"
points = 4


def _blocked(*args, **kwargs):
    raise AssertionError(
        "Do not call torch.nn.functional.cross_entropy or torch.nn.CrossEntropyLoss "
        "inside cross_entropy_from_scratch."
    )


def _make_case(
    seed: int,
    N: int,
    C: int,
    device: str = "cpu",
    ignore_index: int | None = None,
    p_ignore: float = 0.2,
):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    logits = torch.randn(N, C, generator=g, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, C, (N,), generator=g, device=device, dtype=torch.long)

    if ignore_index is not None:
        mask = torch.rand(N, generator=g, device=device) < p_ignore
        targets = targets.clone()
        targets[mask] = ignore_index

    return logits, targets


@test_case(points=2)
def test_1(env):
    device = "cpu"
    ignore_index = 255
    reduction = "mean"
    N, C = 257, 13  # non-round sizes help catch shape bugs
    logits, targets = _make_case(
        seed=123,
        N=N,
        C=C,
        device=device,
        ignore_index=ignore_index)

    with patch.object(F, "cross_entropy", side_effect=_blocked), patch("torch.nn.CrossEntropyLoss", side_effect=_blocked):
        got = env['cross_entropy_from_scratch'](
            logits,
            targets,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    exp = F.cross_entropy(
        logits.detach(),  # compare values; grads are tested separately
        targets,
        ignore_index=ignore_index,
        reduction=reduction,
    )

    assert torch.allclose(got.detach(), exp, atol=5e-6, rtol=5e-5), (
        f"Loss mismatch.\n"
        f"max_abs_diff={(got.detach() - exp).abs().max().item()}"
    )


@test_case(points=1)
def test_2(env):
    device = "cpu"
    ignore_index = 255

    logits = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [5.0,  5.0,  5.0],
    ], device=device, requires_grad=True)

    targets = torch.tensor([0, ignore_index, 2, ignore_index], device=device, dtype=torch.long)

    with patch.object(F, "cross_entropy", side_effect=_blocked), patch("torch.nn.CrossEntropyLoss", side_effect=_blocked):
        got = env['cross_entropy_from_scratch'](
            logits, targets, ignore_index=ignore_index, reduction="mean"
        )

    exp = F.cross_entropy(
        logits.detach(), targets, ignore_index=ignore_index, reduction="mean"
    )
    assert torch.allclose(got.detach(), exp, atol=1e-6, rtol=1e-6)


@test_case(points=1)
def test_3(env):
    device = "cpu"
    ignore_index = 255
    N, C = 128, 11

    logits1, targets = _make_case(seed=999, N=N, C=C, device=device, ignore_index=ignore_index)
    logits2 = logits1.clone().detach().requires_grad_(True)

    with patch.object(F, "cross_entropy", side_effect=_blocked), patch("torch.nn.CrossEntropyLoss", side_effect=_blocked):    
        loss1 = env['cross_entropy_from_scratch'](
            logits1,
            targets,
            ignore_index=ignore_index,
            reduction="mean",
        )

    loss2 = F.cross_entropy(
        logits2,
        targets,
        ignore_index=ignore_index,
        reduction="mean",
    )

    loss1.backward()
    loss2.backward()

    assert torch.allclose(logits1.grad, logits2.grad, atol=5e-5, rtol=5e-4), (
        f"Gradient mismatch.\n"
        f"max_abs_grad_diff={(logits1.grad - logits2.grad).abs().max().item()}"
    )
