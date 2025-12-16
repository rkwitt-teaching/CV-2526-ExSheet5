from otter.test_files import test_case
from unittest.mock import patch
import torch.nn.functional as F
import torch

OK_FORMAT = False

name = "Exercise 5.1"
points = 2


def _blocked(*args, **kwargs):
    raise AssertionError(
        "Do not call torch.nn.functional.cross_entropy or torch.nn.CrossEntropyLoss "
        "inside cross_entropy_from_scratch."
    )


@test_case(points=2)
def test_1(env):
    # check numerical stability
    logits = torch.tensor([[1000.0, 999.0, 998.0],
                           [1200.0, -1200.0, 0.0]], dtype=torch.float32)

    with patch("torch.logsumexp", side_effect=_blocked):
        out = env['stable_log_softmax'](logits, dim=1)

    assert torch.isfinite(out).all(), "Output contains inf/nan; likely naive implementation."

    # check that softmax sums to one
    probs = torch.exp(out)
    s = probs.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6, rtol=1e-6)

    # check against PyTorch's log_softmax
    torch.manual_seed(0)
    logits = torch.randn(64, 10) * 50.0 
    got = env['stable_log_softmax'](logits, dim=1)
    exp = torch.log_softmax(logits, dim=1)
    assert torch.allclose(got, exp, atol=1e-6, rtol=1e-6)


