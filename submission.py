"""Submission for exercise sheet 5

SUBMIT this file as submission_<STUDENTID>.py where
you replace <STUDENTID> with your student ID, e.g.,
submission_1234567.py
"""
import torch


# Exercise 5.1
def stable_log_softmax(logits: torch.Tensor):
    # YOUR CODE GOES HERE   
    pass


# Exercise 5.2
def cross_entropy_from_scratch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    # YOUR CODE GOES HERE
    pass
