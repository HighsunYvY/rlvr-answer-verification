"""TRL v0.24.0 keyword callback contract; all component rewards are in [0, 1]."""
from .verifier import format_compliant, strict_verifier, weak_verifier


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if (isinstance(completion, list) and len(completion) == 1
            and isinstance(completion[0], dict)
            and completion[0].get("role") == "assistant"
            and isinstance(completion[0].get("content"), str)):
        return completion[0]["content"]
    return ""


def _batch(completions):
    if not isinstance(completions, (list, tuple)):
        raise TypeError("completions must be a batch, not a single string")
    return [completion_text(c) for c in completions]


def format_reward(completions, **kwargs):
    return [float(format_compliant(t)) for t in _batch(completions)]


def _accuracy(completions, ground_truth, verifier):
    texts = _batch(completions)
    if not isinstance(ground_truth, (list, tuple)) or len(texts) != len(ground_truth):
        raise ValueError("ground_truth must align one-to-one with completions")
    return [float(verifier(t, g)) for t, g in zip(texts, ground_truth)]


def strict_accuracy_reward(completions, ground_truth, **kwargs):
    return _accuracy(completions, ground_truth, strict_verifier)


def weak_accuracy_reward(completions, ground_truth, **kwargs):
    return _accuracy(completions, ground_truth, weak_verifier)
