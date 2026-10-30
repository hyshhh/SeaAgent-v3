"""ETA evaluation entry points."""

from .paired import paired_comparison
from .reproducibility import compare_run_metrics

__all__ = ["compare_run_metrics", "evaluate_run", "paired_comparison"]


def __getattr__(name: str):
    if name == "evaluate_run":
        from .evaluator import evaluate_run

        return evaluate_run
    raise AttributeError(name)
