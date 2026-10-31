"""ETA evaluation entry points."""

from .reproducibility import compare_run_metrics

__all__ = ["compare_run_metrics", "evaluate_run"]


def __getattr__(name: str):
    if name == "evaluate_run":
        from .evaluator import evaluate_run

        return evaluate_run
    raise AttributeError(name)
