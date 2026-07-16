__all__ = [
    "UMSVerifier",
    "compute_classification_metrics",
    "compute_reliability_metrics",
]


def __getattr__(name):
    if name == "UMSVerifier":
        from .verifier import UMSVerifier

        return UMSVerifier
    if name in {"compute_classification_metrics", "compute_reliability_metrics"}:
        from .metrics import compute_classification_metrics, compute_reliability_metrics

        return {
            "compute_classification_metrics": compute_classification_metrics,
            "compute_reliability_metrics": compute_reliability_metrics,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
