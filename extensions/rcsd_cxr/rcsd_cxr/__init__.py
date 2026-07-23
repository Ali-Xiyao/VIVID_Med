"""RCSD-CXR core package."""

from .config import DatasetRecord, DatasetRegistry
from .d0_d1_contract import (
    AgreementWeight,
    entropy_agreement_weight,
    render_hard_ums_target,
)
from .posterior import PosteriorResult, fuse_log_opinion_pool

__all__ = [
    "AgreementWeight",
    "DatasetRecord",
    "DatasetRegistry",
    "PosteriorResult",
    "entropy_agreement_weight",
    "fuse_log_opinion_pool",
    "render_hard_ums_target",
]

__version__ = "0.1.0"
