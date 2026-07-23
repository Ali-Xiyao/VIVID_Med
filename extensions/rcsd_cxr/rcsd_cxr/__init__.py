"""RCSD-CXR core package."""

from .config import DatasetRecord, DatasetRegistry
from .d0_d1_contract import (
    AgreementWeight,
    entropy_agreement_weight,
    render_hard_ums_target,
)
from .posterior import PosteriorResult, fuse_log_opinion_pool
from .token_objective import (
    finding_block_spans,
    prepare_token_batch,
    token_accuracy,
    token_cross_entropy,
)

__all__ = [
    "AgreementWeight",
    "DatasetRecord",
    "DatasetRegistry",
    "PosteriorResult",
    "entropy_agreement_weight",
    "finding_block_spans",
    "fuse_log_opinion_pool",
    "prepare_token_batch",
    "render_hard_ums_target",
    "token_accuracy",
    "token_cross_entropy",
]

__version__ = "0.1.0"
