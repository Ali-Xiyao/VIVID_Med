"""RCSD-CXR core package."""

from .config import DatasetRecord, DatasetRegistry
from .posterior import PosteriorResult, fuse_log_opinion_pool

__all__ = [
    "DatasetRecord",
    "DatasetRegistry",
    "PosteriorResult",
    "fuse_log_opinion_pool",
]

__version__ = "0.1.0"
