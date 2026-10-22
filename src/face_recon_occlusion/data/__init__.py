"""
Dataset metadata, subject-disjoint splitting, and validation utilities.
"""

from .splits import (
    create_subject_disjoint_split,
    normalize_id,
    summarize_split,
    verify_subject_disjoint,
)

__all__ = [
    "create_subject_disjoint_split",
    "normalize_id",
    "summarize_split",
    "verify_subject_disjoint",
]
