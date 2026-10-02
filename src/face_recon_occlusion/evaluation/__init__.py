"""
Evaluation metrics, ablation study runner, and failure analysis for 3D face PAD.
"""

from .pad_metrics import compute_pad_metrics
from .ablation import run_ablation_experiments
from .failure_analysis import analyze_failures

__all__ = [
    "compute_pad_metrics",
    "run_ablation_experiments",
    "analyze_failures",
]
