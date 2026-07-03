"""robocurate -- informativeness-aware curation of robot demonstrations.

Thesis: low-cost non-expert demonstrations are scalable but noisy; this project
studies whether *quality-weighted diversity* curation can recover useful
demonstrations from such pools. Informativeness = quality x diversity, realized
with a quality-weighted DPP.
"""

from .dpp import build_dpp_kernel, cosine_similarity_matrix, greedy_map, select_dpp
from .metrics import (
    coverage_radius,
    diversity_spread,
    expert_fraction,
    selected_mean_quality,
)
from .quality import normalize_quality, proxy_quality
from .selectors import (
    SELECTORS,
    select_diversity_only,
    select_quality_only,
    select_random,
)
from .synthetic import make_labeled_pool, make_mixed_pool

__all__ = [
    "build_dpp_kernel",
    "cosine_similarity_matrix",
    "greedy_map",
    "select_dpp",
    "select_random",
    "select_quality_only",
    "select_diversity_only",
    "SELECTORS",
    "normalize_quality",
    "proxy_quality",
    "selected_mean_quality",
    "coverage_radius",
    "diversity_spread",
    "expert_fraction",
    "make_mixed_pool",
    "make_labeled_pool",
]
