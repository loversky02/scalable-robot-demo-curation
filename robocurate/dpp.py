"""Determinantal Point Process (DPP) selection with a quality x diversity kernel.

The kernel follows the Kulesza & Taskar (2012) quality/diversity decomposition:

    L = diag(q) @ S @ diag(q)

where ``q_i >= 0`` is a per-item *quality* score and ``S_ij`` is a *similarity*
(here cosine similarity of item embeddings). Because ``S`` is a Gram matrix of
normalized vectors it is positive semi-definite, so ``L`` stays PSD for any
non-negative ``q`` -- a valid DPP kernel.

Selecting a high-probability subset under ``L`` trades off *quality* (pick good
items) against *diversity* (pick dissimilar items). This is exactly the property
we want for demonstration curation: on a noisy, low-cost demonstration pool a
pure-diversity selector chases outliers (novice noise looks "novel"), while a
pure-quality selector collapses onto a few redundant modes. The DPP couples both.

MAP inference uses the fast greedy algorithm of Chen et al. (2018), "Fast Greedy
MAP Inference for Determinantal Point Processes", with an incremental Cholesky
update (O(N k^2)).
"""

from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return the NxN cosine-similarity Gram matrix (always PSD)."""
    X = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Xn = X / norms
    S = Xn @ Xn.T
    return np.clip(S, -1.0, 1.0)


def build_dpp_kernel(
    embeddings: np.ndarray, quality: np.ndarray, jitter: float = 1e-8
) -> np.ndarray:
    """Build the quality x diversity DPP kernel ``L = diag(q) S diag(q)``.

    ``jitter`` adds a tiny ridge for numerical stability of the greedy MAP.
    """
    S = cosine_similarity_matrix(embeddings)
    q = np.asarray(quality, dtype=float)
    if np.any(q < 0):
        raise ValueError("quality scores must be non-negative for a valid DPP kernel")
    L = (q[:, None] * S) * q[None, :]
    n = L.shape[0]
    return L + jitter * np.eye(n)


def greedy_map(L: np.ndarray, k: int, eps: float = 1e-10, return_gains: bool = False):
    """Greedy MAP inference for a DPP with kernel ``L``.

    Returns up to ``k`` selected indices that greedily maximize ``log det(L_S)``.
    With ``return_gains=True`` also returns the per-step marginal gains; their
    product equals ``det(L_S)`` (used as a correctness check in the tests).
    """
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    k = int(min(k, n))
    if k <= 0:
        return ([], []) if return_gains else []

    di2 = np.diag(L).astype(float).copy()      # conditional "gains" d_i^2
    C = np.zeros((k, n))                        # incremental Cholesky rows
    chosen = np.zeros(n, dtype=bool)

    j = int(np.argmax(di2))
    selected = [j]
    gains = [float(di2[j])]
    chosen[j] = True

    for t in range(1, k):
        last = selected[-1]
        if t >= 2:
            prev = C[: t - 1, :]                # (t-1, n)
            dot = prev[:, last] @ prev          # (n,)
        else:
            dot = np.zeros(n)
        denom = np.sqrt(max(di2[last], eps))
        e = (L[last, :] - dot) / denom
        C[t - 1, :] = e
        di2 = di2 - e ** 2
        masked = np.where(chosen, -np.inf, di2)
        j = int(np.argmax(masked))
        if masked[j] <= eps:
            break
        selected.append(j)
        gains.append(float(di2[j]))
        chosen[j] = True

    if return_gains:
        return selected, gains
    return selected


def select_dpp(embeddings: np.ndarray, quality: np.ndarray, k: int) -> list[int]:
    """Convenience wrapper: build the quality x diversity kernel and run greedy MAP."""
    L = build_dpp_kernel(embeddings, quality)
    return greedy_map(L, k)
