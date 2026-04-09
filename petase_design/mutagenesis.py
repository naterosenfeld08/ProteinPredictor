"""
Random mutagenesis helpers: sample positions and amino acids, apply to WT sequence.
"""

from __future__ import annotations

import random
from typing import Collection

from petase_design.sequence_utils import VALID_AA, apply_mutations, mutation_diff


def propose_random_mutations(
    wt: str,
    n_mutations: int,
    *,
    rng: random.Random | None = None,
    protected_indices: Collection[int] = (),
) -> list[tuple[int, str]]:
    """
    Sample n distinct positions (uniform among allowed) and random AAs (excluding same residue).
    """
    r = rng or random.Random()
    n_mutations = min(n_mutations, len(wt))
    allowed = [i for i in range(len(wt)) if i not in set(protected_indices)]
    if n_mutations > len(allowed):
        n_mutations = len(allowed)
    picks = r.sample(allowed, n_mutations)
    muts: list[tuple[int, str]] = []
    for i in picks:
        old = wt[i].upper()
        choices = [a for a in VALID_AA if a != old]
        muts.append((i, r.choice(choices)))
    return muts


def _weighted_sample_without_replacement(
    indices: list[int],
    weights: list[float],
    k: int,
    *,
    rng: random.Random,
) -> list[int]:
    """Simple weighted sample without replacement using repeated draws."""
    k = min(max(k, 0), len(indices))
    if k <= 0:
        return []
    pool_idx = list(indices)
    pool_w = [max(0.0, float(w)) for w in weights]
    if not any(pool_w):
        return rng.sample(pool_idx, k)

    out: list[int] = []
    for _ in range(k):
        total = sum(pool_w)
        if total <= 0:
            remaining = [i for i in pool_idx if i not in out]
            if not remaining:
                break
            out.append(rng.choice(remaining))
            continue
        r = rng.random() * total
        c = 0.0
        chosen_j = 0
        for j, w in enumerate(pool_w):
            c += w
            if r <= c:
                chosen_j = j
                break
        out.append(pool_idx.pop(chosen_j))
        pool_w.pop(chosen_j)
    return out


def propose_weighted_mutations(
    parent_seq: str,
    n_mutations: int,
    *,
    rng: random.Random | None = None,
    protected_indices: Collection[int] = (),
    position_weights: list[float] | None = None,
) -> list[tuple[int, str]]:
    """Sample mutations with optional position-priority weights."""
    r = rng or random.Random()
    n_mutations = min(n_mutations, len(parent_seq))
    protected = set(protected_indices)
    allowed = [i for i in range(len(parent_seq)) if i not in protected]
    if n_mutations > len(allowed):
        n_mutations = len(allowed)
    if n_mutations <= 0:
        return []

    if position_weights is None or len(position_weights) != len(parent_seq):
        picks = r.sample(allowed, n_mutations)
    else:
        w = [position_weights[i] for i in allowed]
        picks = _weighted_sample_without_replacement(allowed, w, n_mutations, rng=r)

    muts: list[tuple[int, str]] = []
    for i in picks:
        old = parent_seq[i].upper()
        choices = [a for a in VALID_AA if a != old]
        muts.append((i, r.choice(choices)))
    return muts


def propose_recombined_variant(
    wt: str,
    parent_a: str,
    parent_b: str,
    n_mutations: int,
    *,
    rng: random.Random | None = None,
    protected_indices: Collection[int] = (),
) -> tuple[str, list[str]]:
    """
    Recombine two parent variants against WT, then trim/expand to mutation budget.
    Returns (child_sequence, parent_job_ids_placeholder_metadata).
    """
    r = rng or random.Random()
    protected = set(protected_indices)
    diff_a = {i: aa for i, _wt, aa in mutation_diff(wt, parent_a)}
    diff_b = {i: aa for i, _wt, aa in mutation_diff(wt, parent_b)}
    union_pos = [i for i in sorted(set(diff_a) | set(diff_b)) if i not in protected]

    child = list(wt)
    chosen_positions: set[int] = set()
    for i in union_pos:
        candidates = []
        if i in diff_a:
            candidates.append(diff_a[i])
        if i in diff_b:
            candidates.append(diff_b[i])
        if not candidates:
            continue
        child[i] = r.choice(candidates)
        chosen_positions.add(i)

    # Enforce mutation budget.
    n_mutations = min(max(1, int(n_mutations)), len(wt))
    diff_positions = [i for i, _wa, _va in mutation_diff(wt, "".join(child)) if i not in protected]
    if len(diff_positions) > n_mutations:
        to_revert = r.sample(diff_positions, len(diff_positions) - n_mutations)
        for i in to_revert:
            child[i] = wt[i]
    elif len(diff_positions) < n_mutations:
        add = propose_random_mutations(
            "".join(child),
            n_mutations - len(diff_positions),
            rng=r,
            protected_indices=protected,
        )
        child = list(apply_mutations("".join(child), add))
    return "".join(child), ["recombine"]


def variant_from_mutations(wt: str, muts: list[tuple[int, str]]) -> str:
    return apply_mutations(wt, muts)
