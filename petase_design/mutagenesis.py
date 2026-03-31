"""
Random mutagenesis helpers: sample positions and amino acids, apply to WT sequence.
"""

from __future__ import annotations

import random
from typing import Collection

from petase_design.sequence_utils import VALID_AA, apply_mutations


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


def variant_from_mutations(wt: str, muts: list[tuple[int, str]]) -> str:
    return apply_mutations(wt, muts)
