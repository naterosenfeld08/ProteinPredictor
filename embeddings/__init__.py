"""
Embedding extraction and composition module.

This module handles:
1. Extraction of embeddings from protein language models (ProtT5-XL, ESM-2)
2. Computation of amino acid composition features
3. Composition of embeddings into fixed-dimensional feature vectors

The composition method is concatenation along the feature dimension:
- ProtT5-XL: 1,024 dimensions
- ESM-2: 1,280 dimensions
- Composition: 20 dimensions
- Total: 2,344 dimensions
"""

from .composition import (
    compose_embeddings,
    compute_amino_acid_composition,
    add_composition_features_to_embeddings
)

__all__ = [
    'compose_embeddings',
    'compute_amino_acid_composition',
    'add_composition_features_to_embeddings'
]

