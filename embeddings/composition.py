"""
Embedding composition utilities.

This module provides functions for composing multiple embedding types
into a single fixed-dimensional feature vector.

Composition Method: Concatenation along feature dimension (axis=1).

Final dimensionality: 2,344 features per protein sequence
- ProtT5-XL: 1,024 dimensions
- ESM-2: 1,280 dimensions  
- Amino acid composition: 20 dimensions
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from collections import defaultdict

from config.constants import (
    EMBEDDING_DIMENSIONS,
    CANONICAL_AMINO_ACIDS,
    CANONICAL_AA_SET
)


def compute_amino_acid_composition(sequence: str) -> np.ndarray:
    """
    Compute amino acid composition features for a protein sequence.
    
    For each of the 20 canonical amino acids, computes the normalized
    frequency (count / sequence_length). This provides a 20-dimensional
    feature vector representing the composition of the sequence.
    
    Args:
        sequence: Protein sequence as a string of single-letter amino acid codes.
                  Must contain only canonical amino acids (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).
        
    Returns:
        numpy.ndarray of shape (20,): Normalized frequency of each canonical amino acid.
        Order: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y.
        Values are in range [0, 1] and sum to 1.0 (if all residues are canonical).
        
    Raises:
        None (returns zeros for empty sequences).
        
    Example:
        >>> seq = "MKTAYIAKQR"
        >>> composition = compute_amino_acid_composition(seq)
        >>> composition.shape
        (20,)
        >>> composition.sum()  # Approximately 1.0 (may be <1.0 if non-canonical AAs present)
    """
    sequence_length = len(sequence)
    if sequence_length == 0:
        return np.zeros(EMBEDDING_DIMENSIONS['amino_acid_composition'])
    
    # Count occurrences of each canonical amino acid
    amino_acid_counts = defaultdict(int)
    for amino_acid in sequence:
        if amino_acid in CANONICAL_AA_SET:
            amino_acid_counts[amino_acid] += 1
    
    # Compute normalized frequencies
    composition_array = np.array([
        amino_acid_counts.get(amino_acid, 0) / sequence_length
        for amino_acid in sorted(CANONICAL_AMINO_ACIDS)
    ])
    
    return composition_array


def compose_embeddings(
    prott5_embedding: Optional[np.ndarray] = None,
    esm2_embedding: Optional[np.ndarray] = None,
    composition_features: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compose multiple embedding types into a single feature vector.
    
    Composition method: Concatenation along feature dimension (axis=0 for 1D arrays).
    
    The function concatenates available embeddings in the order:
    1. ProtT5-XL embedding (if provided): 1,024 dimensions
    2. ESM-2 embedding (if provided): 1,280 dimensions
    3. Composition features (if provided): 20 dimensions
    
    At least one embedding type must be provided.
    
    Args:
        prott5_embedding: Optional ProtT5-XL embedding vector of shape (1024,).
                         Extracted from Rostlab/prot_t5_xl_uniref50 model.
        esm2_embedding: Optional ESM-2 embedding vector of shape (1280,).
                       Extracted from facebook/esm2_t33_650M_UR50D model.
        composition_features: Optional amino acid composition vector of shape (20,).
                             Computed from sequence composition.
        
    Returns:
        numpy.ndarray of shape (n_features,): Composed embedding vector.
        Dimensionality depends on which embeddings are provided:
        - ProtT5 only: 1,024 dimensions
        - ESM-2 only: 1,280 dimensions
        - ProtT5 + ESM-2: 2,304 dimensions
        - ProtT5 + ESM-2 + Composition: 2,344 dimensions (standard configuration)
        
    Raises:
        ValueError: If no embeddings are provided.
        
    Assumptions:
        1. All provided embeddings are 1D arrays (flattened)
        2. Embeddings are already extracted and mean-pooled (if applicable)
        3. Composition features are computed from the same sequence as embeddings
        4. Embeddings are independent (no explicit interaction modeling)
        
    Example:
        >>> prott5 = np.random.randn(1024)
        >>> esm2 = np.random.randn(1280)
        >>> composition = np.random.randn(20)
        >>> composed = compose_embeddings(prott5, esm2, composition)
        >>> composed.shape
        (2344,)
    """
    embedding_components = []
    
    if prott5_embedding is not None:
        if prott5_embedding.shape[0] != EMBEDDING_DIMENSIONS['prott5_xl']:
            raise ValueError(
                f"ProtT5 embedding dimension mismatch: expected {EMBEDDING_DIMENSIONS['prott5_xl']}, "
                f"got {prott5_embedding.shape[0]}"
            )
        embedding_components.append(prott5_embedding)
    
    if esm2_embedding is not None:
        if esm2_embedding.shape[0] != EMBEDDING_DIMENSIONS['esm2_650m']:
            raise ValueError(
                f"ESM-2 embedding dimension mismatch: expected {EMBEDDING_DIMENSIONS['esm2_650m']}, "
                f"got {esm2_embedding.shape[0]}"
            )
        embedding_components.append(esm2_embedding)
    
    if composition_features is not None:
        if composition_features.shape[0] != EMBEDDING_DIMENSIONS['amino_acid_composition']:
            raise ValueError(
                f"Composition features dimension mismatch: expected {EMBEDDING_DIMENSIONS['amino_acid_composition']}, "
                f"got {composition_features.shape[0]}"
            )
        embedding_components.append(composition_features)
    
    if len(embedding_components) == 0:
        raise ValueError("At least one embedding type must be provided")
    
    # Concatenate along feature dimension (axis=0 for 1D arrays)
    composed_embedding = np.concatenate(embedding_components, axis=0)
    
    return composed_embedding


def add_composition_features_to_embeddings(
    embeddings_dict: Dict[str, np.ndarray],
    sequences: pd.Series
) -> Dict[str, np.ndarray]:
    """
    Add amino acid composition features to existing embeddings.
    
    For each embedding in the dictionary, concatenates composition features
    along the feature dimension (axis=1 for 2D arrays, axis=0 for 1D arrays).
    
    This function modifies the embedding dimensionality:
    - Before: Original embedding dimension (e.g., 1024 for ProtT5, 1280 for ESM-2)
    - After: Original dimension + 20 (composition features)
    
    Args:
        embeddings_dict: Dictionary mapping embedding names to numpy arrays.
                        Arrays can be 1D (single sequence) or 2D (batch of sequences).
                        Shape: (n_sequences, n_features) for 2D, or (n_features,) for 1D.
        sequences: pandas.Series of protein sequences (amino acid strings).
                  Length must match the number of sequences in embeddings.
                  Each sequence is used to compute composition features.
        
    Returns:
        Dictionary with same keys as input, but values are enhanced embeddings
        with composition features concatenated.
        Shape: (n_sequences, n_features + 20) for 2D inputs, or (n_features + 20,) for 1D.
        
    Raises:
        ValueError: If number of sequences doesn't match between embeddings and sequences.
        
    Assumptions:
        1. All embeddings in dictionary have the same number of sequences (first dimension)
        2. Sequences correspond to the same order as embeddings
        3. Composition features are computed independently for each sequence
        4. Concatenation assumes composition features are complementary to embeddings
        
    Example:
        >>> embeddings = {'prot_t5': np.random.randn(10, 1024)}
        >>> sequences = pd.Series(['MKTAYIAKQR'] * 10)
        >>> enhanced = add_composition_features_to_embeddings(embeddings, sequences)
        >>> enhanced['prot_t5'].shape
        (10, 1044)  # 1024 + 20
    """
    n_sequences = len(sequences)
    
    # Compute composition features for all sequences
    composition_features_array = np.array([
        compute_amino_acid_composition(sequence)
        for sequence in sequences
    ])
    
    # Verify shape consistency
    if composition_features_array.shape[0] != n_sequences:
        raise ValueError(
            f"Composition features shape mismatch: expected {n_sequences} sequences, "
            f"got {composition_features_array.shape[0]}"
        )
    
    enhanced_embeddings = {}
    
    for embedding_name, embedding_array in embeddings_dict.items():
        # Handle both 1D (single sequence) and 2D (batch) cases
        if embedding_array.ndim == 1:
            # Single sequence: shape (n_features,)
            if n_sequences != 1:
                raise ValueError(
                    f"1D embedding array provided but {n_sequences} sequences given. "
                    "Use 2D array for batch processing."
                )
            # Reshape to 2D for concatenation, then reshape back
            embedding_2d = embedding_array.reshape(1, -1)
            composition_2d = composition_features_array.reshape(1, -1)
            enhanced_2d = np.concatenate([embedding_2d, composition_2d], axis=1)
            enhanced_embeddings[embedding_name] = enhanced_2d.reshape(-1)
        elif embedding_array.ndim == 2:
            # Batch of sequences: shape (n_sequences, n_features)
            if embedding_array.shape[0] != n_sequences:
                raise ValueError(
                    f"Embedding batch size mismatch: embedding has {embedding_array.shape[0]} sequences, "
                    f"but {n_sequences} sequences provided"
                )
            # Concatenate along feature dimension (axis=1)
            enhanced_embeddings[embedding_name] = np.concatenate(
                [embedding_array, composition_features_array],
                axis=1
            )
        else:
            raise ValueError(
                f"Unsupported embedding array dimensionality: {embedding_array.ndim}. "
                "Expected 1D (single sequence) or 2D (batch of sequences)."
            )
    
    return enhanced_embeddings

