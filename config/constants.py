"""
Global constants for protein stability prediction pipeline.

This module centralizes all constants, random seeds, and configuration
parameters to ensure reproducibility and consistency across the codebase.

All random seeds are set here to ensure reproducible results across
different parts of the pipeline.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict

# ============================================================================
# REPRODUCIBILITY: Random Seeds
# ============================================================================

RANDOM_SEED: int = 42
"""
Global random seed for all random operations.

This seed is used for:
- NumPy random number generation
- PyTorch random number generation
- scikit-learn random_state parameters
- XGBoost random_state parameters
- Data splitting

Set once at module import to ensure reproducibility.
"""

# Set seeds immediately upon import
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ============================================================================
# EMBEDDING DIMENSIONS
# ============================================================================

EMBEDDING_DIMENSIONS: Dict[str, int] = {
    'prott5_xl': 1024,
    'esm2_650m': 1280,
    'amino_acid_composition': 20,
    'total_combined': 2344
}
"""
Embedding dimensionality specifications.

Dimensions:
- prott5_xl: ProtT5-XL embedding dimension (1024)
- esm2_650m: ESM-2 650M embedding dimension (1280)
- amino_acid_composition: Composition feature dimension (20)
- total_combined: Total dimension after concatenation (2344)

These dimensions are fixed and must match the actual embedding
extraction output. Any changes to embedding models must update
these constants.
"""

# ============================================================================
# AMINO ACID CONSTANTS
# ============================================================================

CANONICAL_AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"
"""
Canonical amino acid single-letter codes.

Order: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y

This ordering is used for:
- Composition feature computation
- Sequence validation
- Consistent feature ordering across the pipeline
"""

CANONICAL_AA_SET = set(CANONICAL_AMINO_ACIDS)
"""Set of canonical amino acids for fast membership testing."""

# ============================================================================
# DEFAULT PATHS
# ============================================================================

DEFAULT_PATHS: Dict[str, Path] = {
    'training_output': Path("training_output"),
    'embeddings_cache': Path("training_output") / "embeddings_cache",
    'data_splits': Path("training_output") / "data_splits.npz",
    'fireprot_csv': Path("fireprotdb_with_sequences.csv"),
    'embeddings_train': Path("training_output") / "embeddings_train.npz",
    'embeddings_val': Path("training_output") / "embeddings_val.npz",
    'embeddings_test': Path("training_output") / "embeddings_test.npz",
}
"""
Default file paths for data and model files.

All paths are relative to the repository root. These can be overridden
in function calls, but provide sensible defaults for the standard
project structure.

Paths:
- training_output: Directory containing trained models and embeddings
- embeddings_cache: Cache directory for computed embeddings
- data_splits: File containing train/val/test split indices
- fireprot_csv: FireProtDB dataset CSV file
- embeddings_*: Precomputed embedding files for each split
"""

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

MODEL_HYPERPARAMETERS: Dict[str, Dict] = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    },
    'mlp': {
        'input_dim': EMBEDDING_DIMENSIONS['total_combined'],
        'hidden_dims': [1024, 512, 128],
        'dropout': 0.2,
        'learning_rate': 1e-4,
        'weight_decay': 1e-3,
        'batch_size': 128,
        'max_epochs': 150,
        'early_stopping_patience': 10,
        'loss': 'mae'
    }
}
"""
Model hyperparameters for baseline models.

These hyperparameters define the baseline model configurations.
All models use these exact settings for reproducibility.

Random Forest:
- 100 decision trees
- No maximum depth limit
- Minimum 2 samples to split, 1 sample per leaf

XGBoost:
- 100 boosting rounds
- Maximum depth 6
- Learning rate 0.1
- 80% subsample and column sample

MLP:
- Input: 2344 dimensions
- Hidden layers: 1024 → 512 → 128
- Dropout: 0.2 after first two layers
- AdamW optimizer with lr=1e-4, weight_decay=1e-3
- Early stopping with 10 epochs patience
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

SEQUENCE_LENGTH_LIMITS: Dict[str, int] = {
    'min_length': 10,
    'max_length': 5000
}
"""
Sequence length validation limits.

Sequences outside this range are filtered during preprocessing.
These limits match the capabilities of the embedding models:
- ProtT5-XL: max_length=512 (truncated if longer)
- ESM-2: max_length=1024 (truncated if longer)
"""

DATA_SPLIT_RATIOS: Dict[str, float] = {
    'train': 0.7,
    'validation': 0.2,
    'test': 0.1
}
"""
Data split ratios for train/validation/test sets.

These ratios are used when creating new splits. The actual splits
used in experiments are stored in data_splits.npz for consistency.
"""

TARGET_COLUMN_NAME: str = "DDG"
"""
Default target column name in CSV files.

This is the column containing ΔΔG values (in kcal/mol).
Can be overridden in function calls.
"""

SEQUENCE_COLUMN_NAME: str = "sequence"
"""
Default sequence column name in CSV files.

This is the column containing protein sequences (amino acid strings).
Can be overridden in function calls.
"""

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================

PROTT5_MODEL_NAME: str = "Rostlab/prot_t5_xl_uniref50"
"""ProtT5-XL model identifier for Hugging Face transformers."""

ESM2_MODEL_NAME: str = "facebook/esm2_t33_650M_UR50D"
"""ESM-2 650M model identifier for Hugging Face transformers."""

EMBEDDING_MAX_LENGTHS: Dict[str, int] = {
    'prott5': 512,
    'esm2': 1024
}
"""
Maximum sequence lengths for embedding models.

Sequences longer than these limits are truncated during tokenization.
These match the model's maximum context window.
"""

# ============================================================================
# EVALUATION METRICS
# ============================================================================

PRIMARY_METRIC: str = "mae"
"""
Primary metric for model evaluation and selection.

MAE (Mean Absolute Error) is used as the primary metric because:
1. Directly interpretable in kcal/mol
2. Robust to outliers
3. Appropriate for regression tasks
"""

SECONDARY_METRICS: list = ["rmse", "r2", "pearson_r", "spearman_r"]
"""
Secondary metrics for comprehensive evaluation.

These metrics provide additional information about model performance:
- RMSE: Penalizes large errors more than MAE
- R²: Explained variance
- Pearson r: Linear correlation
- Spearman r: Rank correlation (robust to non-linearities)
"""

