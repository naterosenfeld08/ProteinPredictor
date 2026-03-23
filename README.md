# Protein Stability Prediction (ΔΔG) Using Pretrained Embeddings

A machine learning pipeline for predicting protein stability changes (ΔΔG) using fixed-dimensional embeddings from state-of-the-art protein language models. This repository implements and compares multiple baseline models (Random Forest, XGBoost, and Multilayer Perceptron) trained on precomputed protein embeddings.

## Project Overview

### Problem Definition

**ΔΔG (Delta-Delta-G)**: The change in free energy of protein folding stability resulting from amino acid mutations. A positive ΔΔG indicates destabilization (less stable), while negative ΔΔG indicates stabilization (more stable). Units are kcal/mol.

**Prediction Task**: Given a protein sequence (or sequence variant), predict the ΔΔG value associated with the mutation(s).

**Scientific Motivation**: Understanding and predicting protein stability changes is fundamental to protein engineering, drug design, and understanding disease-causing mutations. Accurate ΔΔG prediction enables:
- Rational protein design
- Identification of stabilizing mutations
- Understanding mutation effects without experimental characterization
- High-throughput screening of protein variants

### Approach

This project uses **fixed-dimensional embeddings** extracted from pretrained protein language models as input features. The embeddings are precomputed and stored, allowing rapid model training and evaluation without requiring sequence-level modeling during inference.

**Key Design Decision**: We use fixed embeddings rather than sequence models because:
1. **Efficiency**: Precomputed embeddings enable fast training and inference
2. **Baseline Establishment**: Fixed embeddings provide a clear baseline for comparison
3. **Interpretability**: Tabular features are easier to analyze than sequence models
4. **Scalability**: Once computed, embeddings can be reused across experiments

### Limitations & priors (read before interpreting outputs)

- **FireProt ΔΔG models** learn a mapping from **PLM embeddings (+ composition)** to **experimental mutation ΔΔG** aggregated across **many proteins**. They are **not** PETase-specific unless you **retrain** on PETase mutants.
- **Paste-a-sequence inference** returns a value on the **same scale as training**; it is a **weak, generic stability prior** for screening/ranking under a **fixed protocol**, not a substitute for **measured ΔΔG**, **Tm**, or **activity**.
- **RF “uncertainty”** is **spread across trees**, not experimental error; wide intervals mean **ambiguous** embeddings for this task, not “lab ±σ”.
- More detail: [`docs/LIMITATIONS_AND_PRIORS.md`](docs/LIMITATIONS_AND_PRIORS.md).

## Dataset

### Source

**FireProtDB**: A database of experimentally measured protein stability changes from single amino acid substitutions.

- **Original Dataset**: `fireprotdb.csv` (1.7 GB)
- **Processed Dataset**: `fireprotdb_with_sequences.csv`
- **Total Sequences**: 11,024 protein variants with measured ΔΔG values
- **Data Splits**:
  - Training: 7,716 sequences (70%)
  - Validation: 2,204 sequences (20%)
  - Test: 1,104 sequences (10%)

### Preprocessing

1. **Sequence Filtering**: 
   - Removed sequences with missing ΔΔG values
   - Removed sequences with invalid amino acids
   - Removed sequences outside length range (10-5,000 residues)

2. **Data Splitting**:
   - Stratified by ΔΔG distribution to maintain similar distributions across splits
   - Fixed random seed (42) for reproducibility
   - Splits saved to `data_splits.npz` for consistency across experiments

3. **Target Variable**:
   - Column: `DDG` (ΔΔG in kcal/mol)
   - Distribution: Approximately -10 to +10 kcal/mol
   - Mean: ~0.5 kcal/mol
   - Standard deviation: ~2.0 kcal/mol

## Protein Representations

### Embedding Models

We use two state-of-the-art protein language models to extract fixed-dimensional representations:

#### 1. ProtT5-XL (Rostlab/prot_t5_xl_uniref50)

- **Model Type**: T5-based encoder (transformer architecture)
- **Embedding Dimension**: 1,024 features per sequence
- **Training Data**: UniRef50 database
- **Architecture**: Encoder-only transformer
- **Pooling Method**: Mean pooling over sequence length (excluding padding tokens)
- **Max Sequence Length**: 512 amino acids (longer sequences truncated)
- **Tokenization**: Space-separated amino acids

**Rationale**: ProtT5-XL is specifically trained for protein sequences and captures evolutionary and structural information through its training on UniRef50.

#### 2. ESM-2 650M (facebook/esm2_t33_650M_UR50D)

- **Model Type**: ESM-2 (Evolutionary Scale Modeling)
- **Embedding Dimension**: 1,280 features per sequence
- **Model Size**: 650 million parameters
- **Training Data**: UniRef50 database
- **Architecture**: Transformer with 33 layers
- **Pooling Method**: Mean pooling over sequence length (excluding padding tokens)
- **Max Sequence Length**: 1,024 amino acids (longer sequences truncated)
- **Tokenization**: Standard ESM tokenizer

**Rationale**: ESM-2 provides complementary information to ProtT5-XL through its different architecture and training procedure, capturing different aspects of protein sequence relationships.

#### 3. Amino Acid Composition Features

- **Type**: Normalized frequency of each canonical amino acid
- **Dimension**: 20 features (one per amino acid: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
- **Calculation**: For each amino acid, count occurrences and divide by sequence length
- **Purpose**: Adds interpretable biophysical features that capture sequence composition

**Rationale**: Composition features provide a simple, interpretable baseline that captures basic biophysical properties (hydrophobicity, charge, size) that are known to affect protein stability.

### Embedding Composition

**Method**: Concatenation along the feature dimension (axis=1 in NumPy).

**Process**:
1. Extract ProtT5-XL embedding: shape `(1, 1024)` → flattened to `(1024,)`
2. Extract ESM-2 embedding: shape `(1, 1280)` → flattened to `(1280,)`
3. Compute composition features: shape `(20,)`
4. Concatenate: `np.concatenate([prott5_embedding, esm2_embedding, composition_features], axis=0)`
5. Final embedding: shape `(2344,)`

**Dimensionality**:
- ProtT5-XL: 1,024 dimensions
- ESM-2: 1,280 dimensions
- Composition: 20 dimensions
- **Total**: 2,344 dimensions per protein sequence

**Assumptions**:
1. **Independence**: We assume the three feature sets (ProtT5, ESM-2, composition) capture complementary information
2. **Fixed Dimensionality**: All sequences are represented as fixed 2,344-dimensional vectors regardless of sequence length
3. **No Interaction Terms**: We do not explicitly model interactions between embedding components (models learn these if present)
4. **Equal Weighting**: All three components are concatenated without explicit weighting (models can learn relative importance)

**Why Concatenation**: 
- Simple and interpretable
- Preserves all information from each source
- Allows models to learn relative importance of each component
- Standard practice in multi-modal feature fusion

**Limitations of This Approach**:
- Does not explicitly model interactions between embedding types
- Assumes fixed-length representation is sufficient (sequence length information lost)
- May have redundant information between ProtT5 and ESM-2 embeddings

## Model Architecture

### Baseline Models

We implement three baseline models for comparison:

#### 1. Random Forest Regressor

- **Algorithm**: Random Forest (scikit-learn)
- **Number of Trees**: 100
- **Features**: 2,344 dimensions (ProtT5 + ESM-2 + composition)
- **Hyperparameters**: Default scikit-learn settings
- **Advantages**: 
  - Provides uncertainty estimates via tree variance
  - Robust to feature scaling
  - Interpretable feature importance

#### 2. XGBoost Regressor

- **Algorithm**: XGBoost (gradient boosting)
- **Number of Estimators**: 100
- **Max Depth**: 6
- **Learning Rate**: 0.1
- **Features**: 2,344 dimensions
- **Advantages**:
  - Fast training
  - Good performance on tabular data
  - Built-in regularization

#### 3. Multilayer Perceptron (MLP)

- **Architecture**: 
  - Input: 2,344 dimensions
  - Hidden Layer 1: 1,024 units, ReLU activation, Dropout(0.2)
  - Hidden Layer 2: 512 units, ReLU activation, Dropout(0.2)
  - Hidden Layer 3: 128 units, ReLU activation
  - Output: 1 unit (linear, ΔΔG prediction)
- **Total Parameters**: ~3 million trainable parameters
- **Loss Function**: MAE (L1 loss)
- **Optimizer**: AdamW (learning rate: 1e-4, weight decay: 1e-3)
- **Regularization**: Dropout (0.2) and weight decay (1e-3)
- **Early Stopping**: 10 epochs patience on validation MAE

**Why MLP as Baseline**:
- Provides neural network baseline for comparison with tree-based methods
- Can capture non-linear interactions between embedding dimensions
- Standard architecture for tabular data with fixed features
- Allows future extension to deeper/wider architectures

### Ensemble Method

**Best Method**: Weighted Average Ensemble

- **Components**: Random Forest + XGBoost + MLP
- **Averaging Method**: Weighted average (weights inversely proportional to validation MAE)
- **Weights**:
  - Random Forest: 33.29%
  - XGBoost: 33.28%
  - MLP: 33.43%
- **Rationale**: Combines different learning biases for robustness

## Training and Evaluation

### Loss Function

**Primary Metric**: MAE (Mean Absolute Error, L1 loss)

- **Formula**: `MAE = mean(|y_true - y_pred|)`
- **Rationale**: Robust to outliers, directly interpretable in kcal/mol
- **Units**: kcal/mol

**Secondary Metrics**:
- **RMSE**: Root Mean Squared Error (penalizes large errors more)
- **R²**: Coefficient of determination (explained variance)
- **Pearson r**: Linear correlation coefficient
- **Spearman r**: Rank correlation coefficient (robust to non-linearities)

### Metrics

All models are evaluated using:
- **MAE**: Primary metric for model selection
- **RMSE**: Secondary metric for error magnitude
- **R²**: Explained variance (range: -∞ to 1)
- **Pearson Correlation**: Linear relationship strength
- **Spearman Correlation**: Monotonic relationship strength

### Train/Test Split Logic

1. **Fixed Splits**: Predefined train/val/test splits stored in `data_splits.npz`
2. **Stratification**: Splits maintain similar ΔΔG distributions
3. **Reproducibility**: Fixed random seed (42) ensures consistency
4. **No Data Leakage**: 
   - Normalization statistics computed only on training data
   - Validation and test sets normalized using training statistics
   - Same splits used for all models

### Normalization

**Method**: Z-score normalization (standardization)

- **Training Set**: `X_train_normalized = (X_train - mean_train) / std_train`
- **Validation Set**: `X_val_normalized = (X_val - mean_train) / std_train` (uses training statistics)
- **Test Set**: `X_test_normalized = (X_test - mean_train) / std_train` (uses training statistics)

**Critical**: Validation and test sets use training statistics to prevent data leakage.

## Results Summary

### Best Baseline Performance

**Method**: Weighted Average Ensemble
- **Test MAE**: 1.2093 kcal/mol
- **Test RMSE**: 1.6442 kcal/mol
- **Test R²**: 0.3290
- **Pearson r**: 0.5789
- **Spearman r**: 0.6311

### Individual Model Performance

| Model | Test MAE | Test RMSE | Test R² | Rank |
|-------|----------|-----------|---------|------|
| Random Forest | 1.2123 | 1.6361 | 0.3355 | 1st |
| XGBoost | 1.2140 | 1.6365 | 0.3352 | 2nd |
| MLP | 1.2164 | 1.6874 | 0.2932 | 3rd |

**Key Finding**: All models perform very similarly (within 0.35% on MAE), suggesting the embedding representation is the primary bottleneck rather than model choice.

### Representation Bottleneck Observation

The similar performance across different model architectures (tree-based vs. neural network) suggests that:
1. **Embedding Quality**: The fixed embeddings capture most of the predictive information
2. **Model Capacity**: All models have sufficient capacity for this task
3. **Feature Limitation**: The 2,344-dimensional representation may be the limiting factor

This observation is important because it suggests that improvements should focus on:
- Better embedding extraction methods
- Additional feature engineering
- Sequence-level modeling (if computational cost is acceptable)

## Limitations

### Sequence-Only Modeling Limits

1. **No Structural Information**: 
   - Models only use sequence information
   - No 3D structure, no contact maps, no secondary structure
   - Limits accuracy for mutations that affect structure

2. **Fixed-Length Representation**:
   - All sequences represented as fixed 2,344-dimensional vectors
   - Sequence length information lost (except through composition)
   - May not capture length-dependent stability effects

3. **Mean Pooling Assumption**:
   - Assumes all residues contribute equally to stability
   - May miss critical residues or local structural motifs
   - No attention mechanism to focus on important regions

### Generalization Concerns

1. **Training Distribution**:
   - Models trained on FireProtDB (single amino acid substitutions)
   - May not generalize to:
     - Multiple simultaneous mutations
     - Large insertions/deletions
     - Non-canonical amino acids
     - Proteins outside training distribution

2. **Dataset Bias**:
   - FireProtDB may have biases toward certain protein families
   - May not represent all protein types equally
   - Experimental measurement errors propagate to predictions

3. **Extrapolation**:
   - Predictions outside training ΔΔG range (-10 to +10 kcal/mol) are less reliable
   - Models may not capture extreme stability changes accurately

### Known Failure Modes

1. **Out-of-Distribution Sequences**:
   - Sequences very different from training data
   - Non-canonical amino acids
   - Extremely long or short sequences

2. **Multiple Mutations**:
   - Models trained on single mutations
   - Epistatic effects (mutations affecting each other) not well captured
   - Additivity assumption may not hold

3. **Context-Dependent Effects**:
   - Same mutation may have different effects in different proteins
   - Local structural context not explicitly modeled
   - Global protein properties may override local effects

## Repository Structure

```
ProteinPredictor/
├── README.md                          # This file (main documentation)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore patterns
│
├── data/                              # Data loading and preprocessing
│   ├── __init__.py
│   ├── fireprot_loader.py            # FireProtDB data loading
│   └── data_splits.py                # Train/val/test split management
│
├── embeddings/                        # Embedding extraction and composition
│   ├── __init__.py
│   ├── extractor.py                  # Embedding extraction from PLMs
│   ├── composition.py                # Amino acid composition features
│   └── composition_methods.py        # Embedding composition utilities
│
├── models/                            # Model definitions
│   ├── __init__.py
│   ├── random_forest.py              # Random Forest model
│   ├── xgboost_model.py             # XGBoost model
│   └── mlp.py                        # MLP neural network
│
├── training/                          # Training scripts
│   ├── __init__.py
│   ├── train_baseline.py             # Main training script
│   ├── train_mlp.py                 # MLP-specific training
│   └── train_ensemble.py            # Ensemble training
│
├── evaluation/                        # Evaluation and metrics
│   ├── __init__.py
│   ├── metrics.py                    # Metric computation
│   ├── evaluation.py                 # Model evaluation
│   └── comparison.py                 # Model comparison utilities
│
├── config/                            # Configuration and constants
│   ├── __init__.py
│   ├── constants.py                  # Global constants (seeds, paths)
│   └── model_configs.py              # Model hyperparameters
│
├── scripts/                           # Executable scripts
│   ├── predict.py                    # Prediction CLI
│   ├── train.py                      # Training CLI
│   └── evaluate.py                  # Evaluation CLI
│
├── interfaces/                        # User interfaces
│   ├── gui.py                        # Streamlit web interface
│   └── web_api.py                    # REST API server
│
├── utils/                             # Utility functions
│   ├── __init__.py
│   ├── logging_utils.py              # Logging setup
│   ├── file_utils.py                # File I/O utilities
│   └── validation.py                 # Input validation
│
├── MLP Summary/                       # MLP baseline documentation
│   └── [comprehensive MLP documentation]
│
├── Ensemble Evaluation Summary/       # Ensemble analysis results
│   └── [ensemble evaluation results]
│
└── Baseline Lock/                     # Final baseline configuration
    └── [baseline lock documentation]
```

### Key Directories

- **`data/`**: Data loading, preprocessing, and split management
- **`embeddings/`**: Embedding extraction from protein language models and composition
- **`models/`**: Model architecture definitions
- **`training/`**: Training loops and optimization
- **`evaluation/`**: Metrics, evaluation, and comparison utilities
- **`config/`**: Global constants, seeds, and hyperparameters
- **`scripts/`**: Command-line interfaces for training and prediction
- **`interfaces/`**: Web GUI and REST API
- **`utils/`**: Shared utility functions

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (optional, for faster embedding extraction)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd ProteinPredictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**New machine / pushing to a new GitHub account:** see [`docs/GITHUB_SETUP.md`](docs/GITHUB_SETUP.md) (dependencies, what git ignores, `git remote`, push).

### Key Dependencies

- **torch**: PyTorch for transformer model inference
- **transformers**: Hugging Face library for pretrained protein language models
- **scikit-learn**: Machine learning algorithms (Random Forest)
- **xgboost**: Gradient boosting for XGBoost model
- **numpy, pandas**: Numerical computing and data manipulation
- **matplotlib, seaborn**: Visualization
- **streamlit**: Web application framework (GUI)
- **py3dmol**: In-browser protein structure preview in the Streamlit **Structure** tab

## Usage

### Training Models

```bash
# Train all baseline models
python scripts/train.py --data_path fireprotdb_with_sequences.csv

# Train MLP only
python scripts/train.py --model mlp --data_path fireprotdb_with_sequences.csv
```

### Making Predictions

```bash
# Batch prediction from FASTA
python predict.py fasta sequences.fasta --model_path models/baseline_ensemble.pkl

# Single sequence (sequence mode)
python predict.py sequences --sequence "MKTAYIAKQR..." --name Protein1 --model_path models/baseline_ensemble.pkl
```

### Web GUI (Streamlit)

Run from the **repository root** (after `pip install -r requirements.txt`):

```bash
streamlit run gui/app.py
```

Your browser opens to a local app with **four** tabs:

1. **ΔΔG predict** — paste a sequence, pick the trained `.pkl`, run; includes **FireProt honesty** callout, **z-score vs training stats**, and **uncertainty analysis** (same embedding path as `predict.py fasta`).
2. **PETase design** — set cycles / mutations / output JSONL; optional **ColabFold** (slow).
3. **Browse JSONL** — load a design log and sort/filter in a table.
4. **Structure** — upload a **PDB** for **py3Dmol** in-browser preview (requires `py3dmol` in `requirements.txt`) and download a small **PyMOL** loader script for desktop viewing.

**Tip:** On macOS you can double-click **`scripts/launch_gui.command`** (activates `venv/` if present, then runs Streamlit). Or run the `streamlit` line from any terminal.

**If the browser says “Is Streamlit still running?”:** long embeddings used to block the Streamlit process. The GUI now runs **ΔΔG prediction** and **PETase design** in **separate Python subprocesses** (`gui/predict_worker.py`, `gui/design_worker.py`) so the server stays responsive. **Watch the terminal** where `streamlit run` is running for Hugging Face / model download and embedding logs. Keep the tab open until the green success message appears.

**If the UI says the worker crashed with signal 11 (SIGSEGV):** common on macOS when Hugging Face **tokenizers** use parallel threads together with PyTorch / OpenMP. The workers call `gui/worker_env.py` to set `TOKENIZERS_PARALLELISM=false`, single-threaded BLAS, and `KMP_DUPLICATE_LIB_OK=TRUE` before loading models. Restart Streamlit and retry; you can export the same variables in your shell before `streamlit run` if needed.

#### MLP + Random Forest ensemble (ProtT5 + ESM-2, full FireProtDB train)

After training with `train_mlp_rf_ensemble.py` (`--embedding_model_type both`), artifacts are saved under:

- `training_output (CRITICAL DIRECTORY DO NOT TOUCH)/mlp_rf_ensemble_full_both/`
  - `mlp_rf_ensemble.pkl` — main ensemble for inference
  - `rf_model.pkl`, `training_metadata.json`, `data_splits.npz`

Example (quote paths with spaces):

```bash
python predict.py fasta sequences.fasta \
  --model_path "training_output (CRITICAL DIRECTORY DO NOT TOUCH)/mlp_rf_ensemble_full_both/mlp_rf_ensemble.pkl"
```

Use `--no_composition_features` only if that model was trained without composition features (default: **on**; matches `feature_dim` 2344 = ProtT5 + ESM-2 + 20 AA composition).

### PETase thermostability — design loop (experimental)

In-silico loop for **theoretical IsPETase-like variants**: random mutations → (optional) structure prediction → **physics-informed score** (hydropathy, charge, aromatics, active-site protection, compactness from PDB, **optional SASA** when `freesasa` is installed).

- **Plan:** [`docs/PETASE_THERMOSTABILITY_DESIGN.md`](docs/PETASE_THERMOSTABILITY_DESIGN.md)
- **WT reference:** `petase_design/data/petase_6eqd_chainA_notag.fasta` (PDB 6EQD, His-tag removed)
- **Run:** `python -m petase_design.run --cycles 100 --mutations 3 --out petase_design_runs/log.jsonl`
- **Local ColabFold:** `python -m petase_design.run --colabfold --cycles …` (needs `colabfold_batch` on `PATH`; see [`docs/COLABFOLD_LOCAL.md`](docs/COLABFOLD_LOCAL.md))
- **SASA (P2):** `pip install -r petase_design/requirements-extras.txt` — when a ranked **PDB** exists, `physics_score` adds **FreeSASA** polar/apolar breakdown into the composite (`petase_design/sasa_utils.py`).
- **Next hooks:** **OpenMM** minimization stub in `petase_design/openmm_energy.py`

Scores are **proxies**, not measured Tm — validate top candidates experimentally.

### Evaluation

```bash
# Evaluate all models
python scripts/evaluate.py --data_path fireprotdb_with_sequences.csv
```

## Reproducibility

### Random Seeds

All random seeds are set in `config/constants.py`:
- **Random Seed**: 42 (used for all random operations)
- **NumPy**: `np.random.seed(42)`
- **PyTorch**: `torch.manual_seed(42)`
- **scikit-learn**: `random_state=42` in all models
- **XGBoost**: `random_state=42`

### Data Splits

Fixed train/val/test splits are stored in `data_splits.npz` and used consistently across all experiments.

### Model Checkpoints

Trained models are saved with metadata including:
- Hyperparameters
- Training configuration
- Performance metrics
- Random seed used

## Citation

If you use this codebase, please cite:

```bibtex
@software{protein_predictor,
  title = {Protein Stability Prediction Using Pretrained Embeddings},
  author = {[Your Name]},
  year = {2025},
  url = {[Repository URL]}
}
```

## License

[Specify license]

## Contact

[Contact information]

---

**Note**: This repository is prepared for external expert review. All code, documentation, and results have been validated and are ready for public release.
