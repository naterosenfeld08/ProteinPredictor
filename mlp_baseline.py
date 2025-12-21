"""
Baseline Multilayer Perceptron (MLP) for ΔΔG Prediction Using Pretrained Protein Embeddings

This module implements a baseline MLP neural network for predicting protein stability changes (ΔΔG)
using fixed 2,344-dimensional embeddings from ProtT5-XL, ESM-2, and amino acid composition features.

Technical Overview:
- MLP (Multilayer Perceptron): A feedforward neural network with multiple fully connected layers
- Architecture: 2344 → 1024 → 512 → 128 → 1 (with dropout 0.2 after first two hidden layers)
- Loss: MAE (L1 loss) for robust regression
- Optimizer: AdamW with learning rate 1e-4 and weight decay 1e-3
- Early stopping: 10 epochs patience on validation MAE
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. TECHNICAL EXPLANATION: WHAT IS AN MLP?
# ============================================================================

"""
Multilayer Perceptron (MLP) - Technical Definition:

An MLP is a feedforward artificial neural network consisting of:
1. Input Layer: Receives fixed-size feature vectors (2,344 dimensions in our case)
2. Hidden Layers: One or more fully connected (dense) layers with non-linear activations
3. Output Layer: Produces predictions (1 dimension for regression)

Mathematical Formulation:
For a single hidden layer MLP:
    h = ReLU(W₁x + b₁)          # Hidden layer
    y = W₂h + b₂                  # Output layer

Where:
- x ∈ R^d: input vector (d=2344)
- W₁, W₂: weight matrices
- b₁, b₂: bias vectors
- ReLU: Rectified Linear Unit activation (max(0, x))

For multiple hidden layers, this is composed:
    h₁ = ReLU(W₁x + b₁)
    h₂ = ReLU(W₂h₁ + b₂)
    ...
    y = Wₙhₙ₋₁ + bₙ

Key Properties:
- Universal Function Approximator: Can approximate any continuous function given sufficient capacity
- Non-linear: ReLU activations enable learning complex non-linear relationships
- Gradient-based Learning: Parameters updated via backpropagation using gradient descent
- Regularization: Dropout and weight decay prevent overfitting

Why MLP for Fixed Embeddings:
1. Embeddings are fixed, pre-computed features (no sequence modeling needed)
2. MLPs excel at learning non-linear mappings from high-dimensional features to targets
3. Can capture complex interactions between embedding dimensions
4. Fast inference compared to sequence models
5. Interpretable architecture (vs. black-box ensemble methods)
"""


# ============================================================================
# 2. DATA LOADING
# ============================================================================

class DDGPredictionDataset(Dataset):
    """
    PyTorch Dataset for protein embeddings and ΔΔG (delta-delta-G) labels.
    
    This dataset provides access to precomputed protein embeddings and their
    corresponding experimentally measured stability change values (ΔΔG).
    
    Embeddings are fixed 2,344-dimensional vectors composed of:
    - ProtT5-XL embeddings: 1,024 dimensions
    - ESM-2 embeddings: 1,280 dimensions
    - Amino acid composition: 20 dimensions
    
    Labels are ΔΔG values in kcal/mol, where positive values indicate
    destabilization and negative values indicate stabilization.
    """
    
    def __init__(self, embeddings_array: np.ndarray, ddg_targets: np.ndarray):
        """
        Initialize dataset with embeddings and ΔΔG targets.
        
        Args:
            embeddings_array: numpy.ndarray of shape (n_samples, 2344)
                             Precomputed protein embeddings. Each row is a
                             fixed 2,344-dimensional feature vector for one protein.
            ddg_targets: numpy.ndarray of shape (n_samples,)
                        Experimentally measured ΔΔG values in kcal/mol.
                        Positive values = destabilization, negative = stabilization.
        
        Raises:
            ValueError: If embeddings and labels have different lengths.
            ValueError: If embeddings are not 2,344-dimensional.
        
        Assumptions:
            - Embeddings are already normalized (if normalization is used)
            - Labels are valid floating-point numbers (NaN values should be removed)
            - Embeddings and labels are aligned (same indexing)
        """
        self.embeddings_tensor = torch.FloatTensor(embeddings_array)
        self.ddg_targets_tensor = torch.FloatTensor(ddg_targets)
        
        if len(self.embeddings_tensor) != len(self.ddg_targets_tensor):
            raise ValueError(
                f"Embeddings ({len(self.embeddings_tensor)}) and labels "
                f"({len(self.ddg_targets_tensor)}) must have same length"
            )
        
        expected_dim = 2344
        if self.embeddings_tensor.shape[1] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim}-dimensional embeddings, "
                f"got {self.embeddings_tensor.shape[1]}"
            )
    
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of protein sequences in the dataset.
        """
        return len(self.embeddings_tensor)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample (embedding, label) pair.
        
        Args:
            idx: Index of the sample to retrieve (0-indexed).
        
        Returns:
            tuple: (embedding_tensor, ddg_target)
                  - embedding_tensor: torch.Tensor of shape (2344,)
                  - ddg_target: torch.Tensor of shape (1,) containing ΔΔG value
        """
        return self.embeddings_tensor[idx], self.ddg_targets_tensor[idx]


def load_embeddings_and_labels(
    embeddings_dir: str = "training_output (CRITICAL DIRECTORY DO NOT TOUCH)",
    csv_path: str = "fireprotdb_with_sequences.csv",
    target_col: str = "DDG"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load precomputed protein embeddings and ΔΔG labels from FireProtDB.
    
    This function loads precomputed 2,344-dimensional embeddings (ProtT5-XL + ESM-2 + composition)
    and their corresponding experimentally measured ΔΔG values. The function performs
    critical data preprocessing including normalization and NaN removal.
    
    CRITICAL DATA LEAKAGE PREVENTION:
    Normalization statistics (mean and standard deviation) are computed ONLY on training data.
    Validation and test sets are normalized using training statistics to prevent information
    leakage from validation/test sets into the model.
    
    Args:
        embeddings_dir: str, path to directory containing precomputed embedding files.
                       Must contain:
                       - embeddings_train.npz (key: 'embeddings', shape: (n_train, 2344))
                       - embeddings_val.npz (key: 'embeddings', shape: (n_val, 2344))
                       - embeddings_test.npz (key: 'embeddings', shape: (n_test, 2344))
                       - data_splits.npz (keys: 'train_indices', 'val_indices', 'test_indices')
        csv_path: str, path to CSV file containing FireProtDB data.
                 Must contain a column with ΔΔG values (default: "DDG").
        target_col: str, name of the column containing ΔΔG values in kcal/mol.
                   Default: "DDG"
    
    Returns:
        Tuple of 6 numpy arrays:
        - X_train: numpy.ndarray of shape (n_train_samples, 2344)
                  Normalized training embeddings
        - X_val: numpy.ndarray of shape (n_val_samples, 2344)
                Normalized validation embeddings (using training statistics)
        - X_test: numpy.ndarray of shape (n_test_samples, 2344)
                 Normalized test embeddings (using training statistics)
        - y_train: numpy.ndarray of shape (n_train_samples,)
                  Training ΔΔG targets in kcal/mol (NaN values removed)
        - y_val: numpy.ndarray of shape (n_val_samples,)
                Validation ΔΔG targets in kcal/mol (NaN values removed)
        - y_test: numpy.ndarray of shape (n_test_samples,)
                 Test ΔΔG targets in kcal/mol (NaN values removed)
    
    Raises:
        FileNotFoundError: If embedding files or CSV file not found.
        ValueError: If target column not found in CSV.
        ValueError: If embedding dimensionality is not 2344.
    
    Assumptions:
        1. Embedding files exist and contain 'embeddings' key
        2. Data splits file exists and contains correct indices
        3. CSV file contains target column with numeric values
        4. Embeddings and CSV rows are aligned by index
        5. NaN values in targets indicate missing data (removed)
    
    Normalization Process:
        1. Compute mean and std from X_train only
        2. Normalize X_train: (X_train - mean) / std
        3. Normalize X_val: (X_val - mean) / std  (uses training statistics)
        4. Normalize X_test: (X_test - mean) / std  (uses training statistics)
    
    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test = load_embeddings_and_labels()
        >>> print(X_train.shape, y_train.shape)
        (7716, 2344) (7716,)
    """
    embeddings_dir = Path(embeddings_dir)
    csv_path = Path(csv_path)
    
    print("="*60)
    print("LOADING EMBEDDINGS AND LABELS")
    print("="*60)
    
    # Load embeddings from .npz files
    print("\nLoading embeddings from .npz files...")
    train_emb = np.load(embeddings_dir / "embeddings_train.npz")
    val_emb = np.load(embeddings_dir / "embeddings_val.npz")
    test_emb = np.load(embeddings_dir / "embeddings_test.npz")
    
    # Load embedding arrays from .npz files
    # Each file contains a single key 'embeddings' with shape (n_samples, 2344)
    train_embeddings_array = train_emb['embeddings']
    val_embeddings_array = val_emb['embeddings']
    test_embeddings_array = test_emb['embeddings']
    
    print(f"  Training embeddings: {train_embeddings_array.shape}")
    print(f"  Validation embeddings: {val_embeddings_array.shape}")
    print(f"  Test embeddings: {test_embeddings_array.shape}")
    
    # Verify embedding dimensionality (must be 2344: ProtT5=1024 + ESM2=1280 + Composition=20)
    expected_embedding_dim = 2344
    if train_embeddings_array.shape[1] != expected_embedding_dim:
        raise ValueError(
            f"Expected {expected_embedding_dim}-dimensional embeddings "
            f"(ProtT5=1024 + ESM2=1280 + Composition=20), got {train_embeddings_array.shape[1]}"
        )
    
    # Load data splits to get correct row indices for aligning embeddings with labels
    print("\nLoading data splits...")
    splits_file = np.load(embeddings_dir / "data_splits.npz")
    train_row_indices = splits_file['train_indices']
    val_row_indices = splits_file['val_indices']
    test_row_indices = splits_file['test_indices']
    
    print(f"  Train indices: {len(train_row_indices)}")
    print(f"  Val indices: {len(val_row_indices)}")
    print(f"  Test indices: {len(test_row_indices)}")
    
    # Load target values (ΔΔG) from CSV file
    print(f"\nLoading target values from {csv_path}...")
    fireprot_dataframe = pd.read_csv(csv_path, low_memory=False)
    
    if target_col not in fireprot_dataframe.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in CSV. "
            f"Available columns: {list(fireprot_dataframe.columns)}"
        )
    
    # Extract ΔΔG labels for each split (aligned by row index)
    # Labels are in kcal/mol: positive = destabilization, negative = stabilization
    ddg_train = fireprot_dataframe.iloc[train_row_indices][target_col].values
    ddg_val = fireprot_dataframe.iloc[val_row_indices][target_col].values
    ddg_test = fireprot_dataframe.iloc[test_row_indices][target_col].values
    
    # Remove samples with NaN ΔΔG values
    # CRITICAL: Must remove from both embeddings and labels together to maintain alignment
    train_valid_mask = ~np.isnan(ddg_train)
    val_valid_mask = ~np.isnan(ddg_val)
    test_valid_mask = ~np.isnan(ddg_test)
    
    train_embeddings_array = train_embeddings_array[train_valid_mask]
    ddg_train = ddg_train[train_valid_mask]
    val_embeddings_array = val_embeddings_array[val_valid_mask]
    ddg_val = ddg_val[val_valid_mask]
    test_embeddings_array = test_embeddings_array[test_valid_mask]
    ddg_test = ddg_test[test_valid_mask]
    
    print(f"\nAfter removing NaN:")
    print(f"  Training: {len(ddg_train)} samples")
    print(f"  Validation: {len(ddg_val)} samples")
    print(f"  Test: {len(ddg_test)} samples")
    
    # Normalize features using training statistics ONLY (prevent data leakage)
    # This ensures validation and test sets don't leak information through normalization
    print("\nNormalizing features using training statistics only...")
    train_embedding_mean = train_embeddings_array.mean(axis=0)
    train_embedding_std = train_embeddings_array.std(axis=0) + 1e-8  # Epsilon prevents division by zero
    
    # Normalize all sets using training statistics
    train_embeddings_normalized = (train_embeddings_array - train_embedding_mean) / train_embedding_std
    val_embeddings_normalized = (val_embeddings_array - train_embedding_mean) / train_embedding_std
    test_embeddings_normalized = (test_embeddings_array - train_embedding_mean) / train_embedding_std
    
    print("  Normalization complete (using training mean/std only)")
    
    return (
        train_embeddings_normalized,
        val_embeddings_normalized,
        test_embeddings_normalized,
        ddg_train,
        ddg_val,
        ddg_test
    )


# ============================================================================
# 3. MLP ARCHITECTURE
# ============================================================================

class BaselineMLP(nn.Module):
    """
    Baseline Multilayer Perceptron for ΔΔG prediction.
    
    Architecture:
        Input: 2344 units
        Hidden 1: 1024 units, ReLU, Dropout(0.2)
        Hidden 2: 512 units, ReLU, Dropout(0.2)
        Hidden 3: 128 units, ReLU
        Output: 1 unit (ΔΔG prediction)
    
    No batch normalization (as per requirements).
    """
    
    def __init__(self, input_dim: int = 2344, dropout: float = 0.2):
        """
        Args:
            input_dim: Input feature dimensionality (default: 2344)
            dropout: Dropout probability (default: 0.2)
        """
        super(BaselineMLP, self).__init__()
        
        self.input_dim = input_dim
        self.dropout = dropout
        
        # Hidden Layer 1: 2344 → 1024
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Hidden Layer 2: 1024 → 512
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Hidden Layer 3: 512 → 128
        self.fc3 = nn.Linear(512, 128)
        self.relu3 = nn.ReLU()
        
        # Output Layer: 128 → 1
        self.fc4 = nn.Linear(128, 1)
    
    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            feature_tensor: Input tensor of shape (batch_size, 2344)
                           Precomputed protein embeddings. Each row is a
                           2,344-dimensional feature vector.
        
        Returns:
            torch.Tensor of shape (batch_size, 1): Predicted ΔΔG values in kcal/mol.
            Positive values indicate predicted destabilization,
            negative values indicate predicted stabilization.
        
        Assumptions:
            - Input features are normalized (if normalization is used during training)
            - Input features have the same distribution as training data
        """
        # Layer 1
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.relu3(x)
        
        # Output
        x = self.fc4(x)
        
        return x
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# 4. TRAINING FUNCTION
# ============================================================================

def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 150,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    early_stopping_patience: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Train the MLP model with early stopping.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch.device (CPU or CUDA)
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for AdamW optimizer
        weight_decay: Weight decay (L2 regularization) coefficient
        early_stopping_patience: Number of epochs to wait before early stopping
        verbose: Whether to print training progress
        
    Returns:
        Dictionary containing training history and best model state
    """
    # Move model to device
    model = model.to(device)
    
    # Loss function: MAE (L1 loss)
    criterion = nn.L1Loss()
    
    # Optimizer: AdamW with specified hyperparameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING MLP")
        print("="*60)
        print(f"Device: {device}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Total parameters: {model.count_parameters():,}")
        print()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for batch_embeddings, batch_ddg_targets in train_loader:
            # Move batch to device (GPU if available, else CPU)
            batch_embeddings = batch_embeddings.to(device)
            batch_ddg_targets = batch_ddg_targets.to(device).unsqueeze(1)  # Shape: (batch_size, 1)
            
            # Forward pass: predict ΔΔG from embeddings
            optimizer.zero_grad()
            predicted_ddg = model(batch_embeddings)
            loss = criterion(predicted_ddg, batch_ddg_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_embeddings)
            train_count += len(batch_embeddings)
        
        avg_train_loss = train_loss / train_count
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch_embeddings, batch_ddg_targets in val_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_ddg_targets = batch_ddg_targets.to(device).unsqueeze(1)
                
                predicted_ddg = model(batch_embeddings)
                loss = criterion(predicted_ddg, batch_ddg_targets)
                
                val_loss += loss.item() * len(batch_embeddings)
                val_count += len(batch_embeddings)
        
        avg_val_loss = val_loss / val_count
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['epoch'].append(epoch + 1)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train MAE: {avg_train_loss:.4f} | "
                  f"Val MAE: {avg_val_loss:.4f} | "
                  f"Best Val MAE: {best_val_loss:.4f} | "
                  f"Patience: {patience_counter}/{early_stopping_patience}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best validation MAE: {best_val_loss:.4f}")
            break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    history['best_val_loss'] = best_val_loss
    history['best_epoch'] = epoch + 1 - patience_counter
    
    return history


# ============================================================================
# 5. EVALUATION FUNCTION
# ============================================================================

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model and compute comprehensive metrics.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation data
        device: torch.device
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_embeddings, batch_ddg_targets in data_loader:
            batch_embeddings = batch_embeddings.to(device)
            predicted_ddg = model(batch_embeddings)
            
            all_predictions.append(predicted_ddg.cpu().numpy())
            all_labels.append(batch_ddg_targets.numpy())
    
    # Concatenate all predictions and labels
    predictions = np.concatenate(all_predictions).flatten()
    labels = np.concatenate(all_labels).flatten()
    
    # Compute metrics
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    r2 = r2_score(labels, predictions)
    
    # Correlation metrics
    pearson_r, pearson_p = pearsonr(labels, predictions)
    spearman_r, spearman_p = spearmanr(labels, predictions)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Pearson r': pearson_r,
        'Pearson p': pearson_p,
        'Spearman r': spearman_r,
        'Spearman p': spearman_p,
        'n_samples': len(labels)
    }
    
    return metrics, predictions, labels


# ============================================================================
# 6. OVERFITTING DETECTION
# ============================================================================

def detect_overfitting(history: Dict, val_metrics: Dict, test_metrics: Dict) -> Dict:
    """
    Detect overfitting by analyzing validation-test gap.
    
    Overfitting Indicators:
    1. Validation MAE << Test MAE (large gap indicates poor generalization)
    2. Training loss << Validation loss (model memorizing training data)
    3. Validation loss decreasing while test loss increasing (classic overfitting)
    
    Args:
        history: Training history dictionary
        val_metrics: Validation set metrics
        test_metrics: Test set metrics
        
    Returns:
        Dictionary with overfitting analysis
    """
    val_mae = val_metrics['MAE']
    test_mae = test_metrics['MAE']
    gap = test_mae - val_mae
    gap_percent = (gap / val_mae) * 100 if val_mae > 0 else 0
    
    # Check training vs validation loss
    if len(history['train_loss']) > 0 and len(history['val_loss']) > 0:
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        train_val_gap = final_val_loss - final_train_loss
    else:
        train_val_gap = None
    
    analysis = {
        'validation_mae': val_mae,
        'test_mae': test_mae,
        'validation_test_gap': gap,
        'gap_percent': gap_percent,
        'train_val_gap': train_val_gap,
        'overfitting_detected': gap_percent > 15.0,  # Threshold: >15% gap indicates overfitting
        'severity': 'none' if gap_percent < 5 else ('mild' if gap_percent < 15 else 'severe')
    }
    
    return analysis


# ============================================================================
# 7. MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """
    Main training script for baseline MLP.
    
    This script:
    1. Loads precomputed embeddings and labels
    2. Creates data loaders with proper shuffling
    3. Initializes the baseline MLP architecture
    4. Trains the model with early stopping
    5. Evaluates on train, validation, and test sets
    6. Detects overfitting
    7. Saves model and results
    """
    print("="*60)
    print("BASELINE MLP FOR DDG PREDICTION")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load precomputed embeddings and ΔΔG labels
    train_embeddings, val_embeddings, test_embeddings, \
    train_ddg_targets, val_ddg_targets, test_ddg_targets = load_embeddings_and_labels()
    
    # Create PyTorch datasets for training, validation, and test sets
    train_dataset = DDGPredictionDataset(train_embeddings, train_ddg_targets)
    val_dataset = DDGPredictionDataset(val_embeddings, val_ddg_targets)
    test_dataset = DDGPredictionDataset(test_embeddings, test_ddg_targets)
    
    # Create data loaders
    # CRITICAL: Shuffle training data, but NOT labels independently
    batch_size = 128
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffles (embeddings, labels) pairs together
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"\nBatch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    model = BaselineMLP(input_dim=2344, dropout=0.2)
    print(f"\nModel initialized with {model.count_parameters():,} parameters")
    
    # Training hyperparameters
    num_epochs = 150
    learning_rate = 1e-4
    weight_decay = 1e-3
    early_stopping_patience = 10
    
    # Train model
    history = train_mlp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        verbose=True
    )
    
    # Evaluate on all sets
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    train_metrics, train_pred, train_true = evaluate_model(model, train_loader, device)
    val_metrics, val_pred, val_true = evaluate_model(model, val_loader, device)
    test_metrics, test_pred, test_true = evaluate_model(model, test_loader, device)
    
    print("\nTraining Set Metrics:")
    print(f"  MAE: {train_metrics['MAE']:.4f} kcal/mol")
    print(f"  RMSE: {train_metrics['RMSE']:.4f} kcal/mol")
    print(f"  R²: {train_metrics['R²']:.4f}")
    print(f"  Pearson r: {train_metrics['Pearson r']:.4f} (p={train_metrics['Pearson p']:.2e})")
    
    print("\nValidation Set Metrics:")
    print(f"  MAE: {val_metrics['MAE']:.4f} kcal/mol")
    print(f"  RMSE: {val_metrics['RMSE']:.4f} kcal/mol")
    print(f"  R²: {val_metrics['R²']:.4f}")
    print(f"  Pearson r: {val_metrics['Pearson r']:.4f} (p={val_metrics['Pearson p']:.2e})")
    
    print("\nTest Set Metrics:")
    print(f"  MAE: {test_metrics['MAE']:.4f} kcal/mol")
    print(f"  RMSE: {test_metrics['RMSE']:.4f} kcal/mol")
    print(f"  R²: {test_metrics['R²']:.4f}")
    print(f"  Pearson r: {test_metrics['Pearson r']:.4f} (p={test_metrics['Pearson p']:.2e})")
    
    # Overfitting analysis
    print("\n" + "="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)
    
    overfitting_analysis = detect_overfitting(history, val_metrics, test_metrics)
    
    print(f"\nValidation-Test Gap:")
    print(f"  Validation MAE: {overfitting_analysis['validation_mae']:.4f} kcal/mol")
    print(f"  Test MAE: {overfitting_analysis['test_mae']:.4f} kcal/mol")
    print(f"  Gap: {overfitting_analysis['validation_test_gap']:.4f} kcal/mol ({overfitting_analysis['gap_percent']:.2f}%)")
    print(f"  Severity: {overfitting_analysis['severity']}")
    print(f"  Overfitting detected: {overfitting_analysis['overfitting_detected']}")
    
    if overfitting_analysis['train_val_gap'] is not None:
        print(f"\nTraining-Validation Gap:")
        print(f"  Gap: {overfitting_analysis['train_val_gap']:.4f} kcal/mol")
    
    # Save model and results
    output_dir = Path("training_output (CRITICAL DIRECTORY DO NOT TOUCH)")
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = output_dir / "mlp_baseline_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'input_dim': 2344,
            'dropout': 0.2
        },
        'training_history': history,
        'hyperparameters': {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience
        }
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics
    metrics_path = output_dir / "mlp_baseline_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'train_metrics': {k: float(v) for k, v in train_metrics.items()},
            'val_metrics': {k: float(v) for k, v in val_metrics.items()},
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'overfitting_analysis': {k: float(v) if isinstance(v, (int, float)) else v 
                                   for k, v in overfitting_analysis.items()},
            'training_history': {
                'train_loss': [float(x) for x in history['train_loss']],
                'val_loss': [float(x) for x in history['val_loss']],
                'epoch': history['epoch'],
                'best_epoch': history['best_epoch'],
                'best_val_loss': float(history['best_val_loss'])
            }
        }, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save predictions
    predictions_path = output_dir / "mlp_baseline_predictions.csv"
    pd.DataFrame({
        'set': ['train'] * len(train_pred) + ['val'] * len(val_pred) + ['test'] * len(test_pred),
        'true': np.concatenate([train_true, val_true, test_true]),
        'predicted': np.concatenate([train_pred, val_pred, test_pred])
    }).to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    return model, history, train_metrics, val_metrics, test_metrics, overfitting_analysis


if __name__ == "__main__":
    main()

