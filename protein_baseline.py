"""
Baseline ML Pipeline for Protein Property Prediction
Uses precomputed embeddings from ProtT5-XL and ESM-2 650M models
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import warnings
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import hashlib
from collections import defaultdict
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score,
    make_scorer
)
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import xgboost as xgb
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress bar
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Transformers for embeddings
from transformers import T5EncoderModel, T5Tokenizer
from transformers import EsmModel, EsmTokenizer
import torch

# Visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Web API
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Export formats
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Canonical amino acids
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def setup_logging(output_dir: str = "./results") -> logging.Logger:
    """
    Set up logging to file and console
    
    Args:
        output_dir: Directory to save log file
        
    Returns:
        Logger instance
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    log_file = output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def validate_sequence(
    sequence: str,
    min_length: int = 10,
    max_length: int = 5000
) -> Tuple[bool, Optional[str]]:
    """
    Validate protein sequence
    
    Args:
        sequence: Protein sequence string
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(sequence, str):
        return False, "Sequence must be a string"
    
    if len(sequence) < min_length:
        return False, f"Sequence too short (min {min_length} residues)"
    
    if len(sequence) > max_length:
        return False, f"Sequence too long (max {max_length} residues)"
    
    sequence_upper = sequence.upper()
    invalid_chars = set(sequence_upper) - CANONICAL_AA
    
    if invalid_chars:
        return False, f"Invalid amino acids found: {sorted(invalid_chars)}"
    
    return True, None


def validate_sequences_batch(
    sequences: pd.Series,
    min_length: int = 10,
    max_length: int = 5000
) -> Tuple[pd.Series, pd.Series]:
    """
    Validate a batch of sequences
    
    Args:
        sequences: Series of protein sequences
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (valid_sequences, validation_results)
        validation_results contains 'valid' or error message for each sequence
    """
    validation_results = []
    valid_indices = []
    
    for idx, seq in enumerate(sequences):
        is_valid, error_msg = validate_sequence(seq, min_length, max_length)
        if is_valid:
            validation_results.append('valid')
            valid_indices.append(idx)
        else:
            validation_results.append(error_msg)
    
    valid_sequences = sequences.iloc[valid_indices] if valid_indices else pd.Series([], dtype=object)
    
    return valid_sequences, pd.Series(validation_results, index=sequences.index)


def compute_embedding_diagnostics(
    embeddings: np.ndarray,
    embedding_name: str
) -> Dict:
    """
    Compute diagnostics for embeddings
    
    Args:
        embeddings: Embedding array (n_samples, n_features)
        embedding_name: Name of embedding model
        
    Returns:
        Dictionary with diagnostic metrics
    """
    diagnostics = {
        'name': embedding_name,
        'shape': list(embeddings.shape),
        'dimension': int(embeddings.shape[1]),
        'n_samples': int(embeddings.shape[0]),
        'mean': float(np.mean(embeddings)),
        'std': float(np.std(embeddings)),
        'min': float(np.min(embeddings)),
        'max': float(np.max(embeddings)),
    }
    
    # Check for degenerate embeddings (all zeros or constant)
    per_sample_variance = np.var(embeddings, axis=1)
    zero_variance_samples = np.sum(per_sample_variance < 1e-10)
    all_zero_samples = np.sum(np.all(np.abs(embeddings) < 1e-10, axis=1))
    
    diagnostics['zero_variance_samples'] = int(zero_variance_samples)
    diagnostics['all_zero_samples'] = int(all_zero_samples)
    diagnostics['mean_per_sample_variance'] = float(np.mean(per_sample_variance))
    diagnostics['min_per_sample_variance'] = float(np.min(per_sample_variance))
    
    # Check for NaN or Inf
    nan_count = int(np.sum(np.isnan(embeddings)))
    inf_count = int(np.sum(np.isinf(embeddings)))
    diagnostics['nan_count'] = nan_count
    diagnostics['inf_count'] = inf_count
    
    return diagnostics


class EmbeddingExtractor:
    """Extract embeddings from protein language models"""
    
    def __init__(self, cache_dir: str = "./embeddings_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.prot_t5_model = None
        self.prot_t5_tokenizer = None
        self.esm2_model = None
        self.esm2_tokenizer = None
    
    def load_prot_t5(self):
        """Load ProtT5-XL model and tokenizer"""
        if self.prot_t5_model is None:
            print("Loading ProtT5-XL model...")
            model_name = "Rostlab/prot_t5_xl_uniref50"
            self.prot_t5_tokenizer = T5Tokenizer.from_pretrained(
                model_name, do_lower_case=False
            )
            self.prot_t5_model = T5EncoderModel.from_pretrained(model_name)
            self.prot_t5_model.to(device)
            self.prot_t5_model.eval()
            print("ProtT5-XL loaded successfully")
        return self.prot_t5_model, self.prot_t5_tokenizer
    
    def load_esm2(self):
        """Load ESM-2 650M model and tokenizer"""
        if self.esm2_model is None:
            print("Loading ESM-2 650M model...")
            model_name = "facebook/esm2_t33_650M_UR50D"
            self.esm2_tokenizer = EsmTokenizer.from_pretrained(model_name)
            self.esm2_model = EsmModel.from_pretrained(model_name)
            self.esm2_model.to(device)
            self.esm2_model.eval()
            print("ESM-2 650M loaded successfully")
        return self.esm2_model, self.esm2_tokenizer
    
    def get_prot_t5_embedding(
        self, 
        sequence: str, 
        mean_pool: bool = True
    ) -> np.ndarray:
        """
        Extract ProtT5-XL embedding for a protein sequence
        
        Args:
            sequence: Protein sequence (amino acid string)
            mean_pool: If True, mean-pool per-residue embeddings; else use CLS
            
        Returns:
            Embedding vector of shape (1024,)
        """
        model, tokenizer = self.load_prot_t5()
        
        # ProtT5 expects space-separated amino acids
        sequence = " ".join(list(sequence))
        
        # Tokenize sequence
        encoded = tokenizer(
            sequence,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state  # (1, seq_len, 1024)
        
        if mean_pool:
            # Mean pooling over sequence length (excluding padding)
            attention_mask = encoded['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze(0).cpu().numpy()
        else:
            # Use first token (CLS equivalent)
            embedding = embeddings[0, 0, :].cpu().numpy()
        
        return embedding
    
    def get_esm2_embedding(
        self, 
        sequence: str, 
        mean_pool: bool = True
    ) -> np.ndarray:
        """
        Extract ESM-2 650M embedding for a protein sequence
        
        Args:
            sequence: Protein sequence (amino acid string)
            mean_pool: If True, mean-pool per-residue embeddings; else use CLS
            
        Returns:
            Embedding vector of shape (~1280,)
        """
        model, tokenizer = self.load_esm2()
        
        # Tokenize sequence
        encoded = tokenizer(
            sequence,
            add_special_tokens=True,
            padding="max_length",
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
        
        if mean_pool:
            # Mean pooling over sequence length (excluding padding)
            attention_mask = encoded['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze(0).cpu().numpy()
        else:
            # Use first token (CLS)
            embedding = embeddings[0, 0, :].cpu().numpy()
        
        return embedding
    
    def extract_embeddings_batch(
        self,
        sequences: pd.Series,
        model_type: str = "both",
        mean_pool: bool = True,
        cache: bool = True,
        validate: bool = True,
        min_length: int = 10,
        max_length: int = 5000,
        logger: Optional[logging.Logger] = None,
        use_multi_gpu: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Extract embeddings for a batch of sequences
        
        Args:
            sequences: Series of protein sequences
            model_type: "prot_t5", "esm2", or "both"
            mean_pool: Whether to use mean pooling
            cache: Whether to cache embeddings
            validate: Whether to validate sequences
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            logger: Logger instance
            use_multi_gpu: Whether to use multiple GPUs (if available)
            
        Returns:
            Tuple of (embedding_dict, diagnostics_dict)
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        
        # Validate sequences if requested
        if validate:
            valid_sequences, validation_results = validate_sequences_batch(
                sequences, min_length, max_length
            )
            n_valid = len(valid_sequences)
            n_invalid = len(sequences) - n_valid
            logger.info(f"Sequence validation: {n_valid} valid, {n_invalid} invalid")
            if n_invalid > 0:
                logger.warning(f"Skipping {n_invalid} invalid sequences")
            sequences = valid_sequences
        else:
            validation_results = pd.Series(['valid'] * len(sequences), index=sequences.index)
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences after validation")
        
        results = {}
        diagnostics = {}
        
        if model_type in ["prot_t5", "both"]:
            prot_t5_embeddings = []
            iterator = tqdm(enumerate(sequences), total=len(sequences), desc="Extracting ProtT5-XL embeddings") if TQDM_AVAILABLE else enumerate(sequences)
            for idx, seq in iterator:
                # Use stable hashing for cache keys; Python's built-in `hash()` is randomized per process.
                seq_hash = hashlib.sha1(seq.encode("utf-8")).hexdigest()
                cache_file = self.cache_dir / f"prot_t5_{seq_hash}.npy"
                
                if cache and cache_file.exists():
                    emb = np.load(cache_file)
                else:
                    emb = self.get_prot_t5_embedding(seq, mean_pool=mean_pool)
                    if cache:
                        np.save(cache_file, emb)
                
                prot_t5_embeddings.append(emb)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(sequences)} sequences for ProtT5-XL")
            
            results['prot_t5'] = np.array(prot_t5_embeddings)
            diagnostics['prot_t5'] = compute_embedding_diagnostics(results['prot_t5'], 'prot_t5')
            logger.info(f"ProtT5-XL embeddings shape: {results['prot_t5'].shape}")
            print(f"ProtT5-XL embeddings shape: {results['prot_t5'].shape}")
            print(f"  Dimension: {diagnostics['prot_t5']['dimension']}")
            print(f"  Mean variance: {diagnostics['prot_t5']['mean_per_sample_variance']:.6f}")
            if diagnostics['prot_t5']['all_zero_samples'] > 0:
                logger.warning(f"Found {diagnostics['prot_t5']['all_zero_samples']} all-zero embeddings")
                print(f"  WARNING: {diagnostics['prot_t5']['all_zero_samples']} all-zero embeddings detected!")
        
        if model_type in ["esm2", "both"]:
            esm2_embeddings = []
            iterator = tqdm(enumerate(sequences), total=len(sequences), desc="Extracting ESM-2 embeddings") if TQDM_AVAILABLE else enumerate(sequences)
            for idx, seq in iterator:
                # Use stable hashing for cache keys; Python's built-in `hash()` is randomized per process.
                seq_hash = hashlib.sha1(seq.encode("utf-8")).hexdigest()
                cache_file = self.cache_dir / f"esm2_{seq_hash}.npy"
                
                if cache and cache_file.exists():
                    emb = np.load(cache_file)
                else:
                    emb = self.get_esm2_embedding(seq, mean_pool=mean_pool)
                    if cache:
                        np.save(cache_file, emb)
                
                esm2_embeddings.append(emb)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(sequences)} sequences for ESM-2")
            
            results['esm2'] = np.array(esm2_embeddings)
            diagnostics['esm2'] = compute_embedding_diagnostics(results['esm2'], 'esm2')
            logger.info(f"ESM-2 embeddings shape: {results['esm2'].shape}")
            print(f"ESM-2 embeddings shape: {results['esm2'].shape}")
            print(f"  Dimension: {diagnostics['esm2']['dimension']}")
            print(f"  Mean variance: {diagnostics['esm2']['mean_per_sample_variance']:.6f}")
            if diagnostics['esm2']['all_zero_samples'] > 0:
                logger.warning(f"Found {diagnostics['esm2']['all_zero_samples']} all-zero embeddings")
                print(f"  WARNING: {diagnostics['esm2']['all_zero_samples']} all-zero embeddings detected!")
        
        return results, diagnostics


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    is_classification: bool,
    n_folds: int = 5,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Run k-fold cross-validation
    
    Args:
        X: Feature matrix
        y: Target array
        is_classification: Whether task is classification
        n_folds: Number of folds
        logger: Logger instance
        
    Returns:
        Dictionary with CV results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Running {n_folds}-fold cross-validation...")
    print(f"\n{'='*60}")
    print(f"RUNNING {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    if is_classification:
        models = {
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        scoring = {
            'accuracy': 'accuracy',
            'f1': make_scorer(f1_score, average='weighted')
        }
    else:
        models = {
            'xgboost': xgb.XGBRegressor(random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        scoring = {
            'mae': 'neg_mean_absolute_error',
            'rmse': 'neg_root_mean_squared_error',
            'r2': 'r2'
        }
    
    cv_results = {}
    
    for model_name, model in models.items():
        logger.info(f"Cross-validating {model_name}...")
        print(f"\n{model_name.upper()}:")
        
        model_cv_results = {}
        
        for metric_name, scorer in scoring.items():
            scores = cross_val_score(model, X, y, cv=kf, scoring=scorer, n_jobs=-1)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            model_cv_results[metric_name] = {
                'mean': float(mean_score),
                'std': float(std_score),
                'scores': scores.tolist()
            }
            
            # For negative scores, flip sign for display
            display_score = -mean_score if metric_name in ['mae', 'rmse'] else mean_score
            display_std = std_score
            print(f"  {metric_name.upper()}: {display_score:.4f} (+/- {display_std:.4f})")
        
        cv_results[model_name] = model_cv_results
    
    return cv_results


def parse_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Parse FASTA file and return list of (header, sequence) tuples
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        List of (header, sequence) tuples
    """
    sequences = []
    current_header = None
    current_sequence = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_sequence)))
                # Start new sequence
                current_header = line[1:]  # Remove '>'
                current_sequence = []
            else:
                current_sequence.append(line)
        
        # Don't forget the last sequence
        if current_header is not None:
            sequences.append((current_header, ''.join(current_sequence)))
    
    return sequences


def evaluate_predictions_with_truth(
    predictions: Dict,
    true_values: Union[List, np.ndarray, Dict],
    output_dir: str = "./results",
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Evaluate predictions when true values are available
    
    Args:
        predictions: Dictionary with predictions (from predict_from_fasta or similar)
                   Format: {'predictions': [{'header': 'seq1', 'pred_value': 2.3, ...}, ...]}
        true_values: Either:
                    - List/array of true values (same order as predictions)
                    - Dict mapping header to true value: {'seq1': 2.5, 'seq2': -1.2, ...}
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Dictionary with evaluation metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    pred_list = predictions.get('predictions', [])
    if len(pred_list) == 0:
        logger.warning("No predictions to evaluate")
        return {}
    
    # Extract predicted values
    y_pred = np.array([p['pred_value'] for p in pred_list])
    
    # Extract true values
    if isinstance(true_values, dict):
        # Map headers to true values
        headers = [p['header'] for p in pred_list]
        y_true = np.array([true_values.get(h, np.nan) for h in headers])
    else:
        # Assume same order
        y_true = np.array(true_values)
    
    # Remove NaN values
    valid_mask = ~np.isnan(y_true)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        logger.error("No valid true values found")
        return {}
    
    logger.info(f"Evaluating {len(y_true)} predictions with true values")
    
    # Calculate metrics
    metrics = compute_comprehensive_metrics(y_true, y_pred)
    
    # Save evaluation results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    eval_results = {
        'n_samples': int(len(y_true)),
        'metrics': metrics,
        'predictions_with_truth': [
            {
                'header': pred_list[i]['header'],
                'pred_value': float(y_pred[i]),
                'true_value': float(y_true[i]),
                'error': float(y_true[i] - y_pred[i]),
                'absolute_error': float(np.abs(y_true[i] - y_pred[i]))
            }
            for i in range(len(y_true))
        ]
    }
    
    eval_path = output_dir / "evaluation_with_truth.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Saved evaluation results to {eval_path}")
    print(f"\nEvaluation Results:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Pearson: {metrics['pearson_correlation']:.4f}")
    print(f"  Spearman: {metrics['spearman_correlation']:.4f}")
    
    return eval_results


def predict_from_fasta(
    fasta_path: str,
    model_path: str,
    embedding_model_type: str = "both",
    output_path: Optional[str] = None,
    min_length: int = 10,
    max_length: int = 5000,
    logger: Optional[logging.Logger] = None,
    true_values: Optional[Union[List, np.ndarray, Dict]] = None,
    use_composition_features: bool = True,
) -> Dict:
    """
    Predict properties for sequences in a FASTA file
    
    Args:
        fasta_path: Path to FASTA file
        model_path: Path to trained model pickle file
        embedding_model_type: "prot_t5", "esm2", or "both"
        output_path: Path to save predictions JSON (optional)
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        logger: Logger instance
        use_composition_features: Whether to append amino-acid composition features
            (must match how the model was trained)
        
    Returns:
        Dictionary with predictions
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Processing FASTA file: {fasta_path}")
    print(f"\n{'='*60}")
    print(f"PROCESSING FASTA FILE: {fasta_path}")
    print(f"{'='*60}")
    
    # Parse FASTA
    fasta_sequences = parse_fasta(fasta_path)
    logger.info(f"Found {len(fasta_sequences)} sequences in FASTA file")
    print(f"Found {len(fasta_sequences)} sequences")
    
    # Validate sequences
    valid_sequences = []
    validation_results = []
    
    for header, sequence in fasta_sequences:
        is_valid, error_msg = validate_sequence(sequence, min_length, max_length)
        if is_valid:
            valid_sequences.append((header, sequence))
            validation_results.append({'header': header, 'status': 'valid'})
        else:
            validation_results.append({'header': header, 'status': 'invalid', 'error': error_msg})
            logger.warning(f"Invalid sequence {header}: {error_msg}")
    
    logger.info(f"Valid sequences: {len(valid_sequences)}/{len(fasta_sequences)}")
    print(f"Valid sequences: {len(valid_sequences)}/{len(fasta_sequences)}")
    
    if len(valid_sequences) == 0:
        return {
            'validation_results': validation_results,
            'predictions': []
        }
    
    # Extract embeddings
    extractor = EmbeddingExtractor()
    sequences_series = pd.Series([seq for _, seq in valid_sequences])
    embeddings, _ = extractor.extract_embeddings_batch(
        sequences_series,
        model_type=embedding_model_type,
        mean_pool=True,
        cache=True,
        validate=False,  # Already validated above
        logger=logger
    )

    # Match training-time feature construction. The downstream models expect
    # the same dimensionality that results from optionally appending composition
    # features.
    if use_composition_features:
        embeddings = add_composition_features(embeddings, sequences_series)
    
    # Prepare features
    if 'prot_t5' in embeddings and 'esm2' in embeddings:
        X = np.concatenate([embeddings['prot_t5'], embeddings['esm2']], axis=1)
    elif 'prot_t5' in embeddings:
        X = embeddings['prot_t5']
    elif 'esm2' in embeddings:
        X = embeddings['esm2']
    else:
        raise ValueError("No embeddings found")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Predict
    if isinstance(model, EnsembleModel):
        # Handle ensemble model
        y_pred, y_pred_std = model.predict_with_uncertainty(X)
    elif hasattr(model, "predict_with_uncertainty") and callable(getattr(model, "predict_with_uncertainty")):
        # Generic hook for custom ensembles that can return uncertainties.
        y_pred, y_pred_std = model.predict_with_uncertainty(X)
    elif isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
        is_rf = isinstance(model, RandomForestRegressor)
        if is_rf:
            # Regression Random Forest - get uncertainty
            y_pred, y_pred_std = predict_with_uncertainty(model, X, is_random_forest=True)
        else:
            y_pred = model.predict(X)
            y_pred_std = None
    else:
        y_pred = model.predict(X)
        y_pred_std = None
    
    # Compile results
    predictions = []
    for idx, (header, _) in enumerate(valid_sequences):
        pred_dict = {
            'header': header,
            'pred_value': float(y_pred[idx])
        }
        if y_pred_std is not None:
            pred_dict['uncertainty'] = float(y_pred_std[idx])
        predictions.append(pred_dict)
    
    results = {
        'validation_results': validation_results,
        'predictions': predictions,
        'n_valid': len(valid_sequences),
        'n_invalid': len(fasta_sequences) - len(valid_sequences)
    }
    
    # Evaluate with true values if provided
    if true_values is not None:
        logger.info("True values provided - computing evaluation metrics")
        eval_results = evaluate_predictions_with_truth(
            results, true_values, 
            output_dir=str(Path(output_path).parent) if output_path else "./results",
            logger=logger
        )
        results['evaluation'] = eval_results
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved predictions to {output_path}")
        print(f"Saved predictions to {output_path}")
        
        return results


def predict_single_sequence_with_outputs(
    sequence: str,
    sequence_name: str,
    model_path: str,
    true_values: Optional[Dict[str, float]] = None,
    embedding_model_type: str = "both",
    output_dir: str = "./sequence_results",
    trait_name: str = "Property",
    use_composition_features: bool = True
) -> Dict:
    """
    Predict properties for a single sequence and generate comprehensive outputs:
    - Table file (CSV) with real vs predicted values
    - Beautiful image file with comparison graphs
    - Text file summarizing results and errors
    
    Args:
        sequence: Protein sequence string
        sequence_name: Name/identifier for the sequence
        model_path: Path to trained model pickle file
        true_values: Dictionary mapping trait names to true values (optional)
        embedding_model_type: "prot_t5", "esm2", or "both"
        output_dir: Output directory (will create subfolder for this sequence)
        trait_name: Name of the trait being predicted
        use_composition_features: Whether to add composition features
        
    Returns:
        Dictionary with predictions and file paths
    """
    # Create output directory for this sequence
    seq_output_dir = Path(output_dir) / sequence_name.replace(" ", "_").replace("/", "_")
    seq_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processing sequence: {sequence_name}")
    
    # Validate sequence
    is_valid, error_msg = validate_sequence(sequence, min_length=10, max_length=5000)
    if not is_valid:
        raise ValueError(f"Invalid sequence: {error_msg}")
    
    # Extract embeddings
    extractor = EmbeddingExtractor()
    sequences_series = pd.Series([sequence])
    embeddings, _ = extractor.extract_embeddings_batch(
        sequences_series,
        model_type=embedding_model_type,
        mean_pool=True,
        cache=True,
        validate=False,
        logger=logger
    )
    
    # Add composition features if requested
    if use_composition_features:
        embeddings = add_composition_features(embeddings, sequences_series)
    
    # Prepare features
    if 'prot_t5' in embeddings and 'esm2' in embeddings:
        X = np.concatenate([embeddings['prot_t5'], embeddings['esm2']], axis=1)
    elif 'prot_t5' in embeddings:
        X = embeddings['prot_t5']
    elif 'esm2' in embeddings:
        X = embeddings['esm2']
    else:
        raise ValueError("No embeddings found")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Predict
    if isinstance(model, EnsembleModel):
        # Handle ensemble model
        y_pred, y_pred_std = model.predict_with_uncertainty(X)
    elif hasattr(model, "predict_with_uncertainty") and callable(getattr(model, "predict_with_uncertainty")):
        # Generic hook for custom ensembles that can return uncertainties.
        y_pred, y_pred_std = model.predict_with_uncertainty(X)
    elif isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
        is_rf = isinstance(model, RandomForestRegressor)
        if is_rf:
            y_pred, y_pred_std = predict_with_uncertainty(model, X, is_random_forest=True)
        else:
            y_pred = model.predict(X)
            y_pred_std = None
    else:
        y_pred = model.predict(X)
        y_pred_std = None
    
    pred_value = float(y_pred[0])
    uncertainty = float(y_pred_std[0]) if y_pred_std is not None else None
    
    # Prepare results
    results = {
        'sequence_name': sequence_name,
        'sequence_length': len(sequence),
        'predicted_value': pred_value,
        'uncertainty': uncertainty
    }
    
    # If true values provided, create comparison outputs
    if true_values:
        # Prepare data for table
        table_data = []
        y_true_list = []
        y_pred_list = []
        trait_names_list = []
        
        for trait, true_val in true_values.items():
            # For now, use the same prediction for all traits
            # In a real scenario, you'd have separate models for each trait
            table_data.append({
                'Trait': trait,
                'True_Value': true_val,
                'Predicted_Value': pred_value,
                'Error': true_val - pred_value,
                'Absolute_Error': abs(true_val - pred_value),
                'Percent_Error': abs((true_val - pred_value) / true_val * 100) if true_val != 0 else None,
                'Uncertainty': uncertainty
            })
            y_true_list.append(true_val)
            y_pred_list.append(pred_value)
            trait_names_list.append(trait)
        
        # Create table file (CSV)
        df_table = pd.DataFrame(table_data)
        table_path = seq_output_dir / "comparison_table.csv"
        df_table.to_csv(table_path, index=False)
        logger.info(f"Saved comparison table to {table_path}")
        
        # Create beautiful comparison plot
        y_true_array = np.array(y_true_list)
        y_pred_array = np.array(y_pred_list)
        y_std_array = np.array([uncertainty] * len(y_true_list)) if uncertainty else None
        
        # Use first trait name for main plot, or create multi-trait plot
        if len(trait_names_list) == 1:
            plot_path = seq_output_dir / "comparison_plot.png"
            create_beautiful_comparison_plot(
                y_true_array, y_pred_array, sequence_name,
                str(plot_path), y_std_array, trait_names_list[0]
            )
        else:
            # Multi-trait plot
            plot_path = seq_output_dir / "comparison_plot.png"
            create_multi_trait_comparison_plot(
                y_true_list, y_pred_list, trait_names_list,
                sequence_name, str(plot_path), uncertainty
            )
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_array, y_pred_array)
        rmse = np.sqrt(mean_squared_error(y_true_array, y_pred_array))
        
        # R² and Pearson correlation require at least 2 points
        if len(y_true_array) > 1:
            r2 = r2_score(y_true_array, y_pred_array)
            pearson_corr, pearson_p = pearsonr(y_true_array, y_pred_array)
        else:
            # Single point - metrics are undefined
            r2 = np.nan
            pearson_corr = np.nan
            pearson_p = np.nan
        
        # Create summary text file
        summary_path = seq_output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"PREDICTION SUMMARY: {sequence_name}\n")
            f.write("="*70 + "\n\n")
            f.write(f"Sequence Length: {len(sequence)} amino acids\n")
            f.write(f"Number of Traits Evaluated: {len(true_values)}\n\n")
            f.write("-"*70 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Mean Absolute Error (MAE):     {mae:.6f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {rmse:.6f}\n")
            if not np.isnan(r2):
                f.write(f"R² Score:                      {r2:.6f}\n")
                f.write(f"Pearson Correlation:           {pearson_corr:.6f} (p={pearson_p:.6e})\n")
            else:
                f.write(f"R² Score:                      N/A (single data point)\n")
                f.write(f"Pearson Correlation:           N/A (single data point)\n")
            f.write("\n")
            f.write("-"*70 + "\n")
            f.write("TRAIT-BY-TRAIT RESULTS\n")
            f.write("-"*70 + "\n")
            for row in table_data:
                f.write(f"\nTrait: {row['Trait']}\n")
                f.write(f"  True Value:      {row['True_Value']:.6f}\n")
                f.write(f"  Predicted Value: {row['Predicted_Value']:.6f}\n")
                f.write(f"  Error:           {row['Error']:.6f}\n")
                f.write(f"  Absolute Error:  {row['Absolute_Error']:.6f}\n")
                if row['Percent_Error'] is not None:
                    f.write(f"  Percent Error:   {row['Percent_Error']:.2f}%\n")
                if row['Uncertainty'] is not None:
                    f.write(f"  Uncertainty:     {row['Uncertainty']:.6f}\n")
                    f.write(f"  95% CI:         [{row['Predicted_Value'] - 1.96*row['Uncertainty']:.6f}, "
                           f"{row['Predicted_Value'] + 1.96*row['Uncertainty']:.6f}]\n")
            f.write("\n" + "="*70 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("="*70 + "\n")
        
        logger.info(f"Saved summary to {summary_path}")
        
        results.update({
            'true_values': true_values,
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2) if not np.isnan(r2) else None,
                'pearson_correlation': float(pearson_corr) if not np.isnan(pearson_corr) else None,
                'pearson_p_value': float(pearson_p) if not np.isnan(pearson_p) else None
            },
            'output_files': {
                'table': str(table_path),
                'plot': str(plot_path),
                'summary': str(summary_path)
            }
        })
    else:
        # No true values - just save prediction
        summary_path = seq_output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"PREDICTION RESULT: {sequence_name}\n")
            f.write("="*70 + "\n\n")
            f.write(f"Sequence Length: {len(sequence)} amino acids\n\n")
            f.write(f"Predicted {trait_name}: {pred_value:.6f}\n")
            if uncertainty:
                f.write(f"Uncertainty: {uncertainty:.6f}\n")
                f.write(f"95% Confidence Interval: [{pred_value - 1.96*uncertainty:.6f}, "
                       f"{pred_value + 1.96*uncertainty:.6f}]\n")
            f.write("\n" + "="*70 + "\n")
        
        results['output_files'] = {
            'summary': str(summary_path)
        }
    
    return results


def create_multi_trait_comparison_plot(
    y_true_list: List[float],
    y_pred_list: List[float],
    trait_names: List[str],
    sequence_name: str,
    output_path: str,
    uncertainty: Optional[float] = None
) -> None:
    """
    Create a beautiful multi-trait comparison plot
    
    Args:
        y_true_list: List of true values for each trait
        y_pred_list: List of predicted values for each trait
        trait_names: List of trait names
        sequence_name: Name of the sequence
        output_path: Path to save the image
        uncertainty: Uncertainty value (optional)
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            plt.style.use('ggplot')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{sequence_name} - Multi-Trait Prediction Analysis', 
                fontsize=18, fontweight='bold', y=0.995)
    
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    # 1. Bar chart comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(trait_names))
    width = 0.35
    ax1.bar(x_pos - width/2, y_true, width, label='True Values', 
           alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
    ax1.bar(x_pos + width/2, y_pred, width, label='Predicted Values', 
           alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
    if uncertainty:
        ax1.errorbar(x_pos + width/2, y_pred, yerr=1.96*uncertainty, 
                     fmt='none', color='black', capsize=5, capthick=2)
    ax1.set_xlabel('Traits', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Values', fontsize=12, fontweight='bold')
    ax1.set_title('True vs Predicted by Trait', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(trait_names, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 2. Scatter plot
    ax2 = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(trait_names)))
    for i, (true, pred, name, color) in enumerate(zip(y_true, y_pred, trait_names, colors)):
        ax2.scatter(true, pred, s=200, alpha=0.7, color=color, 
                   edgecolors='black', linewidths=2, label=name, zorder=3)
        if uncertainty:
            ax2.errorbar(true, pred, xerr=0, yerr=1.96*uncertainty, 
                        fmt='none', color=color, capsize=5, capthick=2, alpha=0.5)
    
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], 
            'k--', alpha=0.5, linewidth=2, label='Perfect Prediction', zorder=1)
    ax2.set_xlabel('True Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax2.set_title('True vs Predicted (All Traits)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Error bars
    ax3 = axes[1, 0]
    errors = y_true - y_pred
    colors_bar = ['red' if e < 0 else 'green' for e in errors]
    ax3.barh(trait_names, errors, alpha=0.7, color=colors_bar, 
            edgecolor='black', linewidth=1.5)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax3.set_xlabel('Error (True - Predicted)', fontsize=12, fontweight='bold')
    ax3.set_title('Prediction Errors by Trait', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # 4. Percent error
    ax4 = axes[1, 1]
    percent_errors = [abs((t - p) / t * 100) if t != 0 else 0 
                     for t, p in zip(y_true, y_pred)]
    ax4.barh(trait_names, percent_errors, alpha=0.7, color='orange', 
            edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Absolute Percent Error (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Percent Error by Trait', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add overall metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    textstr = f'Overall Metrics:\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR² = {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black')
    fig.text(0.99, 0.01, textstr, fontsize=11, verticalalignment='bottom',
            horizontalalignment='right', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved multi-trait comparison plot to {output_path}")


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with protein sequences and targets
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {len(df)} samples, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def prepare_features_and_targets(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    target_col: str = None,
    embedding_dict: Dict[str, np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Prepare features (embeddings) and targets from dataframe
    
    Args:
        df: DataFrame with sequences and targets
        sequence_col: Name of column containing sequences
        target_col: Name of target column (auto-detect if None)
        embedding_dict: Dictionary with embedding arrays
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target array
        is_classification: Whether target is classification
    """
    # Auto-detect target column
    if target_col is None:
        # Common target column names
        possible_targets = ['ddG', 'effect_binary', 'temperature', 'label', 'target', 'y']
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # Use last column that's not sequence_col
            target_col = [c for c in df.columns if c != sequence_col][-1]
    
    print(f"Using target column: {target_col}")
    
    # Extract targets
    y = df[target_col].values
    
    # Determine if classification
    unique_values = np.unique(y[~pd.isna(y)])
    is_classification = (
        len(unique_values) <= 10 or 
        df[target_col].dtype == 'bool' or
        df[target_col].dtype.name == 'object'
    )
    
    if is_classification:
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Classification task with {len(unique_values)} classes")
    else:
        print("Regression task")
    
    # Prepare features
    if embedding_dict is None:
        raise ValueError("embedding_dict must be provided")
    
    # Concatenate embeddings if both are available
    if 'prot_t5' in embedding_dict and 'esm2' in embedding_dict:
        X = np.concatenate([embedding_dict['prot_t5'], embedding_dict['esm2']], axis=1)
        print(f"Using concatenated embeddings: {X.shape}")
    elif 'prot_t5' in embedding_dict:
        X = embedding_dict['prot_t5']
    elif 'esm2' in embedding_dict:
        X = embedding_dict['esm2']
    else:
        raise ValueError("No embeddings found in embedding_dict")
    
    # Remove samples with NaN targets
    valid_mask = ~pd.isna(df[target_col])
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Final feature matrix: {X.shape}, target array: {y.shape}")
    
    return X, y, is_classification


# ============================================================================
# NEW FEATURE FUNCTIONS
# ============================================================================

def compute_amino_acid_composition(sequence: str) -> np.ndarray:
    """
    Compute amino acid composition features (20 features)
    
    Args:
        sequence: Protein sequence
        
    Returns:
        Array of 20 composition features (normalized counts)
    """
    aa_counts = defaultdict(int)
    for aa in sequence:
        if aa in CANONICAL_AA:
            aa_counts[aa] += 1
    
    total = len(sequence)
    if total == 0:
        return np.zeros(20)
    
    composition = np.array([aa_counts.get(aa, 0) / total for aa in sorted(CANONICAL_AA)])
    return composition


def add_composition_features(embeddings: Dict[str, np.ndarray], sequences: pd.Series) -> Dict[str, np.ndarray]:
    """
    Add amino acid composition features to embeddings
    
    Args:
        embeddings: Dictionary of embedding arrays
        sequences: Series of protein sequences
        
    Returns:
        Dictionary with enhanced embeddings (original + composition)
    """
    enhanced = {}
    composition_features = np.array([compute_amino_acid_composition(seq) for seq in sequences])
    
    for emb_name, emb_array in embeddings.items():
        enhanced[emb_name] = np.concatenate([emb_array, composition_features], axis=1)
    
    return enhanced


def compute_sequence_similarity(sequences: List[str], method: str = 'hamming') -> np.ndarray:
    """
    Compute pairwise sequence similarity matrix
    
    Args:
        sequences: List of protein sequences
        method: Similarity method ('hamming', 'jaccard', or 'edit_distance')
        
    Returns:
        Similarity matrix (n x n)
    """
    n = len(sequences)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if method == 'hamming':
                # Hamming distance (for equal length sequences)
                if len(sequences[i]) == len(sequences[j]):
                    matches = sum(a == b for a, b in zip(sequences[i], sequences[j]))
                    similarity = matches / max(len(sequences[i]), 1)
                else:
                    similarity = 0.0
            elif method == 'jaccard':
                # Jaccard similarity (set of amino acids)
                set_i = set(sequences[i])
                set_j = set(sequences[j])
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                similarity = intersection / union if union > 0 else 0.0
            else:
                # Simple edit distance approximation
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, sequences[i], sequences[j]).ratio()
            
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    return similarity_matrix


def detect_outliers(y_pred: np.ndarray, y_pred_std: Optional[np.ndarray] = None, 
                    threshold: float = 2.0) -> Dict:
    """
    Detect outliers in predictions based on uncertainty or residuals
    
    Args:
        y_pred: Predicted values
        y_pred_std: Standard deviation of predictions (optional)
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Dictionary with outlier information
    """
    outliers = {
        'high_uncertainty': [],
        'extreme_predictions': []
    }
    
    if y_pred_std is not None:
        # Flag high uncertainty predictions
        mean_uncertainty = np.mean(y_pred_std)
        std_uncertainty = np.std(y_pred_std)
        if std_uncertainty > 0:
            z_scores = (y_pred_std - mean_uncertainty) / std_uncertainty
            high_uncertainty_indices = np.where(z_scores > threshold)[0]
            outliers['high_uncertainty'] = high_uncertainty_indices.tolist()
    
    # Flag extreme predictions (far from mean)
    mean_pred = np.mean(y_pred)
    std_pred = np.std(y_pred)
    if std_pred > 0:
        z_scores = np.abs((y_pred - mean_pred) / std_pred)
        extreme_indices = np.where(z_scores > threshold)[0]
        outliers['extreme_predictions'] = extreme_indices.tolist()
    
    return outliers


def compute_bootstrap_confidence_intervals(
    model,
    X: np.ndarray,
    n_bootstrap: int = 100,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bootstrap confidence intervals for XGBoost predictions
    
    Args:
        model: Trained model
        X: Feature matrix
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (predictions, lower_bound, upper_bound)
    """
    n_samples = X.shape[0]
    bootstrap_predictions = []
    
    iterator = tqdm(range(n_bootstrap), desc="Bootstrap sampling") if TQDM_AVAILABLE else range(n_bootstrap)
    for _ in iterator:
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        pred_boot = model.predict(X_boot)
        bootstrap_predictions.append(pred_boot)
    
    bootstrap_predictions = np.array(bootstrap_predictions)
    
    # Compute percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    pred_mean = np.mean(bootstrap_predictions, axis=0)
    pred_lower = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
    pred_upper = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
    
    return pred_mean, pred_lower, pred_upper


def create_ensemble_predictions(
    models: Dict,
    X: np.ndarray,
    is_classification: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Create ensemble predictions by averaging multiple models
    
    Args:
        models: Dictionary of trained models
        X: Feature matrix
        is_classification: Whether task is classification
        
    Returns:
        Tuple of (ensemble_predictions, ensemble_uncertainty)
    """
    all_predictions = []
    all_uncertainties = []
    
    for model_name, model in models.items():
        if isinstance(model, RandomForestRegressor):
            pred, unc = predict_with_uncertainty(model, X, is_random_forest=True)
            all_predictions.append(pred)
            all_uncertainties.append(unc)
        else:
            pred = model.predict(X)
            all_predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(all_predictions, axis=0)
    
    # Average uncertainties if available
    ensemble_unc = None
    if all_uncertainties and len(all_uncertainties) > 0:
        ensemble_unc = np.mean(all_uncertainties, axis=0)
    
    return ensemble_pred, ensemble_unc


class EnsembleModel:
    """
    Ensemble model that combines Random Forest and XGBoost predictions.
    Can be saved and loaded like a regular sklearn model.
    """
    
    def __init__(self, models: Dict, is_classification: bool = False):
        """
        Initialize ensemble model.
        
        Args:
            models: Dictionary of trained models (e.g., {'random_forest': model, 'xgboost': model})
            is_classification: Whether task is classification
        """
        self.models = models
        self.is_classification = is_classification
        self.model_names = list(models.keys())
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions by averaging model predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Ensemble predictions
        """
        all_predictions = []
        
        for model_name, model in self.models.items():
            pred = model.predict(X)
            all_predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(all_predictions, axis=0)
        
        if self.is_classification:
            # Round to nearest class for classification
            ensemble_pred = np.round(ensemble_pred).astype(int)
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make ensemble predictions with uncertainty estimation.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, uncertainty_std)
        """
        all_predictions = []
        all_uncertainties = []
        
        for model_name, model in self.models.items():
            if isinstance(model, RandomForestRegressor):
                pred, unc = predict_with_uncertainty(model, X, is_random_forest=True)
                all_predictions.append(pred)
                all_uncertainties.append(unc)
            else:
                pred = model.predict(X)
                all_predictions.append(pred)
                # XGBoost doesn't provide uncertainty by default, but we can estimate from variance
                # For now, we'll use None and average only RF uncertainties
        
        # Average predictions
        ensemble_pred = np.mean(all_predictions, axis=0)
        
        # Average uncertainties if available
        ensemble_unc = None
        if all_uncertainties and len(all_uncertainties) > 0:
            # Only average uncertainties from models that provide them (e.g., Random Forest)
            valid_uncertainties = [unc for unc in all_uncertainties if unc is not None]
            if valid_uncertainties:
                ensemble_unc = np.mean(valid_uncertainties, axis=0)
        
        return ensemble_pred, ensemble_unc
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities (for classification only).
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities or None if regression
        """
        if not self.is_classification:
            return None
        
        all_probas = []
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                all_probas.append(proba)
        
        if all_probas:
            # Average probabilities
            ensemble_proba = np.mean(all_probas, axis=0)
            return ensemble_proba
        
        return None
    
    def __getstate__(self):
        """Custom pickle serialization."""
        return {
            'models': self.models,
            'is_classification': self.is_classification,
            'model_names': self.model_names
        }
    
    def __setstate__(self, state):
        """Custom pickle deserialization."""
        self.models = state['models']
        self.is_classification = state['is_classification']
        self.model_names = state['model_names']


def extract_feature_importance(
    model,
    model_name: str,
    feature_names: Optional[List[str]] = None
) -> Dict:
    """
    Extract feature importance from trained model
    
    Args:
        model: Trained model
        model_name: Name of the model
        feature_names: Optional list of feature names
        
    Returns:
        Dictionary with feature importance information
    """
    importance_dict = {
        'model_name': model_name,
        'importance_type': None,
        'importances': None,
        'top_features': []
    }
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_dict['importance_type'] = 'feature_importances_'
        importance_dict['importances'] = importances.tolist()
        
        # Get top features
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        top_indices = np.argsort(importances)[::-1][:20]  # Top 20
        importance_dict['top_features'] = [
            {'feature': feature_names[i], 'importance': float(importances[i])}
            for i in top_indices
        ]
    elif hasattr(model, 'coef_'):
        # For linear models
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        importances = np.abs(coef)
        importance_dict['importance_type'] = 'coefficients'
        importance_dict['importances'] = importances.tolist()
    
    return importance_dict


def create_interactive_html_report(
    results: Dict,
    output_path: str,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> None:
    """
    Create interactive HTML report with Plotly visualizations
    
    Args:
        results: Results dictionary with metrics
        output_path: Path to save HTML file
        y_true: True values (optional)
        y_pred: Predicted values (optional)
        model_name: Name of the model
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for interactive HTML reports. Install with: pip install plotly kaleido")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('True vs Predicted', 'Residuals', 'Residual Distribution', 'Metrics Summary'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "bar"}]]
    )
    
    # True vs Predicted scatter
    if y_true is not None and y_pred is not None:
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(size=5, opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect',
                line=dict(dash='dash', color='red')
            ),
            row=1, col=1
        )
        
        # Residuals
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(size=5, opacity=0.6)
            ),
            row=1, col=2
        )
        
        # Residual histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residual Distribution',
                nbinsx=30
            ),
            row=2, col=1
        )
    
    # Metrics bar chart
    if 'metrics' in results:
        metrics = results['metrics']
        metric_names = []
        metric_values = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metric_names.append(key)
                metric_values.append(value)
        
        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, name='Metrics'),
            row=2, col=2
        )
    
    fig.update_layout(
        title=f'{model_name} - Interactive Report',
        height=800,
        showlegend=True
    )
    
    fig.write_html(output_path)
    print(f"Saved interactive HTML report to {output_path}")


def save_model_with_metadata(
    model,
    model_path: str,
    metadata: Dict,
    output_dir: str
) -> str:
    """
    Save model with comprehensive metadata
    
    Args:
        model: Trained model
        model_path: Path to save model
        metadata: Dictionary with model metadata
        output_dir: Output directory
        
    Returns:
        Path to saved model metadata file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata_path = Path(model_path).with_suffix('.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return str(metadata_path)


def load_model_with_metadata(model_path: str) -> Tuple:
    """
    Load model with its metadata
    
    Args:
        model_path: Path to model file
        
    Returns:
        Tuple of (model, metadata)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    metadata_path = Path(model_path).with_suffix('.metadata.json')
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    is_classification: bool,
    n_trials: int = 50
) -> Dict:
    """
    Tune hyperparameters using Optuna
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_type: 'xgboost' or 'random_forest'
        is_classification: Whether task is classification
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary with best parameters and study
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna is required for hyperparameter tuning. Install with: pip install optuna")
    
    def objective(trial):
        if model_type == 'xgboost':
            if is_classification:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = xgb.XGBRegressor(**params, random_state=42)
        else:  # random_forest
            if is_classification:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
                model = RandomForestClassifier(**params, random_state=42)
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
                model = RandomForestRegressor(**params, random_state=42)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error' if not is_classification else 'accuracy')
        return cv_scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }


def export_to_excel(
    predictions: Dict,
    output_path: str,
    sequences: Optional[List[str]] = None
) -> None:
    """
    Export predictions to Excel format
    
    Args:
        predictions: Dictionary with predictions
        output_path: Path to save Excel file
        sequences: Optional list of sequences
    """
    if not OPENPYXL_AVAILABLE:
        raise ImportError("openpyxl is required for Excel export")
    
    df_data = []
    for model_name, pred_data in predictions.items():
        for idx, (true_val, pred_val) in enumerate(zip(pred_data.get('y_true', []), pred_data.get('y_pred', []))):
            row = {
                'model': model_name,
                'index': idx,
                'true_value': true_val,
                'predicted_value': pred_val
            }
            if 'y_pred_std' in pred_data:
                row['uncertainty'] = pred_data['y_pred_std'][idx]
            if sequences:
                row['sequence'] = sequences[idx]
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_excel(output_path, index=False)
    print(f"Exported predictions to {output_path}")


def export_to_parquet(
    predictions: Dict,
    output_path: str
) -> None:
    """
    Export predictions to Parquet format
    
    Args:
        predictions: Dictionary with predictions
        output_path: Path to save Parquet file
    """
    if not PARQUET_AVAILABLE:
        raise ImportError("pyarrow is required for Parquet export")
    
    df_data = []
    for model_name, pred_data in predictions.items():
        for idx, (true_val, pred_val) in enumerate(zip(pred_data.get('y_true', []), pred_data.get('y_pred', []))):
            row = {
                'model': model_name,
                'index': idx,
                'true_value': true_val,
                'predicted_value': pred_val
            }
            if 'y_pred_std' in pred_data:
                row['uncertainty'] = pred_data['y_pred_std'][idx]
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
    print(f"Exported predictions to {output_path}")


def export_to_annotated_fasta(
    predictions: Dict,
    sequences: List[Tuple[str, str]],  # List of (header, sequence) tuples
    output_path: str,
    model_name: str = 'best'
) -> None:
    """
    Export predictions as annotated FASTA file
    
    Args:
        predictions: Dictionary with predictions
        sequences: List of (header, sequence) tuples
        output_path: Path to save FASTA file
        model_name: Model to use for predictions
    """
    with open(output_path, 'w') as f:
        pred_data = predictions.get(model_name, {})
        y_pred = pred_data.get('y_pred', [])
        
        for idx, (header, sequence) in enumerate(sequences):
            pred_val = y_pred[idx] if idx < len(y_pred) else None
            if pred_val is not None:
                f.write(f">{header} | predicted_value={pred_val:.4f}\n")
            else:
                f.write(f">{header}\n")
            f.write(f"{sequence}\n")
    
    print(f"Exported annotated FASTA to {output_path}")


def setup_database(db_path: str) -> sqlite3.Connection:
    """
    Setup SQLite database for storing predictions
    
    Args:
        db_path: Path to database file
        
    Returns:
        Database connection
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sequence_hash TEXT,
            sequence TEXT,
            model_name TEXT,
            predicted_value REAL,
            uncertainty REAL,
            true_value REAL,
            timestamp TEXT,
            metadata TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            model_path TEXT,
            training_date TEXT,
            metrics TEXT,
            metadata TEXT
        )
    ''')
    
    conn.commit()
    return conn


def save_predictions_to_db(
    conn: sqlite3.Connection,
    predictions: Dict,
    sequences: List[str],
    model_name: str,
    true_values: Optional[List] = None
) -> None:
    """
    Save predictions to database
    
    Args:
        conn: Database connection
        predictions: Dictionary with predictions
        sequences: List of sequences
        model_name: Name of model
        true_values: Optional true values
    """
    cursor = conn.cursor()
    pred_data = predictions.get(model_name, {})
    y_pred = pred_data.get('y_pred', [])
    y_pred_std = pred_data.get('y_pred_std', [None] * len(y_pred))
    
    timestamp = datetime.now().isoformat()
    
    for idx, (seq, pred_val) in enumerate(zip(sequences, y_pred)):
        seq_hash = hashlib.md5(seq.encode()).hexdigest()
        uncertainty = y_pred_std[idx] if idx < len(y_pred_std) and y_pred_std[idx] is not None else None
        true_val = true_values[idx] if true_values and idx < len(true_values) else None
        
        cursor.execute('''
            INSERT INTO predictions 
            (sequence_hash, sequence, model_name, predicted_value, uncertainty, true_value, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (seq_hash, seq, model_name, pred_val, uncertainty, true_val, timestamp))
    
    conn.commit()


def incremental_retrain(
    model,
    X_new: np.ndarray,
    y_new: np.ndarray,
    model_type: str
) -> object:
    """
    Incrementally retrain model on new data
    
    Args:
        model: Existing trained model
        X_new: New feature data
        y_new: New target data
        model_type: Type of model ('xgboost' or 'random_forest')
        
    Returns:
        Retrained model
    """
    if model_type == 'xgboost' and hasattr(model, 'fit'):
        # XGBoost supports incremental training
        model.fit(X_new, y_new, xgb_model=model.get_booster())
    elif model_type == 'random_forest':
        # Random Forest doesn't support incremental training, need to retrain
        # This is a simplified version - in practice, you'd combine old and new data
        print("Warning: Random Forest doesn't support incremental training. Retraining from scratch.")
        if isinstance(model, RandomForestRegressor):
            model = RandomForestRegressor(n_estimators=model.n_estimators, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=model.n_estimators, random_state=42)
        model.fit(X_new, y_new)
    
    return model


# ============================================================================
# END OF NEW FEATURE FUNCTIONS
# ============================================================================


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_std: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute comprehensive evaluation metrics for regression
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        y_pred_std: Standard deviation of predictions (for uncertainty)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics['mae'] = float(mae)
    metrics['mse'] = float(mse)
    metrics['rmse'] = float(rmse)
    metrics['r2'] = float(r2)
    
    # Correlation coefficients
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    
    metrics['pearson_correlation'] = float(pearson_corr)
    metrics['pearson_p_value'] = float(pearson_p)
    metrics['spearman_correlation'] = float(spearman_corr)
    metrics['spearman_p_value'] = float(spearman_p)
    
    # Prediction interval width (using uncertainty if available)
    if y_pred_std is not None:
        # 95% prediction interval width (2 * 1.96 * std)
        pred_interval_width = 2 * 1.96 * np.mean(y_pred_std)
        metrics['prediction_interval_width'] = float(pred_interval_width)
        metrics['mean_uncertainty'] = float(np.mean(y_pred_std))
    else:
        # Estimate from residuals
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        pred_interval_width = 2 * 1.96 * residual_std
        metrics['prediction_interval_width'] = float(pred_interval_width)
        metrics['mean_uncertainty'] = None
    
    # Additional statistics
    metrics['mean_true'] = float(np.mean(y_true))
    metrics['mean_pred'] = float(np.mean(y_pred))
    metrics['std_true'] = float(np.std(y_true))
    metrics['std_pred'] = float(np.std(y_pred))
    
    return metrics


def predict_with_uncertainty(
    model,
    X: np.ndarray,
    is_random_forest: bool = False,
    use_bootstrap: bool = False,
    n_bootstrap: int = 100
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Make predictions with uncertainty estimation
    
    Args:
        model: Trained model
        X: Feature matrix
        is_random_forest: Whether model is Random Forest
        use_bootstrap: Whether to use bootstrap for XGBoost (slower but more accurate)
        n_bootstrap: Number of bootstrap samples (if use_bootstrap=True)
        
    Returns:
        Tuple of (predictions, uncertainty_std)
    """
    if is_random_forest and hasattr(model, 'estimators_'):
        # For Random Forest: use std across trees
        predictions_per_tree = np.array([
            tree.predict(X) for tree in model.estimators_
        ])
        pred_mean = np.mean(predictions_per_tree, axis=0)
        pred_std = np.std(predictions_per_tree, axis=0)
        return pred_mean, pred_std
    elif use_bootstrap and isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier)):
        # For XGBoost: use bootstrap confidence intervals
        pred_mean, pred_lower, pred_upper = compute_bootstrap_confidence_intervals(
            model, X, n_bootstrap=n_bootstrap
        )
        # Estimate std from confidence interval (assuming normal distribution)
        pred_std = (pred_upper - pred_lower) / (2 * 1.96)  # 95% CI
        return pred_mean, pred_std
    else:
        # For other models, no uncertainty estimate
        pred = model.predict(X)
        return pred, None


def create_beautiful_comparison_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sequence_name: str,
    output_path: str,
    y_pred_std: Optional[np.ndarray] = None,
    trait_name: str = "Property"
) -> None:
    """
    Create a beautiful comparison plot for real vs predicted values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        sequence_name: Name of the sequence/protein
        output_path: Path to save the image
        y_pred_std: Standard deviation of predictions (optional)
        trait_name: Name of the trait being predicted
    """
    # Set style for beautiful plots
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            plt.style.use('ggplot')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Main comparison plot (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Scatter plot with beautiful styling
    if len(y_true) == 1:
        # Single point - use a larger, more visible marker
        ax1.scatter(y_true, y_pred, 
                   s=300, alpha=0.8, 
                   color='steelblue',
                   edgecolors='black', 
                   linewidths=2,
                   zorder=3,
                   label='Prediction')
    else:
        scatter = ax1.scatter(y_true, y_pred, 
                             s=150, alpha=0.7, 
                             c=range(len(y_true)), 
                             cmap='viridis',
                             edgecolors='black', 
                             linewidths=1.5,
                             zorder=3)
        
        # Add regression line (only if more than 1 point)
        if len(y_true) > 1:
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.min(y_true), np.max(y_true), 100)
            ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=3, 
                     label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}', zorder=2)
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 
             'k-', alpha=0.5, linewidth=2, 
             label='Perfect Prediction', zorder=1)
    
    # Add uncertainty bands if available
    if y_pred_std is not None:
        for i, (true, pred, std) in enumerate(zip(y_true, y_pred, y_pred_std)):
            ax1.fill_between([true, true], 
                            pred - 1.96*std, pred + 1.96*std,
                            alpha=0.2, color='gray', zorder=0)
    
    ax1.set_xlabel(f'True {trait_name}', fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'Predicted {trait_name}', fontsize=14, fontweight='bold')
    ax1.set_title(f'{sequence_name}: True vs Predicted {trait_name}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add metrics text box
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if len(y_true) > 1:
        r2 = r2_score(y_true, y_pred)
        pearson_corr, _ = pearsonr(y_true, y_pred)
        textstr = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nPearson r = {pearson_corr:.4f}'
    else:
        textstr = f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}\n(Single data point)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black')
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props, fontweight='bold')
    
    # Residuals plot
    ax2 = fig.add_subplot(gs[0, 2])
    residuals = y_true - y_pred
    if len(residuals) == 1:
        ax2.scatter(y_pred, residuals, s=200, alpha=0.8, 
                   color='coral', edgecolors='black', linewidths=2, zorder=3)
    else:
        ax2.scatter(y_pred, residuals, s=100, alpha=0.7, 
                   c=range(len(residuals)), cmap='coolwarm',
                   edgecolors='black', linewidths=1, zorder=3)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, zorder=2)
    ax2.set_xlabel('Predicted Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Residual histogram (only if multiple points)
    ax3 = fig.add_subplot(gs[1, 0])
    if len(residuals) > 1:
        n, bins, patches = ax3.hist(residuals, bins=min(20, len(residuals)), edgecolor='black', 
                                    linewidth=1.5, alpha=0.7, zorder=2)
        # Color bars by height
        cm = plt.cm.get_cmap('viridis')
        for i, (n_val, patch) in enumerate(zip(n, patches)):
            patch.set_facecolor(cm(n_val / max(n) if max(n) > 0 else 0))
    else:
        # Single point - show as a bar
        ax3.bar([residuals[0]], [1], width=abs(residuals[0])*0.1 if abs(residuals[0]) > 0 else 0.1,
               edgecolor='black', linewidth=1.5, alpha=0.7, color='steelblue', zorder=2)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, zorder=3)
    ax3.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Residual Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Error distribution
    ax4 = fig.add_subplot(gs[1, 1])
    abs_errors = np.abs(residuals)
    if len(abs_errors) > 1:
        ax4.hist(abs_errors, bins=min(20, len(abs_errors)), edgecolor='black', 
                linewidth=1.5, alpha=0.7, color='coral', zorder=2)
        ax4.axvline(x=np.mean(abs_errors), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(abs_errors):.4f}', zorder=3)
    else:
        ax4.bar([abs_errors[0]], [1], width=abs_errors[0]*0.1 if abs_errors[0] > 0 else 0.1,
               edgecolor='black', linewidth=1.5, alpha=0.7, color='coral', zorder=2)
        ax4.axvline(x=abs_errors[0], color='r', linestyle='--', 
                   linewidth=2, label=f'Error: {abs_errors[0]:.4f}', zorder=3)
    ax4.set_xlabel('Absolute Error', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Q-Q plot for residuals (only if multiple points)
    ax5 = fig.add_subplot(gs[1, 2])
    if len(residuals) > 1:
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax5)
        ax5.set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold')
    else:
        # Single point - show error value
        ax5.text(0.5, 0.5, f'Residual: {residuals[0]:.4f}\n(Single data point)', 
                transform=ax5.transAxes, fontsize=12, fontweight='bold',
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax5.set_title('Residual Value', fontsize=13, fontweight='bold')
        ax5.axis('off')
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # Add overall title
    fig.suptitle(f'{sequence_name} - Prediction Analysis', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved beautiful comparison plot to {output_path}")


def create_visualizations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str = "./results",
    y_pred_std: Optional[np.ndarray] = None
) -> None:
    """
    Create visualization plots for regression results
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of the model
        output_dir: Output directory
        y_pred_std: Standard deviation of predictions (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 5)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. True vs Predicted scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax1.plot(y_true, p(y_true), "r--", alpha=0.8, linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, label='Perfect prediction')
    
    ax1.set_xlabel('True Values', fontsize=12)
    ax1.set_ylabel('Predicted Values', fontsize=12)
    ax1.set_title(f'{model_name}: True vs Predicted', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add R² to plot
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nPearson r = {pearson_corr:.3f}',
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residuals plot
    ax2 = axes[1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values', fontsize=12)
    ax2.set_ylabel('Residuals (True - Predicted)', fontsize=12)
    ax2.set_title(f'{model_name}: Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual histogram
    ax3 = axes[2]
    ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title(f'{model_name}: Residual Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics to histogram
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax3.text(0.05, 0.95, f'Mean = {mean_residual:.3f}\nStd = {std_residual:.3f}',
             transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    plot_path = output_dir / f"{model_name}_evaluation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {plot_path}")


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    is_classification: bool,
    output_dir: str = "./results",
    logger: Optional[logging.Logger] = None,
    create_plots: bool = True,
    best_params: Optional[Dict] = None,
    use_ensemble: bool = True,
    use_bootstrap_ci: bool = False,
    create_html_report: bool = True
) -> Dict:
    """
    Train baseline ML models and evaluate
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        is_classification: Whether task is classification
        output_dir: Directory to save models and results
        logger: Logger instance
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary with models, predictions, metrics, and best_model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    results = {
        'models': {},
        'predictions': {},
        'metrics': {},
        'best_model': None
    }
    
    if is_classification:
        # Classification models - only XGBoost and Random Forest for comparison
        xgb_params = best_params.get('xgboost', {}) if best_params else {}
        rf_params = best_params.get('random_forest', {}) if best_params else {}
        models = {
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', **xgb_params),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, **rf_params)
        }
    else:
        # Regression models - only XGBoost and Random Forest for comparison
        xgb_params = best_params.get('xgboost', {}) if best_params else {}
        rf_params = best_params.get('random_forest', {}) if best_params else {}
        models = {
            'xgboost': xgb.XGBRegressor(random_state=42, **xgb_params),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, **rf_params)
        }
    
    model_scores = {}  # For selecting best model
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict with uncertainty if Random Forest or if bootstrap CI requested for XGBoost
        is_rf = model_name == 'random_forest' and not is_classification
        use_bootstrap = use_bootstrap_ci and model_name == 'xgboost' and not is_classification
        if is_rf:
            y_pred, y_pred_std = predict_with_uncertainty(model, X_test, is_random_forest=True)
        elif use_bootstrap:
            y_pred, y_pred_std = predict_with_uncertainty(model, X_test, use_bootstrap=True)
        else:
            y_pred = model.predict(X_test)
            y_pred_std = None
        
        # For classification, also get probabilities
        if is_classification and hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        
        # Evaluate
        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            metrics = {
                'accuracy': float(accuracy),
                'f1_score': float(f1)
            }
            
            # ROC-AUC for binary classification
            if len(np.unique(y_test)) == 2 and y_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    metrics['roc_auc'] = float(roc_auc)
                except (ValueError, IndexError) as e:
                    logger.debug(f"Could not calculate ROC-AUC: {e}")
                    pass
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            if 'roc_auc' in metrics:
                print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Use accuracy for model selection
            model_scores[model_name] = accuracy
        else:  # Regression
            # Use comprehensive metrics
            metrics = compute_comprehensive_metrics(y_test, y_pred, y_pred_std)
            
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"R²: {metrics['r2']:.4f}")
            print(f"Pearson Correlation: {metrics['pearson_correlation']:.4f} (p={metrics['pearson_p_value']:.4e})")
            print(f"Spearman Correlation: {metrics['spearman_correlation']:.4f} (p={metrics['spearman_p_value']:.4e})")
            if metrics['mean_uncertainty'] is not None:
                print(f"Mean Uncertainty (std): {metrics['mean_uncertainty']:.4f}")
            print(f"Prediction Interval Width: {metrics['prediction_interval_width']:.4f}")
            
            # Use RMSE for model selection (lower is better)
            model_scores[model_name] = -metrics['rmse']  # Negative because we want to maximize
            
            # Create visualizations
            if create_plots:
                create_visualizations(y_test, y_pred, model_name, output_dir, y_pred_std)
            
            # Create interactive HTML report
            if create_html_report:
                html_path = output_dir / f"{model_name}_interactive_report.html"
                create_interactive_html_report(
                    {'metrics': metrics},
                    str(html_path),
                    y_test, y_pred, model_name
                )
            
            # Detect outliers
            outliers = detect_outliers(y_pred, y_pred_std)
            if outliers['high_uncertainty'] or outliers['extreme_predictions']:
                logger.warning(f"{model_name}: Found {len(outliers['high_uncertainty'])} high-uncertainty and {len(outliers['extreme_predictions'])} extreme predictions")
                results['predictions'][model_name]['outliers'] = outliers
        
        # Save model with metadata
        model_path = output_dir / f"{model_name}_model.pkl"
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'training_date': datetime.now().isoformat(),
            'is_classification': is_classification,
            'n_features': X_train.shape[1],
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'metrics': metrics,
            'best_model': model_name == results.get('best_model')
        }
        save_model_with_metadata(model, str(model_path), metadata, str(output_dir))
        logger.info(f"Saved {model_name} model to {model_path}")
        
        # Store results
        results['models'][model_name] = model_path
        results['predictions'][model_name] = {
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist()
        }
        if y_pred_std is not None:
            results['predictions'][model_name]['y_pred_std'] = y_pred_std.tolist()
        if y_proba is not None:
            results['predictions'][model_name]['y_proba'] = y_proba.tolist()
        
        results['metrics'][model_name] = metrics
        
        # Extract feature importance
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        importance_info = extract_feature_importance(model, model_name, feature_names)
        results['feature_importance'] = results.get('feature_importance', {})
        results['feature_importance'][model_name] = importance_info
    
    # Create ensemble predictions if requested
    if use_ensemble:
        ensemble_pred, ensemble_unc = create_ensemble_predictions(
            models, X_test, is_classification
        )
        if not is_classification:
            ensemble_metrics = compute_comprehensive_metrics(y_test, ensemble_pred, ensemble_unc)
            results['predictions']['ensemble'] = {
                'y_true': y_test.tolist(),
                'y_pred': ensemble_pred.tolist(),
                'y_pred_std': ensemble_unc.tolist() if ensemble_unc is not None else None
            }
            results['metrics']['ensemble'] = ensemble_metrics
            logger.info(f"Ensemble RMSE: {ensemble_metrics['rmse']:.4f}")
        
        # Create and save ensemble model
        ensemble_model = EnsembleModel(models, is_classification=is_classification)
        ensemble_model_path = output_dir / "ensemble_model.pkl"
        ensemble_metadata = {
            'model_name': 'ensemble',
            'model_type': 'EnsembleModel',
            'component_models': list(models.keys()),
            'training_date': datetime.now().isoformat(),
            'is_classification': is_classification,
            'n_features': X_train.shape[1],
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'metrics': results['metrics'].get('ensemble', {}) if not is_classification else {},
            'best_model': False
        }
        save_model_with_metadata(ensemble_model, str(ensemble_model_path), ensemble_metadata, str(output_dir))
        results['models']['ensemble'] = ensemble_model_path
        logger.info(f"Saved ensemble model to {ensemble_model_path}")
        print(f"Saved ensemble model to {ensemble_model_path}")
    
    # Select best model
    if model_scores:
        best_model_name = max(model_scores, key=model_scores.get)
        results['best_model'] = best_model_name
        logger.info(f"Best model selected: {best_model_name} (score: {model_scores[best_model_name]:.4f})")
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_model_name.upper()}")
        print(f"{'='*60}")
    
    return results


def predict_all_data(
    models: Dict,
    X_all: np.ndarray,
    y_all: np.ndarray,
    is_classification: bool,
    output_dir: str = "./results"
) -> Dict:
    """
    Make predictions on all data (not just test set)
    
    Args:
        models: Dictionary of trained models
        X_all: All feature data
        y_all: All target data
        is_classification: Whether task is classification
        output_dir: Output directory
        
    Returns:
        Dictionary with predictions for all data
    """
    output_dir = Path(output_dir)
    all_predictions = {}
    
    print("\n" + "="*60)
    print("PREDICTING ON ALL DATA")
    print("="*60)
    
    for model_name, model in models.items():
        print(f"\nPredicting with {model_name}...")
        
        # Predict on all data with uncertainty
        if isinstance(model, EnsembleModel):
            y_pred_all, y_pred_std_all = model.predict_with_uncertainty(X_all)
        elif isinstance(model, RandomForestRegressor):
            y_pred_all, y_pred_std_all = predict_with_uncertainty(model, X_all, is_random_forest=True)
        else:
            y_pred_all = model.predict(X_all)
            y_pred_std_all = None
        
        # For classification, also get probabilities
        if is_classification and hasattr(model, 'predict_proba'):
            y_proba_all = model.predict_proba(X_all)
        else:
            y_proba_all = None
        
        # Calculate metrics using comprehensive function for regression
        if is_classification:
            accuracy = accuracy_score(y_all, y_pred_all)
            f1 = f1_score(y_all, y_pred_all, average='weighted')
            
            metrics = {
                'accuracy': float(accuracy),
                'f1_score': float(f1)
            }
            
            if len(np.unique(y_all)) == 2 and y_proba_all is not None:
                try:
                    roc_auc = roc_auc_score(y_all, y_proba_all[:, 1])
                    metrics['roc_auc'] = float(roc_auc)
                except (ValueError, IndexError):
                    pass
        else:
            # Use comprehensive metrics function
            metrics = compute_comprehensive_metrics(y_all, y_pred_all, y_pred_std_all)
            print(f"  MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            print(f"  Pearson: {metrics['pearson_correlation']:.4f}, Spearman: {metrics['spearman_correlation']:.4f}")
        
        all_predictions[model_name] = {
            'y_true': y_all.tolist(),
            'y_pred': y_pred_all.tolist(),
            'metrics': metrics
        }
        if y_pred_std_all is not None:
            all_predictions[model_name]['y_pred_std'] = y_pred_std_all.tolist()
        if y_proba_all is not None:
            all_predictions[model_name]['y_proba'] = y_proba_all.tolist()
    
    # Save all predictions
    all_pred_path = output_dir / "predictions_all_data.json"
    with open(all_pred_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\nSaved all-data predictions to {all_pred_path}")
    
    return all_predictions


def compute_statistical_analysis(predictions_dict: Dict) -> Dict:
    """
    Compute statistical analysis for regression predictions
    
    Args:
        predictions_dict: Dictionary with predictions from models
                         Format: {model_name: {'y_true': [...], 'y_pred': [...]}, ...}
    
    Returns:
        Dictionary with statistical analysis for each model
    """
    statistical_analysis = {}
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    for model_name, pred_data in predictions_dict.items():
        y_true = np.array(pred_data['y_true'])
        y_pred = np.array(pred_data['y_pred'])
        
        # Calculate percent errors for each sample
        non_zero_mask = np.abs(y_true) > 1e-10
        percent_errors = np.full(len(y_true), np.nan)
        if np.sum(non_zero_mask) > 0:
            percent_errors[non_zero_mask] = np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
            ) * 100
        
        # Statistical summary
        valid_errors = percent_errors[~np.isnan(percent_errors)]
        stats = {
            'mean_percent_error': float(np.mean(valid_errors)) if len(valid_errors) > 0 else None,
            'median_percent_error': float(np.median(valid_errors)) if len(valid_errors) > 0 else None,
            'std_percent_error': float(np.std(valid_errors)) if len(valid_errors) > 0 else None,
            'min_percent_error': float(np.min(valid_errors)) if len(valid_errors) > 0 else None,
            'max_percent_error': float(np.max(valid_errors)) if len(valid_errors) > 0 else None,
            'percent_errors': [float(x) if not np.isnan(x) else None for x in percent_errors],
            'absolute_errors': (np.abs(y_true - y_pred)).tolist(),
            'squared_errors': ((y_true - y_pred) ** 2).tolist()
        }
        
        statistical_analysis[model_name] = stats
        
        print(f"\n{model_name.upper()} - Percent Error Statistics:")
        if stats['mean_percent_error'] is not None:
            print(f"  Mean: {stats['mean_percent_error']:.2f}%")
            print(f"  Median: {stats['median_percent_error']:.2f}%")
            print(f"  Std: {stats['std_percent_error']:.2f}%")
            print(f"  Min: {stats['min_percent_error']:.2f}%")
            print(f"  Max: {stats['max_percent_error']:.2f}%")
    
    return statistical_analysis


def save_results(
    results: Dict,
    embeddings: Dict[str, np.ndarray],
    output_dir: str = "./results",
    all_data_predictions: Dict = None,
    statistical_analysis: Dict = None
) -> None:
    """
    Save all results to disk
    
    Args:
        results: Results dictionary from train_baseline_models
        embeddings: Dictionary with embedding arrays
        output_dir: Output directory
        all_data_predictions: Predictions on all data
        statistical_analysis: Statistical analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save embeddings
    print("\nSaving embeddings...")
    for emb_name, emb_array in embeddings.items():
        emb_path = output_dir / f"{emb_name}_embeddings.npy"
        np.save(emb_path, emb_array)
        print(f"Saved {emb_name} embeddings to {emb_path}")
    
    # Save predictions
    print("\nSaving predictions...")
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump(results['predictions'], f, indent=2)
    print(f"Saved predictions to {predictions_path}")
    
    # Save metrics
    print("\nSaving metrics...")
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save all data predictions if provided
    if all_data_predictions is not None:
        all_pred_path = output_dir / "predictions_all_data.json"
        with open(all_pred_path, 'w') as f:
            json.dump(all_data_predictions, f, indent=2)
        print(f"Saved all-data predictions to {all_pred_path}")
    
    # Save statistical analysis if provided
    if statistical_analysis is not None:
        stats_path = output_dir / "statistical_analysis.json"
        with open(stats_path, 'w') as f:
            json.dump(statistical_analysis, f, indent=2)
        print(f"Saved statistical analysis to {stats_path}")
    
    # Save comprehensive metrics summary (same as metrics.json but kept for compatibility)
    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    print(f"Saved comprehensive metrics summary to {summary_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    for model_name, metrics in results['metrics'].items():
        print(f"\n{model_name.upper()}:")
        for metric_name, value in metrics.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
    
    # Print best model if available
    if 'best_model' in results and results['best_model']:
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {results['best_model'].upper()}")
        print(f"{'='*60}")


def main(
    csv_path: str,
    sequence_col: str = "sequence",
    target_col: str = None,
    model_type: str = "both",
    mean_pool: bool = True,
    test_size: float = 0.2,
    output_dir: str = "./results",
    cache_embeddings: bool = True,
    predict_all: bool = True,
    predict_multiple_properties: bool = False,
    use_cross_validation: bool = False,
    n_folds: int = 5,
    create_plots: bool = True,
    min_length: int = 10,
    max_length: int = 5000,
    tune_hyperparameters: bool = False,
    n_trials: int = 50,
    use_composition_features: bool = True,
    use_ensemble: bool = True,
    use_bootstrap_ci: bool = False,
    export_excel: bool = False,
    export_parquet: bool = False,
    save_to_db: bool = False,
    create_html_report: bool = True
):
    """
    Main pipeline function
    
    Args:
        csv_path: Path to CSV file with sequences and targets
        sequence_col: Name of sequence column
        target_col: Name of target column (auto-detect if None)
        model_type: "prot_t5", "esm2", or "both"
        mean_pool: Whether to use mean pooling for embeddings
        test_size: Test set fraction
        output_dir: Output directory for results
        cache_embeddings: Whether to cache computed embeddings
        predict_all: Whether to predict on all data
        predict_multiple_properties: Whether to predict multiple properties
        use_cross_validation: Whether to run cross-validation
        n_folds: Number of folds for cross-validation
        create_plots: Whether to create visualization plots
        min_length: Minimum sequence length
        max_length: Maximum sequence length
    """
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("="*60)
    logger.info("PROTEIN PROPERTY PREDICTION BASELINE PIPELINE")
    logger.info("="*60)
    logger.info(f"Input CSV: {csv_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Embedding model: {model_type}")
    logger.info(f"Device: {device}")
    
    print("="*60)
    print("PROTEIN PROPERTY PREDICTION BASELINE PIPELINE")
    print("="*60)
    
    # Load dataset
    df = load_dataset(csv_path)
    logger.info(f"Loaded dataset: {len(df)} samples")
    
    # Extract embeddings with validation and diagnostics
    print("\n" + "="*60)
    print("EXTRACTING EMBEDDINGS")
    print("="*60)
    extractor = EmbeddingExtractor()
    embeddings, embedding_diagnostics = extractor.extract_embeddings_batch(
        df[sequence_col],
        model_type=model_type,
        mean_pool=mean_pool,
        cache=cache_embeddings,
        validate=True,
        min_length=min_length,
        max_length=max_length,
        logger=logger
    )
    
    logger.info("Embedding diagnostics:")
    for emb_name, diag in embedding_diagnostics.items():
        logger.info(f"  {emb_name}: {json.dumps(diag, indent=2)}")
    
    # Add amino acid composition features if requested
    if use_composition_features:
        print("Adding amino acid composition features...")
        embeddings = add_composition_features(embeddings, df[sequence_col])
        logger.info("Added amino acid composition features (20 additional features per embedding)")
    
    # Prepare features and targets
    print("\n" + "="*60)
    print("PREPARING FEATURES AND TARGETS")
    print("="*60)
    X, y, is_classification = prepare_features_and_targets(
        df, sequence_col, target_col, embeddings
    )
    
    # Split data
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    if is_classification:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Cross-validation if requested
    cv_results = None
    if use_cross_validation:
        cv_results = run_cross_validation(X, y, is_classification, n_folds, logger)
        logger.info("Cross-validation results:")
        logger.info(json.dumps(cv_results, indent=2))
    
    # Hyperparameter tuning if requested
    best_params = {}
    if tune_hyperparameters:
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        for model_type_name in ['xgboost', 'random_forest']:
            print(f"\nTuning {model_type_name}...")
            tuning_results = tune_hyperparameters(
                X_train, y_train, model_type_name, is_classification, n_trials
            )
            best_params[model_type_name] = tuning_results['best_params']
            logger.info(f"{model_type_name} best params: {tuning_results['best_params']}")
            print(f"Best score: {tuning_results['best_score']:.4f}")
    
    # Train models
    print("\n" + "="*60)
    print("TRAINING BASELINE MODELS")
    print("="*60)
    results = train_baseline_models(
        X_train, y_train, X_test, y_test,
        is_classification, output_dir, logger, create_plots,
        best_params=best_params if tune_hyperparameters else None,
        use_ensemble=use_ensemble,
        use_bootstrap_ci=use_bootstrap_ci,
        create_html_report=create_html_report
    )
    
    logger.info(f"Best model: {results.get('best_model', 'N/A')}")
    
    # Load trained models for all-data prediction
    trained_models = {}
    for model_name, model_path in results['models'].items():
        with open(model_path, 'rb') as f:
            trained_models[model_name] = pickle.load(f)
    
    # Predict on all data if requested
    all_data_predictions = None
    if predict_all:
        all_data_predictions = predict_all_data(
            trained_models, X, y, is_classification, output_dir
        )
    
    # Statistical analysis
    statistical_analysis = {}
    if not is_classification:
        statistical_analysis = compute_statistical_analysis(
            all_data_predictions if all_data_predictions else results['predictions']
        )
    
    # Handle multiple properties if requested
    if predict_multiple_properties:
        print("\n" + "="*60)
        print("PREDICTING MULTIPLE PROPERTIES")
        print("="*60)
        
        # Find all potential target columns
        potential_targets = [col for col in df.columns 
                             if col not in [sequence_col, 'id'] and df[col].dtype in ['float64', 'int64', 'bool', 'object']]
        
        print(f"Found potential target columns: {potential_targets}")
        
        multi_property_results = {}
        for prop_col in potential_targets:
            if prop_col == target_col:
                continue  # Skip the one we already processed
            
            print(f"\nProcessing property: {prop_col}")
            try:
                X_prop, y_prop, is_clf_prop = prepare_features_and_targets(
                    df, sequence_col, prop_col, embeddings
                )
                
                if len(X_prop) == 0:
                    print(f"  Skipping {prop_col}: No valid data")
                    continue
                
                # Use the best model (XGBoost) for quick prediction
                best_model_name = 'xgboost'
                if best_model_name in trained_models:
                    model = trained_models[best_model_name]
                    
                    # Check if model type matches
                    if is_clf_prop and not hasattr(model, 'predict_proba'):
                        # Need classification model
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_prop_encoded = le.fit_transform(y_prop)
                        
                        if is_classification:
                            # Use existing classification model
                            y_pred_prop = model.predict(X_prop)
                        else:
                            # Train new model for this property
                            X_train_prop, X_test_prop, y_train_prop, y_test_prop = train_test_split(
                                X_prop, y_prop_encoded, test_size=0.2, random_state=42, stratify=y_prop_encoded
                            )
                            model_prop = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                            model_prop.fit(X_train_prop, y_train_prop)
                            y_pred_prop = model_prop.predict(X_prop)
                    else:
                        y_pred_prop = model.predict(X_prop)
                    
                    # Calculate metrics
                    if is_clf_prop:
                        acc = accuracy_score(y_prop, y_pred_prop)
                        f1 = f1_score(y_prop, y_pred_prop, average='weighted')
                        multi_property_results[prop_col] = {
                            'type': 'classification',
                            'accuracy': float(acc),
                            'f1_score': float(f1),
                            'y_true': y_prop.tolist(),
                            'y_pred': y_pred_prop.tolist()
                        }
                    else:
                        mse = mean_squared_error(y_prop, y_pred_prop)
                        mae = mean_absolute_error(y_prop, y_pred_prop)
                        r2 = r2_score(y_prop, y_pred_prop)
                        
                        non_zero_mask = np.abs(y_prop) > 1e-10
                        if np.sum(non_zero_mask) > 0:
                            mape = np.mean(np.abs((y_prop[non_zero_mask] - y_pred_prop[non_zero_mask]) / y_prop[non_zero_mask])) * 100
                        else:
                            mape = np.nan
                        
                        pearson_corr, _ = pearsonr(y_prop, y_pred_prop)
                        
                        multi_property_results[prop_col] = {
                            'type': 'regression',
                            'mse': float(mse),
                            'mae': float(mae),
                            'r2': float(r2),
                            'mape': float(mape) if not np.isnan(mape) else None,
                            'pearson_correlation': float(pearson_corr),
                            'y_true': y_prop.tolist(),
                            'y_pred': y_pred_prop.tolist()
                        }
                    
                    print(f"  Completed prediction for {prop_col}")
            except Exception as e:
                print(f"  Error processing {prop_col}: {e}")
                continue
        
        # Save multi-property results
        if multi_property_results:
            multi_prop_path = Path(output_dir) / "multi_property_predictions.json"
            with open(multi_prop_path, 'w') as f:
                json.dump(multi_property_results, f, indent=2)
            print(f"\nSaved multi-property predictions to {multi_prop_path}")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    save_results(results, embeddings, output_dir, all_data_predictions, statistical_analysis)
    
    # Save embedding diagnostics
    diag_path = Path(output_dir) / "embedding_diagnostics.json"
    with open(diag_path, 'w') as f:
        json.dump(embedding_diagnostics, f, indent=2)
    logger.info(f"Saved embedding diagnostics to {diag_path}")
    
    # Save cross-validation results if available
    if cv_results is not None:
        cv_path = Path(output_dir) / "cross_validation_results.json"
        with open(cv_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        logger.info(f"Saved cross-validation results to {cv_path}")
    
    # Compute sequence similarity analysis
    print("\n" + "="*60)
    print("COMPUTING SEQUENCE SIMILARITY")
    print("="*60)
    sequences_list = df[sequence_col].tolist()
    similarity_matrix = compute_sequence_similarity(sequences_list[:100], method='jaccard')  # Limit to 100 for performance
    similarity_path = Path(output_dir) / "sequence_similarity_matrix.npy"
    np.save(similarity_path, similarity_matrix)
    logger.info(f"Saved sequence similarity matrix to {similarity_path}")
    
    # Export to additional formats if requested
    if all_data_predictions:
        if export_excel or export_parquet or save_to_db:
            print("\n" + "="*60)
            print("EXPORTING TO ADDITIONAL FORMATS")
            print("="*60)
        
        if export_excel:
            try:
                excel_path = Path(output_dir) / "predictions.xlsx"
                export_to_excel(all_data_predictions, str(excel_path), sequences_list)
            except Exception as e:
                logger.warning(f"Could not export to Excel: {e}")
        
        if export_parquet:
            try:
                parquet_path = Path(output_dir) / "predictions.parquet"
                export_to_parquet(all_data_predictions, str(parquet_path))
            except Exception as e:
                logger.warning(f"Could not export to Parquet: {e}")
        
        if save_to_db:
            try:
                db_path = Path(output_dir) / "predictions.db"
                conn = setup_database(str(db_path))
                for model_name in trained_models.keys():
                    save_predictions_to_db(
                        conn, all_data_predictions if all_data_predictions else results['predictions'],
                        sequences_list, model_name, y.tolist() if not is_classification else None
                    )
                conn.close()
                logger.info(f"Saved predictions to database: {db_path}")
            except Exception as e:
                logger.warning(f"Could not save to database: {e}")
    
    # Log summary
    logger.info("="*60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Best model: {results.get('best_model', 'N/A')}")
    logger.info(f"Number of sequences processed: {len(df)}")
    logger.info(f"Embeddings used: {model_type}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    
    return results, embeddings, all_data_predictions, statistical_analysis, embedding_diagnostics, cv_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Baseline ML pipeline for protein property prediction"
    )
    
    # Input arguments
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to CSV file with sequences and targets"
    )
    input_group.add_argument(
        "--fasta_path",
        type=str,
        default=None,
        help="Path to FASTA file for prediction (requires --model_path)"
    )
    input_group.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model for FASTA prediction"
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument(
        "--sequence_col",
        type=str,
        default="sequence",
        help="Name of sequence column (default: 'sequence')"
    )
    data_group.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="Name of target column (auto-detect if not provided)"
    )
    data_group.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum sequence length (default: 10)"
    )
    data_group.add_argument(
        "--max_length",
        type=int,
        default=5000,
        help="Maximum sequence length (default: 5000)"
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument(
        "--model_type",
        type=str,
        default="both",
        choices=["prot_t5", "esm2", "both"],
        help="Embedding model to use (default: 'both')"
    )
    model_group.add_argument(
        "--mean_pool",
        action="store_true",
        default=True,
        help="Use mean pooling for embeddings (default: True)"
    )
    model_group.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)"
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument(
        "--use_cross_validation",
        action="store_true",
        help="Run k-fold cross-validation"
    )
    train_group.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)"
    )
    train_group.add_argument(
        "--tune_hyperparameters",
        action="store_true",
        help="Tune hyperparameters using Optuna (slower but may improve performance)"
    )
    train_group.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of hyperparameter optimization trials (default: 50)"
    )
    train_group.add_argument(
        "--use_composition_features",
        action="store_true",
        default=True,
        help="Add amino acid composition features (default: True; use --no_composition_features to disable)"
    )
    train_group.add_argument(
        "--no_composition_features",
        action="store_true",
        help="Disable amino acid composition feature augmentation"
    )
    train_group.add_argument(
        "--use_ensemble",
        action="store_true",
        default=True,
        help="Create ensemble predictions from all models (default: True)"
    )
    train_group.add_argument(
        "--use_bootstrap_ci",
        action="store_true",
        help="Use bootstrap confidence intervals for XGBoost (slower but more accurate)"
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results (default: './results')"
    )
    output_group.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable embedding caching"
    )
    output_group.add_argument(
        "--no_predict_all",
        action="store_true",
        help="Skip prediction on all data (only predict on test set)"
    )
    output_group.add_argument(
        "--predict_multiple",
        action="store_true",
        help="Predict all relevant properties in the dataset"
    )
    output_group.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip creating visualization plots"
    )
    output_group.add_argument(
        "--export_excel",
        action="store_true",
        help="Export predictions to Excel format"
    )
    output_group.add_argument(
        "--export_parquet",
        action="store_true",
        help="Export predictions to Parquet format"
    )
    output_group.add_argument(
        "--save_to_db",
        action="store_true",
        help="Save predictions to SQLite database"
    )
    output_group.add_argument(
        "--create_html_report",
        action="store_true",
        default=True,
        help="Create interactive HTML report (default: True)"
    )
    
    args = parser.parse_args()
    
    # Handle FASTA prediction mode
    if args.fasta_path is not None:
        if args.model_path is None:
            parser.error("--model_path is required when using --fasta_path")
        
        predict_from_fasta(
            fasta_path=args.fasta_path,
            model_path=args.model_path,
            embedding_model_type=args.model_type,
            output_path=str(Path(args.output_dir) / "fasta_predictions.json"),
            min_length=args.min_length,
            max_length=args.max_length
        )
    else:
        # Normal training mode
        if args.csv_path is None:
            parser.error("--csv_path is required (or use --fasta_path with --model_path)")
    
    main(
        csv_path=args.csv_path,
        sequence_col=args.sequence_col,
        target_col=args.target_col,
        model_type=args.model_type,
        mean_pool=args.mean_pool,
        test_size=args.test_size,
        output_dir=args.output_dir,
        cache_embeddings=not args.no_cache,
        predict_all=not args.no_predict_all,
            predict_multiple_properties=args.predict_multiple,
            use_cross_validation=args.use_cross_validation,
            n_folds=args.n_folds,
            create_plots=not args.no_plots,
            min_length=args.min_length,
            max_length=args.max_length,
            tune_hyperparameters=args.tune_hyperparameters,
            n_trials=args.n_trials,
        use_composition_features=(args.use_composition_features and not args.no_composition_features),
            use_ensemble=args.use_ensemble,
            use_bootstrap_ci=args.use_bootstrap_ci,
            export_excel=args.export_excel,
            export_parquet=args.export_parquet,
            save_to_db=args.save_to_db,
            create_html_report=args.create_html_report
    )

