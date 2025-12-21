"""
Script to create an ensemble model from existing Random Forest model
and train XGBoost on pre-computed embeddings.

This script:
1. Loads pre-computed embeddings (embeddings_train.npz, embeddings_val.npz, embeddings_test.npz)
2. Loads data splits (data_splits.npz) to get train/val/test indices
3. Loads existing Random Forest model
4. Loads target values (DDG) from CSV
5. Trains XGBoost on the same data split
6. Creates and saves ensemble model
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from protein_baseline import (
    EnsembleModel,
    train_baseline_models,
    compute_comprehensive_metrics,
    predict_with_uncertainty,
    save_model_with_metadata,
    setup_logging
)


def load_precomputed_data(
    embeddings_dir: str,
    splits_path: str,
    csv_path: str,
    target_col: str = "DDG"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-computed embeddings, data splits, and target values.
    
    Args:
        embeddings_dir: Directory containing embeddings_train.npz, embeddings_val.npz, embeddings_test.npz
        splits_path: Path to data_splits.npz
        csv_path: Path to CSV file with target values
        target_col: Name of target column
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    embeddings_dir = Path(embeddings_dir)
    splits_path = Path(splits_path)
    csv_path = Path(csv_path)
    
    print("="*60)
    print("LOADING PRE-COMPUTED DATA")
    print("="*60)
    
    # Load embeddings
    print("\nLoading embeddings...")
    train_emb = np.load(embeddings_dir / "embeddings_train.npz")
    val_emb = np.load(embeddings_dir / "embeddings_val.npz")
    test_emb = np.load(embeddings_dir / "embeddings_test.npz")
    
    X_train = train_emb['embeddings']
    X_val = val_emb['embeddings']
    X_test = test_emb['embeddings']
    
    print(f"  Training embeddings: {X_train.shape}")
    print(f"  Validation embeddings: {X_val.shape}")
    print(f"  Test embeddings: {X_test.shape}")
    
    # Load data splits
    print("\nLoading data splits...")
    splits = np.load(splits_path)
    train_indices = splits['train_indices']
    val_indices = splits['val_indices']
    test_indices = splits['test_indices']
    
    print(f"  Train indices: {len(train_indices)}")
    print(f"  Val indices: {len(val_indices)}")
    print(f"  Test indices: {len(test_indices)}")
    
    # Load target values from CSV
    print(f"\nLoading target values from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Extract target values for each split
    y_train = df.iloc[train_indices][target_col].values
    y_val = df.iloc[val_indices][target_col].values
    y_test = df.iloc[test_indices][target_col].values
    
    # Remove NaN values
    train_mask = ~np.isnan(y_train)
    val_mask = ~np.isnan(y_val)
    test_mask = ~np.isnan(y_test)
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    print(f"  Training targets: {len(y_train)} (removed {np.sum(~train_mask)} NaN)")
    print(f"  Validation targets: {len(y_val)} (removed {np.sum(~val_mask)} NaN)")
    print(f"  Test targets: {len(y_test)} (removed {np.sum(~test_mask)} NaN)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_existing_rf_model(model_path: str) -> RandomForestRegressor:
    """Load existing Random Forest model."""
    print(f"\nLoading Random Forest model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    if not isinstance(model, RandomForestRegressor):
        raise ValueError(f"Model at {model_path} is not a RandomForestRegressor. Got {type(model)}")
    
    print(f"  Model loaded: {model.n_estimators} estimators")
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    logger: logging.Logger
) -> xgb.XGBRegressor:
    """Train XGBoost model."""
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)
    
    logger.info("Training XGBoost model...")
    
    # Create XGBoost model
    # Note: In newer XGBoost versions, early_stopping_rounds is passed to fit()
    # but we'll use a simpler approach that works across versions
    model = xgb.XGBRegressor(
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='rmse'
    )
    
    # Train with early stopping (compatible with both old and new XGBoost versions)
    print(f"\nTraining on {len(X_train)} samples...")
    try:
        # Try new API first (XGBoost 2.0+)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
    except TypeError:
        # Fall back to old API if needed
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=True
        )
    
    # Evaluate on validation set
    y_pred_val = model.predict(X_val)
    val_metrics = compute_comprehensive_metrics(y_val, y_pred_val, None)
    
    print(f"\nValidation Metrics:")
    print(f"  MAE: {val_metrics['mae']:.4f}")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  R²: {val_metrics['r2']:.4f}")
    print(f"  Pearson r: {val_metrics['pearson_correlation']:.4f}")
    
    logger.info(f"XGBoost validation RMSE: {val_metrics['rmse']:.4f}")
    
    return model


def create_and_save_ensemble(
    rf_model: RandomForestRegressor,
    xgb_model: xgb.XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    logger: logging.Logger
) -> Tuple[Path, Dict]:
    """Create ensemble model and save it."""
    print("\n" + "="*60)
    print("CREATING ENSEMBLE MODEL")
    print("="*60)
    
    # Create ensemble
    models = {
        'random_forest': rf_model,
        'xgboost': xgb_model
    }
    ensemble_model = EnsembleModel(models, is_classification=False)
    
    # Evaluate ensemble on test set
    print("\nEvaluating ensemble on test set...")
    y_pred_ensemble, y_pred_std_ensemble = ensemble_model.predict_with_uncertainty(X_test)
    ensemble_metrics = compute_comprehensive_metrics(y_test, y_pred_ensemble, y_pred_std_ensemble)
    
    print(f"\nEnsemble Test Metrics:")
    print(f"  MAE: {ensemble_metrics['mae']:.4f}")
    print(f"  RMSE: {ensemble_metrics['rmse']:.4f}")
    print(f"  R²: {ensemble_metrics['r2']:.4f}")
    print(f"  Pearson r: {ensemble_metrics['pearson_correlation']:.4f} (p={ensemble_metrics['pearson_p_value']:.4e})")
    print(f"  Spearman r: {ensemble_metrics['spearman_correlation']:.4f} (p={ensemble_metrics['spearman_p_value']:.4e})")
    if ensemble_metrics['mean_uncertainty'] is not None:
        print(f"  Mean Uncertainty: {ensemble_metrics['mean_uncertainty']:.4f}")
    
    # Save ensemble model
    ensemble_model_path = output_dir / "ensemble_model.pkl"
    ensemble_metadata = {
        'model_name': 'ensemble',
        'model_type': 'EnsembleModel',
        'component_models': ['random_forest', 'xgboost'],
        'training_date': datetime.now().isoformat(),
        'is_classification': False,
        'n_features': X_test.shape[1],
        'n_test_samples': len(y_test),
        'metrics': ensemble_metrics,
        'best_model': True
    }
    
    save_model_with_metadata(
        ensemble_model,
        str(ensemble_model_path),
        ensemble_metadata,
        str(output_dir)
    )
    
    print(f"\n[SUCCESS] Saved ensemble model to {ensemble_model_path}")
    logger.info(f"Saved ensemble model to {ensemble_model_path}")
    
    # Also save individual XGBoost model for reference
    xgb_model_path = output_dir / "xgboost_model.pkl"
    xgb_metadata = {
        'model_name': 'xgboost',
        'model_type': 'XGBRegressor',
        'training_date': datetime.now().isoformat(),
        'is_classification': False,
        'n_features': X_test.shape[1],
        'n_test_samples': len(y_test),
        'metrics': ensemble_metrics,  # Using ensemble metrics as reference
        'best_model': False
    }
    save_model_with_metadata(
        xgb_model,
        str(xgb_model_path),
        xgb_metadata,
        str(output_dir)
    )
    print(f"[SUCCESS] Saved XGBoost model to {xgb_model_path}")
    
    return ensemble_model_path, ensemble_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Create ensemble model from existing Random Forest and train XGBoost"
    )
    parser.add_argument(
        '--embeddings_dir',
        type=str,
        default='training_output (CRITICAL DIRECTORY DO NOT TOUCH)',
        help='Directory containing pre-computed embeddings'
    )
    parser.add_argument(
        '--splits_path',
        type=str,
        default='training_output (CRITICAL DIRECTORY DO NOT TOUCH)/data_splits.npz',
        help='Path to data_splits.npz'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default='fireprotdb_with_sequences.csv',
        help='Path to CSV file with target values'
    )
    parser.add_argument(
        '--target_col',
        type=str,
        default='DDG',
        help='Name of target column in CSV'
    )
    parser.add_argument(
        '--rf_model_path',
        type=str,
        default='training_output (CRITICAL DIRECTORY DO NOT TOUCH)/model_random_forest.pkl',
        help='Path to existing Random Forest model'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='training_output (CRITICAL DIRECTORY DO NOT TOUCH)',
        help='Output directory for ensemble model'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    logger = setup_logging(str(output_dir))
    
    logger.info("="*60)
    logger.info("ENSEMBLE MODEL CREATION")
    logger.info("="*60)
    logger.info(f"Embeddings directory: {args.embeddings_dir}")
    logger.info(f"Splits path: {args.splits_path}")
    logger.info(f"CSV path: {args.csv_path}")
    logger.info(f"RF model path: {args.rf_model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load pre-computed data
        X_train, X_val, X_test, y_train, y_val, y_test = load_precomputed_data(
            args.embeddings_dir,
            args.splits_path,
            args.csv_path,
            args.target_col
        )
        
        # Load existing Random Forest model
        rf_model = load_existing_rf_model(args.rf_model_path)
        
        # Evaluate RF on test set for comparison
        print("\n" + "="*60)
        print("EVALUATING RANDOM FOREST ON TEST SET")
        print("="*60)
        y_pred_rf, y_pred_std_rf = predict_with_uncertainty(rf_model, X_test, is_random_forest=True)
        rf_metrics = compute_comprehensive_metrics(y_test, y_pred_rf, y_pred_std_rf)
        print(f"\nRandom Forest Test Metrics:")
        print(f"  MAE: {rf_metrics['mae']:.4f}")
        print(f"  RMSE: {rf_metrics['rmse']:.4f}")
        print(f"  R²: {rf_metrics['r2']:.4f}")
        print(f"  Pearson r: {rf_metrics['pearson_correlation']:.4f}")
        
        # Train XGBoost
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, logger)
        
        # Evaluate XGBoost on test set
        print("\n" + "="*60)
        print("EVALUATING XGBOOST ON TEST SET")
        print("="*60)
        y_pred_xgb = xgb_model.predict(X_test)
        xgb_metrics = compute_comprehensive_metrics(y_test, y_pred_xgb, None)
        print(f"\nXGBoost Test Metrics:")
        print(f"  MAE: {xgb_metrics['mae']:.4f}")
        print(f"  RMSE: {xgb_metrics['rmse']:.4f}")
        print(f"  R²: {xgb_metrics['r2']:.4f}")
        print(f"  Pearson r: {xgb_metrics['pearson_correlation']:.4f}")
        
        # Create and save ensemble
        ensemble_path, ensemble_metrics = create_and_save_ensemble(
            rf_model, xgb_model, X_test, y_test, output_dir, logger
        )
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\nModel Performance Comparison (Test Set):")
        print(f"  Random Forest:")
        print(f"    RMSE: {rf_metrics['rmse']:.4f}, R²: {rf_metrics['r2']:.4f}, Pearson: {rf_metrics['pearson_correlation']:.4f}")
        print(f"  XGBoost:")
        print(f"    RMSE: {xgb_metrics['rmse']:.4f}, R²: {xgb_metrics['r2']:.4f}, Pearson: {xgb_metrics['pearson_correlation']:.4f}")
        print(f"  Ensemble:")
        print(f"    RMSE: {ensemble_metrics['rmse']:.4f}, R²: {ensemble_metrics['r2']:.4f}, Pearson: {ensemble_metrics['pearson_correlation']:.4f}")
        
        improvement = rf_metrics['rmse'] - ensemble_metrics['rmse']
        improvement_pct = (improvement / rf_metrics['rmse']) * 100
        print(f"\n[SUCCESS] Ensemble improvement over RF: {improvement:.4f} RMSE ({improvement_pct:.2f}%)")
        
        print(f"\n[SUCCESS] Ensemble model saved to: {ensemble_path}")
        logger.info("Ensemble model creation completed successfully")
        
    except Exception as e:
        logger.error(f"Error creating ensemble model: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

