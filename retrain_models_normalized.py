"""
Retrain Random Forest and XGBoost on normalized data for fair comparison with MLP.

This script:
1. Loads embeddings and normalizes using training statistics only
2. Retrains Random Forest on normalized data
3. Retrains XGBoost on normalized data
4. Evaluates all models on normalized test set
5. Creates fair comparison table
"""

import numpy as np
import pandas as pd
import pickle
import torch
from pathlib import Path
from datetime import datetime
import json
import logging

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from core.mlp_baseline import load_embeddings_and_labels, BaselineMLP
from core.protein_baseline import (
    compute_comprehensive_metrics,
    predict_with_uncertainty,
    save_model_with_metadata,
    setup_logging
)


def normalize_features(X_train, X_val, X_test):
    """
    Normalize features using training statistics only.
    Prevents data leakage.
    
    Returns:
        Normalized X_train, X_val, X_test
    """
    # Compute statistics from training data only
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero
    
    # Normalize all sets using training statistics
    X_train_norm = (X_train - train_mean) / train_std
    X_val_norm = (X_val - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std
    
    return X_train_norm, X_val_norm, X_test_norm, train_mean, train_std


def train_random_forest_normalized(X_train, y_train, X_val, y_val):
    """
    Train Random Forest on normalized data.
    
    Uses same hyperparameters as original RF model.
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST ON NORMALIZED DATA")
    print("="*60)
    
    # Same hyperparameters as original
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred_val = rf_model.predict(X_val)
    val_metrics = compute_comprehensive_metrics(y_val, y_pred_val, None)
    
    print(f"\nValidation Metrics:")
    print(f"  MAE: {val_metrics['mae']:.4f} kcal/mol")
    print(f"  RMSE: {val_metrics['rmse']:.4f} kcal/mol")
    print(f"  R²: {val_metrics['r2']:.4f}")
    print(f"  Pearson r: {val_metrics['pearson_correlation']:.4f}")
    
    return rf_model


def train_xgboost_normalized(X_train, y_train, X_val, y_val):
    """
    Train XGBoost on normalized data.
    
    Uses same hyperparameters as original XGBoost model.
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST ON NORMALIZED DATA")
    print("="*60)
    
    # Same hyperparameters as original
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    print("Training XGBoost...")
    try:
        # Try new API (XGBoost >= 2.0)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    except TypeError:
        # Fallback to old API
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
    
    # Evaluate on validation set
    y_pred_val = xgb_model.predict(X_val)
    val_metrics = compute_comprehensive_metrics(y_val, y_pred_val, None)
    
    print(f"\nValidation Metrics:")
    print(f"  MAE: {val_metrics['mae']:.4f} kcal/mol")
    print(f"  RMSE: {val_metrics['rmse']:.4f} kcal/mol")
    print(f"  R²: {val_metrics['r2']:.4f}")
    print(f"  Pearson r: {val_metrics['pearson_correlation']:.4f}")
    
    return xgb_model


def load_mlp_model(device='cpu'):
    """Load trained MLP model."""
    model_dir = Path("training_output (CRITICAL DIRECTORY DO NOT TOUCH)")
    checkpoint = torch.load(model_dir / "mlp_baseline_model.pt", map_location=device)
    
    model = BaselineMLP(input_dim=2344, dropout=0.2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    return model


def predict_mlp(model, X, device='cpu'):
    """Predict using MLP model."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy().flatten()
    return predictions


def evaluate_all_models(X_test, y_test, rf_model, xgb_model, mlp_model, device='cpu'):
    """
    Evaluate all models on normalized test set.
    Returns comprehensive metrics for each model.
    """
    print("\n" + "="*60)
    print("EVALUATING ALL MODELS ON NORMALIZED TEST SET")
    print("="*60)
    
    results = {}
    
    # Random Forest
    print("\nEvaluating Random Forest...")
    y_pred_rf, y_std_rf = predict_with_uncertainty(rf_model, X_test, is_random_forest=True)
    rf_metrics = compute_comprehensive_metrics(y_test, y_pred_rf, y_std_rf)
    results['Random Forest'] = {
        'predictions': y_pred_rf,
        'uncertainty': y_std_rf,
        'metrics': rf_metrics
    }
    print(f"  MAE: {rf_metrics['mae']:.4f} kcal/mol")
    print(f"  RMSE: {rf_metrics['rmse']:.4f} kcal/mol")
    print(f"  R²: {rf_metrics['r2']:.4f}")
    print(f"  Pearson r: {rf_metrics['pearson_correlation']:.4f}")
    
    # XGBoost
    print("\nEvaluating XGBoost...")
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_metrics = compute_comprehensive_metrics(y_test, y_pred_xgb, None)
    results['XGBoost'] = {
        'predictions': y_pred_xgb,
        'uncertainty': None,
        'metrics': xgb_metrics
    }
    print(f"  MAE: {xgb_metrics['mae']:.4f} kcal/mol")
    print(f"  RMSE: {xgb_metrics['rmse']:.4f} kcal/mol")
    print(f"  R²: {xgb_metrics['r2']:.4f}")
    print(f"  Pearson r: {xgb_metrics['pearson_correlation']:.4f}")
    
    # MLP
    print("\nEvaluating MLP...")
    y_pred_mlp = predict_mlp(mlp_model, X_test, device)
    mlp_metrics = compute_comprehensive_metrics(y_test, y_pred_mlp, None)
    results['MLP'] = {
        'predictions': y_pred_mlp,
        'uncertainty': None,
        'metrics': mlp_metrics
    }
    print(f"  MAE: {mlp_metrics['mae']:.4f} kcal/mol")
    print(f"  RMSE: {mlp_metrics['rmse']:.4f} kcal/mol")
    print(f"  R²: {mlp_metrics['r2']:.4f}")
    print(f"  Pearson r: {mlp_metrics['pearson_correlation']:.4f}")
    
    return results


def create_fair_comparison_table(results, output_dir):
    """Create fair comparison table of all models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("FAIR COMPARISON TABLE (ALL MODELS ON NORMALIZED DATA)")
    print("="*60)
    
    # Create comparison table
    comparison_data = []
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': model_name,
            'MAE (kcal/mol)': f"{metrics['mae']:.4f}",
            'RMSE (kcal/mol)': f"{metrics['rmse']:.4f}",
            'R²': f"{metrics['r2']:.4f}",
            'Pearson r': f"{metrics['pearson_correlation']:.4f}",
            'Spearman r': f"{metrics['spearman_correlation']:.4f}"
        })
    
    # Sort by MAE (primary metric)
    comparison_data.sort(key=lambda x: float(x['MAE (kcal/mol)']))
    
    # Print table
    print("\n" + "-"*80)
    print(f"{'Model':<20} {'MAE':<15} {'RMSE':<15} {'R²':<10} {'Pearson r':<12}")
    print("-"*80)
    for row in comparison_data:
        print(f"{row['Model']:<20} {row['MAE (kcal/mol)']:<15} {row['RMSE (kcal/mol)']:<15} "
              f"{row['R²']:<10} {row['Pearson r']:<12}")
    print("-"*80)
    
    # Save to file
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = output_dir / "fair_comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved comparison table to: {comparison_path}")
    
    # Save detailed metrics JSON
    detailed_metrics = {
        model_name: {
            k: float(v) if isinstance(v, (np.integer, np.floating)) else v
            for k, v in result['metrics'].items()
        }
        for model_name, result in results.items()
    }
    
    metrics_path = output_dir / "normalized_models_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"Saved detailed metrics to: {metrics_path}")
    
    return comparison_df


def main():
    """Main retraining script."""
    print("="*60)
    print("RETRAINING MODELS ON NORMALIZED DATA FOR FAIR COMPARISON")
    print("="*60)
    
    # Setup
    output_dir = Path("training_output (CRITICAL DIRECTORY DO NOT TOUCH)")
    output_dir.mkdir(exist_ok=True)
    logger = setup_logging(str(output_dir))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data (this function already normalizes)
    print("\nLoading and normalizing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_embeddings_and_labels()
    
    print(f"\nData shapes:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Verify normalization (should be ~0 mean, ~1 std)
    print(f"\nVerifying normalization:")
    print(f"  Train mean: {X_train.mean():.6f} (should be ~0)")
    print(f"  Train std: {X_train.std():.6f} (should be ~1)")
    
    # Train Random Forest on normalized data
    rf_model = train_random_forest_normalized(X_train, y_train, X_val, y_val)
    
    # Train XGBoost on normalized data
    xgb_model = train_xgboost_normalized(X_train, y_train, X_val, y_val)
    
    # Load MLP model (already trained on normalized data)
    print("\n" + "="*60)
    print("LOADING MLP MODEL (ALREADY TRAINED ON NORMALIZED DATA)")
    print("="*60)
    mlp_model = load_mlp_model(device)
    print("MLP model loaded")
    
    # Evaluate all models
    results = evaluate_all_models(X_test, y_test, rf_model, xgb_model, mlp_model, device)
    
    # Create fair comparison table
    comparison_df = create_fair_comparison_table(results, output_dir)
    
    # Save retrained models
    print("\n" + "="*60)
    print("SAVING RETRAINED MODELS")
    print("="*60)
    
    # Save Random Forest
    rf_path = output_dir / "model_random_forest_normalized.pkl"
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"Saved normalized RF to: {rf_path}")
    
    # Save XGBoost
    xgb_path = output_dir / "xgboost_model_normalized.pkl"
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"Saved normalized XGBoost to: {xgb_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - FAIR COMPARISON (ALL MODELS ON NORMALIZED DATA)")
    print("="*60)
    
    # Sort by MAE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['mae'])
    
    print("\nRanking by MAE (primary metric):")
    for rank, (model_name, result) in enumerate(sorted_results, 1):
        metrics = result['metrics']
        print(f"  {rank}. {model_name}:")
        print(f"     MAE: {metrics['mae']:.4f} kcal/mol")
        print(f"     RMSE: {metrics['rmse']:.4f} kcal/mol")
        print(f"     R²: {metrics['r2']:.4f}")
    
    best_model = sorted_results[0][0]
    best_mae = sorted_results[0][1]['metrics']['mae']
    print(f"\n[SUCCESS] Best model: {best_model} (MAE: {best_mae:.4f} kcal/mol)")
    
    print(f"\n[SUCCESS] Retraining complete! Fair comparison established.")
    print(f"Results saved to: {output_dir}/")
    
    return results, rf_model, xgb_model


if __name__ == "__main__":
    main()

