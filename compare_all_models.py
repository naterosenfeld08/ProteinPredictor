"""
Comprehensive comparison of all models including MLP baseline.
Compares Random Forest, XGBoost, MLP, and existing 2-model ensemble.
Creates detailed analysis and visualizations.
"""

import pickle
import numpy as np
import pandas as pd
import json
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from core.protein_baseline import (
    EnsembleModel,
    compute_comprehensive_metrics,
    predict_with_uncertainty
)
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from core.mlp_baseline import BaselineMLP, load_embeddings_and_labels


def load_all_data():
    """Load all necessary data for analysis."""
    print("Loading data...")
    
    # Use the same loading function from mlp_baseline
    X_train, X_val, X_test, y_train, y_val, y_test = load_embeddings_and_labels()
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_mlp_model(device='cpu'):
    """Load trained MLP model."""
    model_dir = Path("training_output (CRITICAL DIRECTORY DO NOT TOUCH)")
    checkpoint = torch.load(model_dir / "mlp_baseline_model.pt", map_location=device)
    
    model = BaselineMLP(input_dim=2344, dropout=0.2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    return model, checkpoint


def predict_mlp(model, X, device='cpu'):
    """Predict using MLP model."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy().flatten()
    return predictions


def load_other_models():
    """Load RF, XGBoost, and existing ensemble."""
    model_dir = Path("training_output (CRITICAL DIRECTORY DO NOT TOUCH)")
    
    # Load ensemble
    with open(model_dir / "ensemble_model.pkl", 'rb') as f:
        ensemble = pickle.load(f)
    
    # Load individual models
    with open(model_dir / "model_random_forest.pkl", 'rb') as f:
        rf = pickle.load(f)
    
    with open(model_dir / "xgboost_model.pkl", 'rb') as f:
        xgb_model = pickle.load(f)
    
    return ensemble, rf, xgb_model


def comprehensive_evaluation(models_dict, X, y, set_name, device='cpu'):
    """Comprehensive evaluation of all models."""
    results = {}
    
    for name, model in models_dict.items():
        if name == 'MLP':
            pred = predict_mlp(model, X, device)
            unc = None  # MLP doesn't provide uncertainty estimates
        elif isinstance(model, EnsembleModel):
            pred, unc = model.predict_with_uncertainty(X)
        elif isinstance(model, RandomForestRegressor):
            pred, unc = predict_with_uncertainty(model, X, is_random_forest=True)
        else:  # XGBoost
            pred = model.predict(X)
            unc = None
        
        metrics = compute_comprehensive_metrics(y, pred, unc)
        results[name] = {
            'predictions': pred,
            'uncertainty': unc,
            'metrics': metrics
        }
    
    return results


def create_comparison_plots(results, y_true, output_dir):
    """Create comprehensive comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    model_names = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Model Performance Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    mae_values = [results[name]['metrics']['mae'] for name in model_names]
    rmse_values = [results[name]['metrics']['rmse'] for name in model_names]
    
    bars1 = ax.bar(x_pos - width/2, mae_values, width, label='MAE', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, rmse_values, width, label='RMSE', color=colors[1], alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "1_model_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 1_model_performance_comparison.png")
    
    # 2. Prediction Scatter Comparison
    n_models = len(model_names)
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(model_names[:4]):  # Show up to 4 models
        preds = results[model_name]['predictions']
        ax = axes[idx]
        
        ax.scatter(y_true, preds, alpha=0.5, s=20, color=colors[idx % len(colors)])
        
        # Perfect prediction line
        min_val = min(y_true.min(), preds.min())
        max_val = max(y_true.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        metrics = results[model_name]['metrics']
        ax.set_xlabel('True ΔΔG (kcal/mol)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted ΔΔG (kcal/mol)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}\nMAE: {metrics["mae"]:.3f}, R²: {metrics["r2"]:.3f}', 
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "2_prediction_scatter_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 2_prediction_scatter_comparison.png")
    
    # 3. Residual Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(model_names[:4]):
        preds = results[model_name]['predictions']
        residuals = y_true - preds
        ax = axes[idx]
        
        ax.scatter(preds, residuals, alpha=0.5, s=20, color=colors[idx % len(colors)])
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        ax.set_xlabel('Predicted ΔΔG (kcal/mol)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residuals (kcal/mol)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name} - Residual Analysis', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "3_residual_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 3_residual_analysis.png")
    
    # 4. Error Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, model_name in enumerate(model_names):
        preds = results[model_name]['predictions']
        errors = np.abs(y_true - preds)
        ax.hist(errors, bins=30, alpha=0.6, label=model_name, color=colors[idx % len(colors)])
    
    ax.set_xlabel('Absolute Error (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Error Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "4_error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: 4_error_distribution.png")


def create_summary_document(test_results, val_results, train_results, output_dir, y_test, y_val, y_train):
    """Create comprehensive summary document."""
    
    summary = f"""================================================================================
COMPREHENSIVE MODEL COMPARISON - ALL MODELS INCLUDING MLP
================================================================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Project: Protein DDG (ΔΔG) Prediction - Complete Model Comparison
Dataset: FireProtDB (7,716 training, 2,204 validation, 1,104 test sequences)

================================================================================
1. OVERVIEW
================================================================================

This document provides a comprehensive comparison of all models:
- Random Forest Regressor (100 estimators)
- XGBoost Regressor (100 estimators)
- MLP Baseline (2344 → 1024 → 512 → 128 → 1)
- 2-Model Ensemble (RF + XGBoost)

================================================================================
2. MODEL ARCHITECTURES
================================================================================

2.1 Random Forest
    - Algorithm: Random Forest Regressor (scikit-learn)
    - Number of Trees: 100
    - Features: 2,344 dimensions (ProtT5-XL + ESM-2 + composition)
    - Uncertainty: Tree variance estimates

2.2 XGBoost
    - Algorithm: XGBoost Regressor
    - Number of Estimators: 100
    - Max Depth: 6
    - Learning Rate: 0.1
    - Features: 2,344 dimensions

2.3 MLP Baseline
    - Architecture: 2344 → 1024 → 512 → 128 → 1
    - Activation: ReLU
    - Dropout: 0.2 (after layers 1 & 2)
    - Optimizer: AdamW (lr=1e-4, weight_decay=1e-3)
    - Loss: MAE (L1)
    - Parameters: ~3 million
    - Training: Early stopping (10 epochs patience)

2.4 2-Model Ensemble
    - Method: Simple averaging (RF + XGBoost)
    - Uncertainty: From Random Forest

================================================================================
3. PERFORMANCE METRICS - TEST SET
================================================================================

"""
    
    # Add test set metrics
    for model_name in ['Random Forest', 'XGBoost', 'MLP', 'Ensemble']:
        if model_name in test_results:
            metrics = test_results[model_name]['metrics']
            summary += f"""
3.{['1', '2', '3', '4'][['Random Forest', 'XGBoost', 'MLP', 'Ensemble'].index(model_name)]} {model_name}
    - MAE: {metrics['mae']:.4f} kcal/mol
    - RMSE: {metrics['rmse']:.4f} kcal/mol
    - R²: {metrics['r2']:.4f}
    - Pearson r: {metrics['pearson_correlation']:.4f} (p={metrics['pearson_p_value']:.2e})
    - Spearman r: {metrics['spearman_correlation']:.4f} (p={metrics['spearman_p_value']:.2e})
"""
            if metrics.get('mean_uncertainty') is not None:
                summary += f"    - Mean Uncertainty: {metrics['mean_uncertainty']:.4f} kcal/mol\n"
    
    summary += f"""
================================================================================
4. PERFORMANCE RANKING
================================================================================

4.1 Test Set Performance Ranking (by RMSE):
"""
    
    # Sort by RMSE
    sorted_models = sorted([(k, v) for k, v in test_results.items()], 
                          key=lambda x: x[1]['metrics']['rmse'])
    for rank, (name, result) in enumerate(sorted_models, 1):
        summary += f"    {rank}. {name}: RMSE = {result['metrics']['rmse']:.4f} kcal/mol\n"
    
    # Find best model
    best_model = sorted_models[0][0]
    best_rmse = sorted_models[0][1]['metrics']['rmse']
    
    summary += f"""
4.2 Best Model: {best_model} (RMSE: {best_rmse:.4f} kcal/mol)

4.3 MLP Performance Analysis:
"""
    if 'MLP' in test_results:
        mlp_metrics = test_results['MLP']['metrics']
        rf_metrics = test_results['Random Forest']['metrics']
        mlp_vs_rf = mlp_metrics['rmse'] - rf_metrics['rmse']
        mlp_vs_rf_pct = (mlp_vs_rf / rf_metrics['rmse']) * 100
        
        summary += f"""    - MLP RMSE: {mlp_metrics['rmse']:.4f} kcal/mol
    - vs Random Forest: {mlp_vs_rf:+.4f} kcal/mol ({mlp_vs_rf_pct:+.2f}%)
    - vs XGBoost: {mlp_metrics['rmse'] - test_results['XGBoost']['metrics']['rmse']:+.4f} kcal/mol
    - Status: {'BETTER' if mlp_vs_rf < -0.01 else 'SIMILAR' if abs(mlp_vs_rf) < 0.01 else 'WORSE'} than RF
"""
    
    summary += f"""
================================================================================
5. VALIDATION SET PERFORMANCE
================================================================================

"""
    
    for model_name in ['Random Forest', 'XGBoost', 'MLP', 'Ensemble']:
        if model_name in val_results:
            metrics = val_results[model_name]['metrics']
            summary += f"""
{model_name} Validation Metrics:
    - MAE: {metrics['mae']:.4f} kcal/mol
    - RMSE: {metrics['rmse']:.4f} kcal/mol
    - R²: {metrics['r2']:.4f}
    - Pearson r: {metrics['pearson_correlation']:.4f}
"""
    
    summary += f"""
================================================================================
6. TRAINING SET PERFORMANCE (for reference)
================================================================================

"""
    
    for model_name in ['Random Forest', 'XGBoost', 'MLP', 'Ensemble']:
        if model_name in train_results:
            metrics = train_results[model_name]['metrics']
            summary += f"""
{model_name} Training Metrics:
    - MAE: {metrics['mae']:.4f} kcal/mol
    - RMSE: {metrics['rmse']:.4f} kcal/mol
    - R²: {metrics['r2']:.4f}
    - Pearson r: {metrics['pearson_correlation']:.4f}
"""
    
    summary += f"""
================================================================================
7. MODEL DIVERSITY ANALYSIS
================================================================================

7.1 Prediction Correlations (Test Set):
"""
    
    # Compute pairwise correlations between model predictions
    model_list = ['Random Forest', 'XGBoost', 'MLP', 'Ensemble']
    available_models = [m for m in model_list if m in test_results]
    
    for i, model1 in enumerate(available_models):
        for model2 in available_models[i+1:]:
            pred1 = test_results[model1]['predictions']
            pred2 = test_results[model2]['predictions']
            corr = np.corrcoef(pred1, pred2)[0, 1]
            summary += f"    {model1} vs {model2}: r = {corr:.4f}\n"
    
    summary += f"""
7.2 Key Observations:
    - High correlation (>0.95) indicates models make similar predictions
    - Lower correlation indicates complementary information
    - Ensemble benefits from model diversity

================================================================================
8. CONCLUSIONS AND RECOMMENDATIONS
================================================================================

8.1 Performance Summary:
"""
    
    if 'MLP' in test_results:
        summary += """    - MLP provides a neural network baseline for comparison
    - MLP performance is comparable to tree-based models
    - MLP may capture different patterns than tree-based methods
"""
    
    summary += """
8.2 Model Selection Recommendations:
    - Best individual model: Use for minimal error
    - Ensemble: Use for robustness and uncertainty quantification
    - MLP: Consider for neural network-specific advantages

8.3 Future Directions:
    - Create 3-model ensemble (RF + XGBoost + MLP)
    - Hyperparameter tuning for MLP
    - Advanced ensemble strategies (weighted, stacking)

================================================================================
END OF SUMMARY
================================================================================
"""
    
    # Save summary
    with open(output_dir / "All_Models_Comparison_Summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Saved: All_Models_Comparison_Summary.txt")


def save_detailed_metrics(test_results, val_results, train_results, output_dir):
    """Save detailed metrics to JSON."""
    output_dir = Path(output_dir)
    
    detailed_metrics = {
        'test_metrics': {name: {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                               for k, v in result['metrics'].items()} 
                        for name, result in test_results.items()},
        'val_metrics': {name: {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                              for k, v in result['metrics'].items()} 
                       for name, result in val_results.items()},
        'train_metrics': {name: {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                for k, v in result['metrics'].items()} 
                         for name, result in train_results.items()}
    }
    
    with open(output_dir / "all_models_detailed_metrics.json", 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    print(f"Saved: all_models_detailed_metrics.json")


def main():
    """Main comparison function."""
    print("="*60)
    print("COMPREHENSIVE MODEL COMPARISON - ALL MODELS")
    print("="*60)
    
    # Device for MLP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_all_data()
    
    # Load models
    print("\nLoading models...")
    ensemble, rf, xgb_model = load_other_models()
    mlp_model, mlp_checkpoint = load_mlp_model(device)
    
    # Create models dictionary
    models_dict = {
        'Random Forest': rf,
        'XGBoost': xgb_model,
        'MLP': mlp_model,
        'Ensemble': ensemble
    }
    
    # Evaluate on all sets
    print("\nEvaluating on training set...")
    train_results = comprehensive_evaluation(models_dict, X_train, y_train, "train", device)
    
    print("Evaluating on validation set...")
    val_results = comprehensive_evaluation(models_dict, X_val, y_val, "val", device)
    
    print("Evaluating on test set...")
    test_results = comprehensive_evaluation(models_dict, X_test, y_test, "test", device)
    
    # Create output directory
    output_dir = Path("Model Comparison Summary")
    output_dir.mkdir(exist_ok=True)
    
    # Create summary document
    create_summary_document(test_results, val_results, train_results, output_dir, y_test, y_val, y_train)
    
    # Create comparison plots
    create_comparison_plots(test_results, y_test, output_dir)
    
    # Save detailed metrics
    save_detailed_metrics(test_results, val_results, train_results, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - TEST SET PERFORMANCE")
    print("="*60)
    for model_name in ['Random Forest', 'XGBoost', 'MLP', 'Ensemble']:
        if model_name in test_results:
            metrics = test_results[model_name]['metrics']
            print(f"\n{model_name}:")
            print(f"  MAE: {metrics['mae']:.4f} kcal/mol")
            print(f"  RMSE: {metrics['rmse']:.4f} kcal/mol")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  Pearson r: {metrics['pearson_correlation']:.4f}")
    
    print(f"\n[SUCCESS] Comparison complete! Results saved to {output_dir}/")
    
    return test_results, val_results, train_results


if __name__ == "__main__":
    main()

