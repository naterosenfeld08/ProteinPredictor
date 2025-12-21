"""
Lock the baseline by documenting the final best method and freezing hyperparameters.

This script:
1. Identifies the best method (ensemble or individual)
2. Documents final baseline configuration
3. Creates baseline lock document
4. Saves final model configuration
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def load_final_results():
    """Load final evaluation results."""
    results_dir = Path("Ensemble Evaluation Summary")
    
    # Load comparison table
    comparison_df = pd.read_csv(results_dir / "ensemble_comparison_table.csv")
    
    # Load detailed results
    with open(results_dir / "detailed_results.json", 'r') as f:
        detailed_results = json.load(f)
    
    return comparison_df, detailed_results


def create_baseline_lock_document():
    """Create comprehensive baseline lock document."""
    
    print("="*60)
    print("LOCKING BASELINE")
    print("="*60)
    
    # Load results
    results_dir = Path("Ensemble Evaluation Summary")
    comparison_df = pd.read_csv(results_dir / "ensemble_comparison_table.csv")
    
    # Find best method
    best_method = comparison_df.iloc[0]
    
    baseline_lock = f"""================================================================================
BASELINE LOCK DOCUMENT
================================================================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Project: Protein DDG (ΔΔG) Prediction - Baseline Establishment
Status: BASELINE LOCKED

================================================================================
1. FINAL BASELINE METHOD
================================================================================

Best Method: {best_method['Method']}
Test MAE: {best_method['MAE (kcal/mol)']} kcal/mol
Test RMSE: {best_method['RMSE (kcal/mol)']} kcal/mol
Test R²: {best_method['R²']}

Method Type: {'Ensemble' if 'Average' in best_method['Method'] or 'Stacking' in best_method['Method'] else 'Individual Model'}

================================================================================
2. BASELINE CONFIGURATION
================================================================================

2.1 Data Configuration
    - Dataset: FireProtDB
    - Training Samples: 7,716
    - Validation Samples: 2,204
    - Test Samples: 1,104
    - Features: 2,344 dimensions (ProtT5-XL + ESM-2 + composition)
    - Normalization: Z-score using training statistics only
    - Target: ΔΔG (kcal/mol)

2.2 Model Components
"""
    
    if 'Average' in best_method['Method']:
        baseline_lock += """    - Random Forest: 100 estimators, normalized data
    - XGBoost: 100 estimators, normalized data
    - MLP: 2344→1024→512→128→1, normalized data
    - Ensemble Method: Weighted averaging (weights from validation MAE)
"""
    elif 'Stacking' in best_method['Method']:
        baseline_lock += """    - Random Forest: 100 estimators, normalized data
    - XGBoost: 100 estimators, normalized data
    - MLP: 2344→1024→512→128→1, normalized data
    - Meta-Learner: Linear regression on validation predictions
"""
    else:
        baseline_lock += f"""    - Model: {best_method['Method']}
    - Configuration: See individual model documentation
"""
    
    baseline_lock += f"""
2.3 Hyperparameters (FROZEN)
"""
    
    if 'Average' in best_method['Method']:
        baseline_lock += """    - Ensemble Weights:
      * Random Forest: 33.29%
      * XGBoost: 33.28%
      * MLP: 33.43%
    - Weight Calculation: Inverse of validation MAE
"""
    elif 'Stacking' in best_method['Method']:
        baseline_lock += """    - Meta-Learner Coefficients:
      * Random Forest: 0.8282
      * XGBoost: 0.1919
      * MLP: -0.0029
      * Intercept: 0.0079
"""
    
    baseline_lock += f"""
================================================================================
3. PERFORMANCE METRICS (TEST SET)
================================================================================

Primary Metric (MAE): {best_method['MAE (kcal/mol)']} kcal/mol
Secondary Metric (RMSE): {best_method['RMSE (kcal/mol)']} kcal/mol
R² Score: {best_method['R²']}
Pearson Correlation: {best_method['Pearson r']}
Spearman Correlation: {best_method['Spearman r']}

================================================================================
4. COMPARISON WITH ALTERNATIVES
================================================================================

"""
    
    # Add comparison with other methods
    for idx, row in comparison_df.iterrows():
        if row['Method'] != best_method['Method']:
            mae_diff = float(row['MAE (kcal/mol)']) - float(best_method['MAE (kcal/mol)'])
            mae_pct = (mae_diff / float(best_method['MAE (kcal/mol)'])) * 100
            baseline_lock += f"{row['Method']}:\n"
            baseline_lock += f"  MAE: {row['MAE (kcal/mol)']} kcal/mol ({mae_pct:+.2f}% vs baseline)\n"
    
    baseline_lock += f"""
================================================================================
5. MODEL FILES
================================================================================

Baseline Model Files:
"""
    
    if 'Average' in best_method['Method']:
        baseline_lock += """  - Random Forest: training_output (CRITICAL DIRECTORY DO NOT TOUCH)/model_random_forest_normalized.pkl
  - XGBoost: training_output (CRITICAL DIRECTORY DO NOT TOUCH)/xgboost_model_normalized.pkl
  - MLP: training_output (CRITICAL DIRECTORY DO NOT TOUCH)/mlp_baseline_model.pt
  - Ensemble Weights: See Ensemble Evaluation Summary/ensemble_detailed_results.json
"""
    else:
        baseline_lock += f"""  - Model: See individual model documentation
"""
    
    baseline_lock += f"""
Evaluation Results:
  - Comparison Table: Ensemble Evaluation Summary/ensemble_comparison_table.csv
  - Detailed Metrics: Ensemble Evaluation Summary/ensemble_detailed_results.json
  - Visualizations: Ensemble Evaluation Summary/*.png

================================================================================
6. USAGE INSTRUCTIONS
================================================================================

To use the baseline model:

1. Load normalized models (RF, XGBoost, MLP)
2. Get predictions from each model
3. Apply ensemble method:
"""
    
    if 'Weighted Average' in best_method['Method']:
        baseline_lock += """   - Weighted Average: 
     prediction = 0.3329 * RF_pred + 0.3328 * XGB_pred + 0.3343 * MLP_pred
"""
    elif 'Simple Average' in best_method['Method']:
        baseline_lock += """   - Simple Average:
     prediction = (RF_pred + XGB_pred + MLP_pred) / 3
"""
    
    baseline_lock += f"""
4. Use prediction as final ΔΔG estimate

================================================================================
7. BASELINE STATUS
================================================================================

Status: LOCKED
Date Locked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: 1.0

This baseline is now frozen. Any future improvements will be documented as
new versions or experimental improvements, not as baseline changes.

================================================================================
8. NOTES
================================================================================

- All models trained on normalized data for fair comparison
- Ensemble provides 0.24% improvement over best individual model
- Baseline is ready for production use
- Future work: MLP hyperparameter tuning (if allowed)

================================================================================
END OF BASELINE LOCK DOCUMENT
================================================================================
"""
    
    # Save baseline lock
    output_dir = Path("Baseline Lock")
    output_dir.mkdir(exist_ok=True)
    
    lock_path = output_dir / "BASELINE_LOCK.txt"
    with open(lock_path, 'w', encoding='utf-8') as f:
        f.write(baseline_lock)
    
    print(f"\n[SUCCESS] Baseline locked!")
    print(f"Best Method: {best_method['Method']}")
    print(f"Test MAE: {best_method['MAE (kcal/mol)']} kcal/mol")
    print(f"\nBaseline lock document saved to: {lock_path}")
    
    # Also save as JSON for programmatic access
    baseline_json = {
        'baseline_method': best_method['Method'],
        'test_mae': float(best_method['MAE (kcal/mol)']),
        'test_rmse': float(best_method['RMSE (kcal/mol)']),
        'test_r2': float(best_method['R²']),
        'locked_date': datetime.now().isoformat(),
        'version': '1.0',
        'status': 'LOCKED'
    }
    
    json_path = output_dir / "baseline_lock.json"
    with open(json_path, 'w') as f:
        json.dump(baseline_json, f, indent=2)
    
    print(f"Baseline lock JSON saved to: {json_path}")
    
    return baseline_lock, baseline_json


if __name__ == "__main__":
    create_baseline_lock_document()

