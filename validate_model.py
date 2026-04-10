"""
Model Validation Script
Validates trained model on unseen FireProtDB sequences.

This script:
1. Loads the trained Random Forest model
2. Identifies validation sequences from FireProtDB (not in training set)
3. Makes predictions on validation sequences
4. Compares predictions to experimental DDG values
5. Generates comprehensive validation metrics and visualizations
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

from core.protein_baseline import (
    EmbeddingExtractor,
    add_composition_features,
    validate_sequence,
    predict_with_uncertainty
)
from core.fireprot_data_loader import FireProtDBLoader, get_all_training_indices

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class ModelValidator:
    """
    Validates trained model on unseen data.
    """
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./validation_output",
        splits_path: Optional[str] = None,
        use_composition_features: bool = True,
    ):
        """
        Initialize model validator.
        
        Args:
            model_path: Path to trained model pickle file
            output_dir: Directory to save validation results
            splits_path: Path to data_splits.npz (if None, searches training_output)
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_composition_features = use_composition_features
        
        # Find splits file
        if splits_path is None:
            possible_paths = [
                Path("training_output/data_splits.npz"),
                Path("training_output") / "data_splits.npz",
                Path("training_output (CRITICAL DIRECTORY DO NOT TOUCH)/data_splits.npz")
            ]
            for path in possible_paths:
                if path.exists():
                    splits_path = str(path)
                    break
            else:
                raise FileNotFoundError("Could not find data_splits.npz file")
        
        self.splits_path = splits_path
        
        # Setup logging
        log_file = self.output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.logger.info(f"Loading model from {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.logger.info("Model loaded successfully")
        
        # Initialize embedding extractor
        self.embedding_extractor = EmbeddingExtractor()
        
    def load_validation_data(
        self,
        fireprot_csv: str,
        n_samples: Optional[int] = None,
        min_ddg_samples: int = 100
    ) -> pd.DataFrame:
        """
        Load validation sequences from FireProtDB.
        
        Args:
            fireprot_csv: Path to fireprotdb_with_sequences.csv
            n_samples: Maximum number of validation samples (None for all available)
            min_ddg_samples: Minimum number of samples required
            
        Returns:
            DataFrame with validation sequences and DDG values
        """
        self.logger.info("Loading validation data from FireProtDB...")
        
        # Get training indices to exclude
        training_indices = get_all_training_indices(self.splits_path)
        self.logger.info(f"Excluding {len(training_indices)} training indices")
        
        # Load validation set
        loader = FireProtDBLoader(fireprot_csv)
        validation_df = loader.get_validation_set(
            exclude_indices=training_indices,
            min_ddg_samples=min_ddg_samples,
            max_samples=n_samples
        )
        
        self.logger.info(f"Loaded {len(validation_df)} validation sequences")
        self.logger.info(f"DDG range: {validation_df['DDG'].min():.2f} to {validation_df['DDG'].max():.2f} kcal/mol")
        self.logger.info(f"DDG mean: {validation_df['DDG'].mean():.2f} kcal/mol")
        self.logger.info(f"DDG std: {validation_df['DDG'].std():.2f} kcal/mol")
        
        return validation_df
    
    def extract_embeddings(self, sequences: pd.Series) -> np.ndarray:
        """
        Extract embeddings for sequences.
        
        Args:
            sequences: Series of protein sequences
            
        Returns:
            Array of embeddings (n_samples, n_features)
        """
        self.logger.info(f"Extracting embeddings for {len(sequences)} sequences...")
        
        embeddings_dict, diagnostics = self.embedding_extractor.extract_embeddings_batch(
            sequences,
            model_type="both",
            mean_pool=True,
            cache=True,
            validate=True,
            logger=self.logger
        )
        
        # Combine embeddings
        if 'prot_t5' in embeddings_dict and 'esm2' in embeddings_dict:
            embeddings = np.concatenate([embeddings_dict['prot_t5'], embeddings_dict['esm2']], axis=1)
        elif 'prot_t5' in embeddings_dict:
            embeddings = embeddings_dict['prot_t5']
        elif 'esm2' in embeddings_dict:
            embeddings = embeddings_dict['esm2']
        else:
            raise ValueError("No embeddings extracted")
        
        # Optionally add composition features to match training-time features.
        if self.use_composition_features:
            embeddings_dict_combined = {'combined': embeddings}
            enhanced = add_composition_features(embeddings_dict_combined, sequences)
            final_embeddings = enhanced['combined']
        else:
            final_embeddings = embeddings
        
        self.logger.info(f"Final embeddings shape: {final_embeddings.shape}")
        
        return final_embeddings
    
    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions using trained model.
        
        Args:
            embeddings: Embedding array (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        self.logger.info("Making predictions...")
        
        predictions = self.model.predict(embeddings)
        
        # Try to get uncertainties if Random Forest
        uncertainties = None
        if hasattr(self.model, 'estimators_'):
            # Random Forest - compute std across trees
            tree_predictions = np.array([tree.predict(embeddings) for tree in self.model.estimators_])
            uncertainties = np.std(tree_predictions, axis=0)
        
        return predictions, uncertainties
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute validation metrics.
        
        Args:
            y_true: True DDG values
            y_pred: Predicted DDG values
            
        Returns:
            Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'n_samples': len(y_true)
        }
    
    def create_visualizations(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Create validation visualizations.
        
        Args:
            y_true: True DDG values
            y_pred: Predicted DDG values
            uncertainties: Prediction uncertainties (optional)
            metrics: Validation metrics (optional)
        """
        self.logger.info("Creating validation visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. True vs Predicted scatter
        ax = axes[0, 0]
        if uncertainties is not None:
            ax.errorbar(y_true, y_pred, yerr=uncertainties, alpha=0.5, fmt='o', 
                       markersize=3, capsize=2, label='Predictions with uncertainty')
        else:
            ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
        
        # Diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        if metrics:
            ax.text(0.05, 0.95, 
                   f"R² = {metrics['r2']:.3f}\nPearson r = {metrics['pearson_r']:.3f}\nMAE = {metrics['mae']:.3f} kcal/mol",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=11, fontweight='bold')
        
        ax.set_xlabel('True DDG (kcal/mol)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted DDG (kcal/mol)', fontsize=12, fontweight='bold')
        ax.set_title('True vs Predicted DDG', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # 2. Residuals plot
        ax = axes[0, 1]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted DDG (kcal/mol)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residuals (True - Predicted)', fontsize=12, fontweight='bold')
        ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 3. Residual histogram
        ax = axes[1, 0]
        ax.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Residuals (kcal/mol)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Error distribution
        ax = axes[1, 1]
        absolute_errors = np.abs(residuals)
        ax.hist(absolute_errors, bins=50, color='coral', alpha=0.7, edgecolor='black', linewidth=1)
        if metrics:
            ax.axvline(x=metrics['mae'], color='r', linestyle='--', linewidth=2, 
                      label=f"MAE = {metrics['mae']:.3f} kcal/mol")
        ax.set_xlabel('Absolute Error (kcal/mol)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plot_path = self.output_dir / "validation_plots.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved validation plots to {plot_path}")
    
    def run_validation(
        self,
        fireprot_csv: str,
        n_samples: Optional[int] = None,
        min_ddg_samples: int = 100
    ) -> Dict:
        """
        Run complete validation pipeline.
        
        Args:
            fireprot_csv: Path to fireprotdb_with_sequences.csv
            n_samples: Maximum number of validation samples
            min_ddg_samples: Minimum number of samples required
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("=" * 80)
        self.logger.info("MODEL VALIDATION")
        self.logger.info("=" * 80)
        
        # Load validation data
        validation_df = self.load_validation_data(
            fireprot_csv,
            n_samples=n_samples,
            min_ddg_samples=min_ddg_samples
        )
        
        if len(validation_df) == 0:
            raise ValueError("No validation data available")
        
        # Extract embeddings
        sequences = validation_df['sequence']
        embeddings = self.extract_embeddings(sequences)
        
        # Make predictions
        predictions, uncertainties = self.predict(embeddings)
        
        # Get true values
        y_true = validation_df['DDG'].values
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, predictions)
        
        # Create visualizations
        self.create_visualizations(y_true, predictions, uncertainties, metrics)
        
        # Save results
        results = {
            'validation_date': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'n_validation_samples': len(validation_df),
            'metrics': metrics,
            'predictions': {
                'true_values': y_true.tolist(),
                'predicted_values': predictions.tolist(),
                'uncertainties': uncertainties.tolist() if uncertainties is not None else None
            }
        }
        
        results_path = self.output_dir / "validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save predictions CSV
        predictions_df = pd.DataFrame({
            'sequence': sequences.values,
            'true_ddg': y_true,
            'predicted_ddg': predictions,
            'error': y_true - predictions,
            'absolute_error': np.abs(y_true - predictions)
        })
        if uncertainties is not None:
            predictions_df['uncertainty'] = uncertainties
        
        csv_path = self.output_dir / "validation_predictions.csv"
        predictions_df.to_csv(csv_path, index=False)
        
        # Print summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("VALIDATION RESULTS")
        self.logger.info("=" * 80)
        self.logger.info(f"Validation samples: {metrics['n_samples']}")
        self.logger.info(f"MAE: {metrics['mae']:.4f} kcal/mol")
        self.logger.info(f"RMSE: {metrics['rmse']:.4f} kcal/mol")
        self.logger.info(f"R²: {metrics['r2']:.4f}")
        self.logger.info(f"Pearson r: {metrics['pearson_r']:.4f} (p = {metrics['pearson_p']:.2e})")
        self.logger.info(f"Spearman r: {metrics['spearman_r']:.4f} (p = {metrics['spearman_p']:.2e})")
        self.logger.info("=" * 80)
        
        self.logger.info(f"\nResults saved to: {self.output_dir}")
        self.logger.info(f"  - Validation plots: {self.output_dir / 'validation_plots.png'}")
        self.logger.info(f"  - Results JSON: {results_path}")
        self.logger.info(f"  - Predictions CSV: {csv_path}")
        
        return results


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate trained model on unseen FireProtDB data")
    parser.add_argument(
        '--model_path',
        type=str,
        default='training_output (CRITICAL DIRECTORY DO NOT TOUCH)/model_random_forest.pkl',
        help='Path to trained model pickle file'
    )
    parser.add_argument(
        '--fireprot_csv',
        type=str,
        default='fireprotdb_with_sequences.csv',
        help='Path to fireprotdb_with_sequences.csv'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./validation_output',
        help='Output directory for validation results'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Maximum number of validation samples (None for all available)'
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=100,
        help='Minimum number of validation samples required'
    )
    parser.add_argument(
        '--no_composition_features',
        action='store_true',
        help='Disable amino acid composition feature augmentation'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = ModelValidator(
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_composition_features=not args.no_composition_features,
    )
    
    results = validator.run_validation(
        fireprot_csv=args.fireprot_csv,
        n_samples=args.n_samples,
        min_ddg_samples=args.min_samples
    )
    
    print("\nValidation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

