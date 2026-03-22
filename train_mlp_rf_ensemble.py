"""
Train an efficient ensemble of:
  - RandomForestRegressor (best hyperparams chosen on validation)
  - BaselineMLP (with early stopping on validation)
and combine them via weighted averaging on validation.

Because this repo may not have precomputed embedding artifacts, this script
recomputes embeddings on demand (with caching via EmbeddingExtractor).

Overfitting controls:
  - RF: hyperparameter selection on validation (limits capacity)
  - MLP: early stopping on validation MAE + dropout/weight decay
  - Post-hoc: train/val gap and val/test gap warnings
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from fireprot_data_loader import FireProtDBLoader
from mlp_baseline import BaselineMLP, DDGPredictionDataset, evaluate_model, train_mlp
from mlp_rf_ensemble import MLPEngineConfig, MLPRandomForestEnsemble
from protein_baseline import EmbeddingExtractor, add_composition_features, validate_sequence


def make_stratified_splits(
    df: pd.DataFrame,
    target_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified splits by binning continuous DDG into quantile bins.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    y = df[target_col].values

    # Stratification bins
    try:
        bins = pd.qcut(y, q=n_bins, duplicates="drop", labels=False)
        if bins.isna().all():
            raise ValueError("All bins are NaN")
        bins = bins.astype(int)
    except Exception:
        # Fallback: no stratification
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        return train_idx, val_idx, test_idx

    # Two-stage split to get val/test sizes exactly.
    from sklearn.model_selection import train_test_split

    test_val_frac = val_frac + test_frac
    # Split train vs (val+test)
    train_idx, temp_idx = train_test_split(
        idx, test_size=test_val_frac, random_state=seed, stratify=bins
    )

    # Split temp into val and test
    temp_bins = bins[temp_idx]
    val_size_in_temp = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_size_in_temp),
        random_state=seed,
        stratify=temp_bins,
    )

    return train_idx, val_idx, test_idx


def compute_features_for_split(
    sequences: pd.Series,
    extractor: EmbeddingExtractor,
    embedding_model_type: str,
    mean_pool: bool,
    cache: bool,
    use_composition_features: bool,
    validate_seqs: bool,
) -> Tuple[np.ndarray, Dict]:
    embeddings_dict, diagnostics = extractor.extract_embeddings_batch(
        sequences,
        model_type=embedding_model_type,
        mean_pool=mean_pool,
        cache=cache,
        validate=validate_seqs,
        logger=None,
    )

    if use_composition_features:
        embeddings_dict = add_composition_features(embeddings_dict, sequences)

    # Build X based on what embedding backbones were returned
    if "prot_t5" in embeddings_dict and "esm2" in embeddings_dict:
        X = np.concatenate([embeddings_dict["prot_t5"], embeddings_dict["esm2"]], axis=1)
    elif "prot_t5" in embeddings_dict:
        X = embeddings_dict["prot_t5"]
    elif "esm2" in embeddings_dict:
        X = embeddings_dict["esm2"]
    else:
        raise ValueError("No embeddings found (unexpected embeddings_dict keys).")

    return X, diagnostics


def grid_search_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    n_jobs: int,
) -> Tuple[RandomForestRegressor, Dict]:
    """
    Pick RF hyperparameters that minimize validation MAE.
    """
    grid = []
    # Keep this small for efficiency while still limiting overfitting.
    for n_estimators in [100, 200]:
        for max_depth in [None, 12, 6]:
            for min_samples_split in [2, 5, 10]:
                for min_samples_leaf in [1, 2, 4]:
                    grid.append(
                        {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                        }
                    )

    best = None
    best_mae = float("inf")
    best_info: Dict = {}

    for params in grid:
        rf = RandomForestRegressor(
            random_state=seed,
            n_jobs=n_jobs,
            **params,
        )
        rf.fit(X_train, y_train)
        pred_val = rf.predict(X_val)
        mae = mean_absolute_error(y_val, pred_val)
        if mae < best_mae:
            best_mae = mae
            best = rf
            best_info = {"params": params, "val_mae": float(mae)}

    assert best is not None
    return best, best_info


def predict_mlp_from_trained_model(
    mlp_model: BaselineMLP,
    X_norm: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    mlp_model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X_norm.shape[0], batch_size):
            batch = torch.from_numpy(X_norm[start : start + batch_size]).to(device)
            out = mlp_model(batch).cpu().numpy().reshape(-1)
            preds.append(out)
    return np.concatenate(preds, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Train MLP + RF ensemble for ΔΔG prediction")

    parser.add_argument("--fireprot_csv", type=str, default=None, help="Path to FireProtDB CSV export (without sequences).")
    parser.add_argument(
        "--with_sequences_csv",
        type=str,
        default="fireprotdb_with_sequences.csv",
        help="Path to FireProtDB CSV that already includes a `sequence` column.",
    )
    parser.add_argument(
        "--fetch_sequences",
        action="store_true",
        help="If `with_sequences_csv` is missing, fetch sequences using UNIPROTKB and save to `with_sequences_csv`.",
    )
    parser.add_argument("--output_dir", type=str, default="training_output (CRITICAL DIRECTORY DO NOT TOUCH)/mlp_rf_ensemble")

    parser.add_argument("--target_col", type=str, default="DDG", help="Target column in CSV")
    parser.add_argument("--sequence_col", type=str, default="sequence", help="Sequence column in CSV")
    parser.add_argument("--embedding_model_type", type=str, default="both", choices=["prot_t5", "esm2", "both"])
    parser.add_argument("--use_composition_features", action="store_true", default=True)
    parser.add_argument("--no_composition_features", action="store_true", help="Disable composition features augmentation")

    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=5000)
    parser.add_argument("--max_rows", type=int, default=None, help="Optional limit for quick experiments")

    # Split fractions
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify_bins", type=int, default=20)

    # MLP hyperparams
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--mlp_lr", type=float, default=1e-4)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-3)
    parser.add_argument("--mlp_batch_size", type=int, default=128)
    parser.add_argument("--mlp_max_epochs", type=int, default=200)
    parser.add_argument("--mlp_early_stopping_patience", type=int, default=10)

    # Efficiency controls
    parser.add_argument("--rf_n_jobs", type=int, default=-1)
    parser.add_argument("--ensemble_weight_grid_step", type=float, default=0.01)

    # Overfitting thresholds (for warnings)
    parser.add_argument("--overfit_gap_warn_kcal", type=float, default=0.25)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_comp = args.use_composition_features and not args.no_composition_features

    # 1) Ensure dataset exists
    with_seq_path = Path(args.with_sequences_csv)
    if not with_seq_path.exists():
        if not args.fetch_sequences:
            raise FileNotFoundError(
                f"Missing {with_seq_path}. Provide it, or pass --fetch_sequences (and --fireprot_csv). "
                f"FireProtDB download page: https://loschmidt.chemi.muni.cz/fireprotdb/download/"
            )
        if args.fireprot_csv is None:
            raise ValueError("--fetch_sequences requires --fireprot_csv to point at the FireProtDB CSV export.")
        from uniprot_fetcher import fetch_sequences_for_fireprot

        print(f"Fetching UniProt sequences into {with_seq_path} ...")
        fetch_sequences_for_fireprot(
            csv_path=args.fireprot_csv,
            output_csv_path=str(with_seq_path),
            max_rows=args.max_rows,
        )

    # 2) Load dataset (sequence + target) with filtering
    loader = FireProtDBLoader(str(with_seq_path))
    filters = {
        args.sequence_col: lambda x: x.notna()
        & (x.astype(str).str.len() >= args.min_length)
        & (x.astype(str).str.len() <= args.max_length),
        args.target_col: lambda x: x.notna(),
    }
    df = loader.load_with_filters(
        required_columns=[args.sequence_col, args.target_col],
        filters=filters,
        max_rows=args.max_rows,
        exclude_indices=None,
    )
    if df.empty:
        raise ValueError("No data left after filtering.")

    # Extra amino-acid validation (defensive)
    valid_mask = []
    sequences = df[args.sequence_col].astype(str).values
    for s in sequences:
        ok, _ = validate_sequence(s, min_length=args.min_length, max_length=args.max_length)
        valid_mask.append(ok)
    df = df.loc[valid_mask].reset_index(drop=True)

    # 3) Stratified splits + save indices
    train_idx, val_idx, test_idx = make_stratified_splits(
        df,
        target_col=args.target_col,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        n_bins=args.stratify_bins,
    )
    np.savez(
        output_dir / "data_splits.npz",
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # 4) Feature extraction (embeddings + composition)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = EmbeddingExtractor()

    # For reproducibility/debugging, let the cache exist; embeddings are deterministic for a given model.
    # We validate sequences upstream, so this skips extra validation work.
    X_train, diag_train = compute_features_for_split(
        train_df[args.sequence_col],
        extractor=extractor,
        embedding_model_type=args.embedding_model_type,
        mean_pool=True,
        cache=True,
        use_composition_features=use_comp,
        validate_seqs=False,
    )
    X_val, diag_val = compute_features_for_split(
        val_df[args.sequence_col],
        extractor=extractor,
        embedding_model_type=args.embedding_model_type,
        mean_pool=True,
        cache=True,
        use_composition_features=use_comp,
        validate_seqs=False,
    )
    X_test, diag_test = compute_features_for_split(
        test_df[args.sequence_col],
        extractor=extractor,
        embedding_model_type=args.embedding_model_type,
        mean_pool=True,
        cache=True,
        use_composition_features=use_comp,
        validate_seqs=False,
    )

    y_train = train_df[args.target_col].astype(float).values
    y_val = val_df[args.target_col].astype(float).values
    y_test = test_df[args.target_col].astype(float).values

    # 5) Train RF (validation-tuned)
    rf_model, rf_info = grid_search_rf(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seed=args.seed,
        n_jobs=args.rf_n_jobs,
    )

    rf_train_pred = rf_model.predict(X_train)
    rf_val_pred = rf_model.predict(X_val)
    rf_test_pred = rf_model.predict(X_test)

    rf_train_mae = float(mean_absolute_error(y_train, rf_train_pred))
    rf_val_mae = float(mean_absolute_error(y_val, rf_val_pred))
    rf_test_mae = float(mean_absolute_error(y_test, rf_test_pred))

    # 6) Train MLP with early stopping
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0) + 1e-8

    X_train_norm = (X_train - feature_mean) / feature_std
    X_val_norm = (X_val - feature_mean) / feature_std
    X_test_norm = (X_test - feature_mean) / feature_std

    mlp = BaselineMLP(input_dim=X_train.shape[1], dropout=args.mlp_dropout)

    train_dataset = DDGPredictionDataset(X_train_norm, y_train)
    val_dataset = DDGPredictionDataset(X_val_norm, y_val)
    test_dataset = DDGPredictionDataset(X_test_norm, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.mlp_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.mlp_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.mlp_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    history = train_mlp(
        model=mlp,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.mlp_max_epochs,
        learning_rate=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        early_stopping_patience=args.mlp_early_stopping_patience,
        verbose=True,
    )

    val_metrics, mlp_val_pred, _ = evaluate_model(mlp, val_loader, device)
    test_metrics, mlp_test_pred, _ = evaluate_model(mlp, test_loader, device)

    # Also compute train MAE for overfitting signal
    train_loader_for_eval = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.mlp_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    _, mlp_train_pred, _ = evaluate_model(mlp, train_loader_for_eval, device)
    mlp_train_mae = float(mean_absolute_error(y_train, mlp_train_pred))

    mlp_val_mae = float(val_metrics["MAE"])
    mlp_test_mae = float(test_metrics["MAE"])

    # 7) Pick ensemble weight on validation
    weights = np.arange(0.0, 1.0 + 1e-9, args.ensemble_weight_grid_step)
    best_w = None
    best_ens_mae = float("inf")
    for w in weights:
        ens_pred_val = w * rf_val_pred + (1.0 - w) * mlp_val_pred
        mae = mean_absolute_error(y_val, ens_pred_val)
        if mae < best_ens_mae:
            best_ens_mae = float(mae)
            best_w = float(w)

    assert best_w is not None

    # 8) Evaluate ensemble on test
    ensemble = MLPRandomForestEnsemble(
        rf_model=rf_model,
        mlp_state_dict=mlp.state_dict(),
        mlp_config=MLPEngineConfig(input_dim=int(X_train.shape[1]), dropout=float(args.mlp_dropout)),
        feature_mean=feature_mean,
        feature_std=feature_std,
        weight_rf=best_w,
        device=str(device),
    )

    ens_pred_test, ens_std_test = ensemble.predict_with_uncertainty(X_test)
    ens_test_mae = float(mean_absolute_error(y_test, ens_pred_test))

    # 9) Overfitting checks (simple gap metrics)
    rf_gap_train_val = rf_val_mae - rf_train_mae
    mlp_gap_train_val = mlp_val_mae - mlp_train_mae
    rf_gap_val_test = rf_test_mae - rf_val_mae
    mlp_gap_val_test = mlp_test_mae - mlp_val_mae

    def _gap_flag(gap: float) -> str:
        if gap >= args.overfit_gap_warn_kcal:
            return "warn"
        return "ok"

    metadata = {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "feature_dim": int(X_train.shape[1]),
        "use_composition_features": bool(use_comp),
        "embedding_model_type": args.embedding_model_type,
        "seed": args.seed,
        "rf_best": rf_info,
        "mlp": {
            "dropout": args.mlp_dropout,
            "learning_rate": args.mlp_lr,
            "weight_decay": args.mlp_weight_decay,
            "batch_size": args.mlp_batch_size,
            "max_epochs": args.mlp_max_epochs,
            "early_stopping_patience": args.mlp_early_stopping_patience,
            "history": history,
        },
        "ensemble": {
            "weight_rf": best_w,
            "val_mae": best_ens_mae,
            "test_mae": ens_test_mae,
        },
        "overfitting_gaps_kcalmol": {
            "rf_train_val_gap": rf_gap_train_val,
            "rf_val_test_gap": rf_gap_val_test,
            "rf_train_val_flag": _gap_flag(rf_gap_train_val),
            "rf_val_test_flag": _gap_flag(rf_gap_val_test),
            "mlp_train_val_gap": mlp_gap_train_val,
            "mlp_val_test_gap": mlp_gap_val_test,
            "mlp_train_val_flag": _gap_flag(mlp_gap_train_val),
            "mlp_val_test_flag": _gap_flag(mlp_gap_val_test),
        },
        "metrics": {
            "rf": {"train_mae": rf_train_mae, "val_mae": rf_val_mae, "test_mae": rf_test_mae},
            "mlp": {"train_mae": mlp_train_mae, "val_mae": mlp_val_mae, "test_mae": mlp_test_mae},
        },
        "embedding_diagnostics": {
            "train": diag_train,
            "val": diag_val,
            "test": diag_test,
        },
    }

    # 10) Save models
    import pickle

    with open(output_dir / "rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    # Save ensemble as a pickle object with deterministic predict().
    with open(output_dir / "mlp_rf_ensemble.pkl", "wb") as f:
        pickle.dump(ensemble, f)

    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\nDone.")
    print(f"Best RF params: {rf_info['params']}")
    print(f"Best ensemble weight_rf={best_w:.3f}")
    print(f"Test MAE: RF={rf_test_mae:.4f}, MLP={mlp_test_mae:.4f}, Ensemble={ens_test_mae:.4f}")


if __name__ == "__main__":
    main()

