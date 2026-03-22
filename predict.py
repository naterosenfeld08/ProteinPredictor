"""
Unified CLI for protein property prediction.

Two usage modes:
  1. FASTA mode for batch prediction from FASTA files.
  2. Sequence mode for curated sequences that need per-sequence folders with
     comparison tables, plots, and summaries.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from protein_baseline import (
    predict_from_fasta,
    predict_single_sequence_with_outputs,
)


def parse_true_values(values_str: str) -> Dict[str, float]:
    """Parse strings like 'trait1=1.2, trait2:3.4' into a dict."""
    true_values: Dict[str, float] = {}
    for pair in values_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" in pair:
            trait, value = pair.split("=", 1)
        elif ":" in pair:
            trait, value = pair.split(":", 1)
        else:
            continue
        try:
            true_values[trait.strip()] = float(value.strip())
        except ValueError:
            continue
    return true_values


def parse_sequences_file(file_path: str) -> List[Dict]:
    """Parse a FASTA-like text file with optional true values comments."""
    sequences: List[Dict] = []
    current_name: Optional[str] = None
    current_seq: List[str] = []
    current_true_values: Optional[Dict[str, float]] = None

    with open(file_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if "True values" in line:
                    _, values_part = line.split("True values:", 1)
                    current_true_values = parse_true_values(values_part.strip())
                continue

            if line.startswith(">"):
                if current_name and current_seq:
                    sequences.append(
                        {
                            "name": current_name,
                            "sequence": "".join(current_seq),
                            "true_values": current_true_values,
                        }
                    )
                current_name = line[1:].strip()
                current_seq = []
                current_true_values = None
                continue

            if ":" in line and not current_name:
                maybe_name, maybe_seq = line.split(":", 1)
                sequences.append(
                    {
                        "name": maybe_name.strip(),
                        "sequence": maybe_seq.strip(),
                        "true_values": None,
                    }
                )
                continue

            current_seq.append(line)

    if current_name and current_seq:
        sequences.append(
            {
                "name": current_name,
                "sequence": "".join(current_seq),
                "true_values": current_true_values,
            }
        )

    return sequences


def handle_fasta(args: argparse.Namespace) -> None:
    """Run FASTA mode predictions."""
    fasta_path = Path(args.fasta_file)
    if not fasta_path.exists():
        print(f"ERROR: FASTA file not found: {fasta_path}")
        sys.exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("\nTrain a model first with:")
        print("  python protein_baseline.py --csv_path your_data.csv")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.json"

    print("=" * 60)
    print("FASTA MODE PREDICTION")
    print("=" * 60)
    print(f"\nFASTA file : {fasta_path}")
    print(f"Model      : {model_path}")
    print(f"Output     : {output_path}")
    print(f"Embeddings : {args.model_type}")
    print()

    try:
        results = predict_from_fasta(
            fasta_path=str(fasta_path),
            model_path=str(model_path),
            embedding_model_type=args.model_type,
            output_path=str(output_path),
            min_length=args.min_length,
            max_length=args.max_length,
            use_composition_features=not args.no_composition_features,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)

    n_valid = results.get("n_valid", 0)
    n_invalid = results.get("n_invalid", 0)

    print(f"\nValid sequences  : {n_valid}")
    print(f"Invalid sequences: {n_invalid}")

    if n_valid > 0 and "predictions" in results:
        print(f"\nPredictions saved to {output_path}")
        preview = results["predictions"][: min(5, len(results["predictions"]))]
        for i, pred in enumerate(preview, 1):
            header = pred.get("header", f"Sequence {i}")
            print(f"\n  {header}:")
            print(f"    Predicted value: {pred['pred_value']:.6f}")
            if "uncertainty" in pred and pred["uncertainty"] is not None:
                unc = pred["uncertainty"]
                lower = pred["pred_value"] - 1.96 * unc
                upper = pred["pred_value"] + 1.96 * unc
                print(f"    Uncertainty    : {unc:.6f}")
                print(f"    95% interval   : [{lower:.6f}, {upper:.6f}]")

        remaining = len(results["predictions"]) - len(preview)
        if remaining > 0:
            print(f"\n  ... plus {remaining} additional predictions")


def handle_sequences(args: argparse.Namespace) -> None:
    """Run curated sequence mode predictions."""
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    sequences: List[Dict]
    if args.sequences_file:
        seq_file = Path(args.sequences_file)
        if not seq_file.exists():
            print(f"ERROR: Sequences file not found: {seq_file}")
            sys.exit(1)
        sequences = parse_sequences_file(str(seq_file))
    elif args.sequence:
        if not args.name:
            print("ERROR: --name is required when providing --sequence")
            sys.exit(1)
        true_vals = parse_true_values(args.true_values) if args.true_values else None
        sequences = [
            {
                "name": args.name,
                "sequence": args.sequence,
                "true_values": true_vals,
            }
        ]
    else:
        print("ERROR: Provide either --sequences_file or --sequence/--name")
        sys.exit(1)

    if not sequences:
        print("ERROR: No sequences to process")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("SEQUENCE MODE PREDICTION")
    print("=" * 60)
    print(f"Sequences to process: {len(sequences)}")
    print(f"Output directory    : {output_dir}")
    print()

    all_results: List[Dict] = []
    for idx, seq in enumerate(sequences, 1):
        print(f"[{idx}/{len(sequences)}] {seq['name']}")
        try:
            result = predict_single_sequence_with_outputs(
                sequence=seq["sequence"],
                sequence_name=seq["name"],
                model_path=str(model_path),
                true_values=seq.get("true_values"),
                embedding_model_type=args.model_type,
                output_dir=str(output_dir),
                trait_name=args.trait_name,
                use_composition_features=not args.no_composition_features,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback

            traceback.print_exc()
            continue

        all_results.append(result)
        print("  Output files:")
        if "output_files" in result:
            outputs = result["output_files"]
            for label, path in outputs.items():
                print(f"    {label.title():<10}: {path}")
        if "metrics" in result and result["metrics"]:
            metrics = result["metrics"]
            mae = metrics.get("mae")
            rmse = metrics.get("rmse")
            r2 = metrics.get("r2")
            metric_line = f"    Metrics  : MAE={mae:.4f} RMSE={rmse:.4f}"
            if r2 is not None:
                metric_line += f" R²={r2:.4f}"
            print(metric_line)

    summary_path = output_dir / "sequence_mode_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"\nSaved consolidated summary to {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FASTA mode
  python predict.py fasta my_sequences.fasta --model_path ./models/random_forest.pkl

  # Sequence mode from a curated file
  python predict.py sequences --sequences_file curated.txt --model_path ./models/random_forest.pkl --trait_name ddG

  # Sequence mode for a single sequence
  python predict.py sequences --sequence MKT... --name Protein1 --true_values "ddG=-0.9"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    fasta_parser = subparsers.add_parser(
        "fasta", help="Predict from a FASTA file (batch mode)"
    )
    fasta_parser.add_argument("fasta_file", type=str, help="Path to FASTA file")
    fasta_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model pickle file",
    )
    fasta_parser.add_argument(
        "--output_dir",
        type=str,
        default="./predictions",
        help="Directory to store predictions.json (default: ./predictions)",
    )
    fasta_parser.add_argument(
        "--model_type",
        type=str,
        default="both",
        choices=["prot_t5", "esm2", "both"],
        help="Embedding backbone to use (default: both)",
    )
    fasta_parser.add_argument(
        "--min_length", type=int, default=10, help="Minimum sequence length"
    )
    fasta_parser.add_argument(
        "--max_length", type=int, default=5000, help="Maximum sequence length"
    )
    fasta_parser.add_argument(
        "--no_composition_features",
        action="store_true",
        help="Disable amino acid composition feature augmentation",
    )
    fasta_parser.set_defaults(func=handle_fasta)

    seq_parser = subparsers.add_parser(
        "sequences",
        help="Predict hand-curated sequences (per-sequence folders and plots)",
    )
    seq_parser.add_argument(
        "--sequences_file",
        type=str,
        help="Path to text file with sequences (FASTA-like, optional true values)",
    )
    seq_parser.add_argument("--sequence", type=str, help="Single sequence string")
    seq_parser.add_argument("--name", type=str, help="Name for --sequence mode")
    seq_parser.add_argument(
        "--true_values",
        type=str,
        help='True values string, e.g. "ddG=-0.9,stability:1.2"',
    )
    seq_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model pickle file",
    )
    seq_parser.add_argument(
        "--output_dir",
        type=str,
        default="./sequence_predictions",
        help="Base directory for per-sequence folders",
    )
    seq_parser.add_argument(
        "--model_type",
        type=str,
        default="both",
        choices=["prot_t5", "esm2", "both"],
        help="Embedding backbone to use (default: both)",
    )
    seq_parser.add_argument(
        "--trait_name",
        type=str,
        default="Property",
        help="Name of the predicted trait (used in summaries)",
    )
    seq_parser.add_argument(
        "--no_composition_features",
        action="store_true",
        help="Disable amino acid composition feature augmentation",
    )
    seq_parser.set_defaults(func=handle_sequences)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

