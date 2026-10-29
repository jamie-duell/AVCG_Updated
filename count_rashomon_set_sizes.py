"""Count MC-dropout Rashomon-set membership by dataset and epsilon.

This script rebuilds only the parts needed for Rashomon-set sizes:
dataset loading, base MC-dropout training, frozen-mask ensemble sampling, and
validation-loss filtering. It does not train counterfactual generators.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_DATASETS = ("breast_cancer", "heart_disease", "spambase", "adult_income")
DEFAULT_EPSILONS = (0.0, 0.01, 0.05, 0.1, 0.5)
DEFAULT_SEEDS = (11, 22, 33, 44, 55)

np = None
torch = None
DataLoader = None
DEVICE = None
MCDropoutMLP = None
MCDropoutRashomonSet = None
get_dataset = None
set_seed = None
train_mc_dropout = None


def _load_experiment_dependencies() -> None:
    global np
    global torch
    global DataLoader
    global DEVICE
    global MCDropoutMLP
    global MCDropoutRashomonSet
    global get_dataset
    global set_seed
    global train_mc_dropout

    if np is not None:
        return

    import numpy as _np
    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader

    from main import (
        DEVICE as _DEVICE,
        MCDropoutMLP as _MCDropoutMLP,
        MCDropoutRashomonSet as _MCDropoutRashomonSet,
        get_dataset as _get_dataset,
        set_seed as _set_seed,
        train_mc_dropout as _train_mc_dropout,
    )

    np = _np
    torch = _torch
    DataLoader = _DataLoader
    DEVICE = _DEVICE
    MCDropoutMLP = _MCDropoutMLP
    MCDropoutRashomonSet = _MCDropoutRashomonSet
    get_dataset = _get_dataset
    set_seed = _set_seed
    train_mc_dropout = _train_mc_dropout


def _parse_csv_values(raw: str, cast):
    return tuple(cast(item.strip()) for item in raw.split(",") if item.strip())


def _sample_std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def count_for_dataset_seed(
    dataset_name: str,
    seed: int,
    epsilons: Iterable[float],
    *,
    n_ensemble: int,
    ensemble_epochs: int,
    batch_size: int,
) -> list[dict]:
    _load_experiment_dependencies()

    set_seed(seed)
    print(f"\n{'=' * 60}")
    print(f"DATASET: {dataset_name.upper()} | SEED {seed}")
    print(f"{'=' * 60}")

    train_ds, val_ds, _, _, _, class_names, _, _ = get_dataset(dataset_name)
    input_dim = train_ds.features.shape[1]
    num_classes = len(class_names)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    base_model = MCDropoutMLP(input_dim, num_classes).to(DEVICE)
    train_mc_dropout(base_model, train_loader, epochs=ensemble_epochs)

    ensemble = MCDropoutRashomonSet(base_model, n_samples=n_ensemble)
    ensemble.evaluate_all(val_loader, base_model=base_model)

    l_star = (
        ensemble.base_loss
        if ensemble.base_loss is not None
        else float(np.min(ensemble.val_losses))
    )
    rows = []
    for epsilon in epsilons:
        raw_size = ensemble.rashomon_size(epsilon)
        rows.append(
            {
                "Dataset": dataset_name,
                "Seed": seed,
                "Epsilon": float(epsilon),
                "Rashomon Size": raw_size,
                "Effective Size": max(raw_size, 1),
                "N Ensemble": n_ensemble,
                "L Star": float(l_star),
                "Threshold": float(l_star + epsilon),
                "Val Loss Min": float(np.min(ensemble.val_losses)),
                "Val Loss Mean": float(np.mean(ensemble.val_losses)),
                "Val Loss Max": float(np.max(ensemble.val_losses)),
            }
        )

    torch.cuda.empty_cache()
    return rows


def summarize_counts(seed_rows: Sequence[dict]) -> list[dict]:
    _load_experiment_dependencies()

    groups: dict[tuple[str, float], list[dict]] = {}
    for row in seed_rows:
        key = (row["Dataset"], row["Epsilon"])
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (dataset, epsilon), group in sorted(groups.items()):
        raw = [float(row["Rashomon Size"]) for row in group]
        effective = [float(row["Effective Size"]) for row in group]
        seeds = {row["Seed"] for row in group}
        summary_rows.append(
            {
                "Dataset": dataset,
                "Epsilon": epsilon,
                "N Seeds": len(seeds),
                "Rashomon Size Mean": float(np.mean(raw)),
                "Rashomon Size Std": _sample_std(raw),
                "Rashomon Size Min": int(np.min(raw)),
                "Rashomon Size Max": int(np.max(raw)),
                "Effective Size Mean": float(np.mean(effective)),
                "Effective Size Std": _sample_std(effective),
                "Effective Size Min": int(np.min(effective)),
                "Effective Size Max": int(np.max(effective)),
                "L Star Mean": float(np.mean([row["L Star"] for row in group])),
                "Val Loss Min Mean": float(np.mean([row["Val Loss Min"] for row in group])),
                "Val Loss Mean Mean": float(np.mean([row["Val Loss Mean"] for row in group])),
                "Val Loss Max Mean": float(np.mean([row["Val Loss Max"] for row in group])),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}.")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        print("(no rows)")
        return

    headers = list(rows[0])
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    print(" ".join(header.rjust(widths[header]) for header in headers))
    for row in rows:
        print(" ".join(str(row[header]).rjust(widths[header]) for header in headers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count how many MC-dropout models are in each Rashomon set."
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--epsilons",
        default=",".join(str(eps) for eps in DEFAULT_EPSILONS),
        help="Comma-separated epsilon values.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--n-ensemble",
        type=int,
        default=50,
        help="Number of frozen MC-dropout masks to sample.",
    )
    parser.add_argument(
        "--ensemble-epochs",
        type=int,
        default=50,
        help="Training epochs for the MC-dropout base model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--out-dir",
        default="rashomon_results",
        help="Directory for output CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = _parse_csv_values(args.datasets, str)
    epsilons = _parse_csv_values(args.epsilons, float)
    seeds = _parse_csv_values(args.seeds, int)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_name in datasets:
        for seed in seeds:
            rows.extend(
                count_for_dataset_seed(
                    dataset_name,
                    seed,
                    epsilons,
                    n_ensemble=args.n_ensemble,
                    ensemble_epochs=args.ensemble_epochs,
                    batch_size=args.batch_size,
                )
            )

    summary_rows = summarize_counts(rows)

    by_seed_path = out_dir / "rashomon_set_sizes_by_seed.csv"
    summary_path = out_dir / "rashomon_set_sizes_summary.csv"
    write_csv(by_seed_path, rows)
    write_csv(summary_path, summary_rows)

    print("\nRashomon set-size summary:")
    print_table(summary_rows)
    print(f"\nSaved by-seed counts to {by_seed_path}")
    print(f"Saved summary counts to {summary_path}")


if __name__ == "__main__":
    main()
