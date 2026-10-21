"""Run AVCG-CVaR on the existing tabular AVCG experiment without rerunning baselines.

This script mirrors the model, data, generator, evaluation, and metric machinery used
by the existing AVCG tabular code, but replaces the mean hypothesis loss in AVCG with
an empirical Conditional Value at Risk (CVaR) objective.

For a set of K sampled hypotheses and per-hypothesis loss
    l_theta(x') = -log P_theta(y_target | x'),
CVaR_alpha is approximated by the mean of the largest ceil(alpha * K) losses.
Therefore:
    alpha = 1.0  -> ordinary mean hypothesis loss;
    alpha -> 0   -> increasingly worst-tail / worst-case-like training.

By default, P_Phi is the full empirical MC-dropout frozen-mask distribution used by
MCDropoutRashomonSet. A Rashomon-restricted CVaR variant can instead be selected via
--hypothesis-scope rashomon.

The script sweeps alpha over {0.1, 0.2, 0.5, 1.0}, writes per-instance and summary
CSV files, and creates one line plot per metric showing how the metric changes with
alpha. It does not rerun the existing baselines.

Examples
--------
python run_avcg_cvar.py --tabular-source main.py --out avcg_cvar_results
python run_avcg_cvar.py --tabular-source 'Pasted markdown(20260926-105957).md'
python run_avcg_cvar.py --tabular-source main.py --hypothesis-scope rashomon \
    --hypothesis-epsilon 0.05 --metric-epsilon 0.05

Requires the original experiment's dependencies: torch, numpy, pandas,
scikit-learn, ucimlrepo, matplotlib. The original get_dataset function may download
UCI datasets. Outputs AVCG-CVaR results only.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


DATASETS = ("breast_cancer", "heart_disease", "spambase", "adult_income")
ALPHAS = (0.1, 0.2, 0.5, 1.0)
SEEDS = (11, 22, 33, 44, 55)


# ---------------------------------------------------------------------------
# Load the user's existing AVCG experiment without executing its main block.
# ---------------------------------------------------------------------------
def load_original(path: Path) -> ModuleType:
    """Load definitions from the original .py or fenced .md, excluding main."""
    if not path.is_file():
        raise FileNotFoundError(f"Original tabular experiment not found: {path}")

    content = path.read_text(encoding="utf-8-sig").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]

    tree = ast.parse(content, filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        )
    ]

    source = ModuleType("avcg_original_tabular")
    source.__file__ = str(path)
    sys.modules[source.__name__] = source
    exec(compile(tree, str(path), "exec"), source.__dict__)

    required = (
        "MCDropoutMLP",
        "MCDropoutRashomonSet",
        "StandardMLP",
        "TabularCVAE",
        "AutoEncoder",
        "get_dataset",
        "set_seed",
        "train_mc_dropout",
        "train_standard_mlp",
        "train_autoencoder",
        "_collect_metrics",
        "generate_cf_amortized",
    )
    for name in required:
        if not hasattr(source, name):
            raise ValueError(f"Original experiment is missing required definition: {name}")
    return source


# ---------------------------------------------------------------------------
# CVaR hypothesis objective.
# ---------------------------------------------------------------------------
def select_hypotheses(ensemble, scope: str, epsilon: float):
    """Return the fixed hypotheses defining P_Phi for CVaR training."""
    if scope == "full":
        members = list(ensemble.models)
    elif scope == "rashomon":
        members = list(ensemble.get_rashomon_models(epsilon))
    else:
        raise ValueError(f"Unknown hypothesis scope: {scope}")

    if not members:
        raise ValueError("The selected hypothesis distribution is empty")

    # Predictor weights are fixed: gradients are required only w.r.t. x' so that
    # they can flow back into the CVAE generator.
    for model in members:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return members


def empirical_cvar_loss(x_prime, targets, members, alpha: float):
    """Empirical CVaR of -log target probability over the hypothesis dimension.

    Parameters
    ----------
    x_prime : Tensor [B, d]
        Generated counterfactual batch.
    targets : LongTensor [B]
        Target counterfactual labels.
    members : sequence of predictors
        Fixed sampled hypotheses defining the empirical P_Phi.
    alpha : float in (0, 1]
        Fraction of worst hypotheses averaged. alpha=1 is the ordinary mean.

    Returns
    -------
    scalar Tensor
        Mean per-instance CVaR across the minibatch.
    int
        Number of hypotheses in the empirical tail.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    if len(members) < 1:
        raise ValueError("CVaR requires at least one hypothesis")

    # Each FrozenMaskMLP returns log-softmax probabilities. Stack as [B, K].
    log_target = torch.stack(
        [
            model(x_prime).gather(1, targets.view(-1, 1)).squeeze(1)
            for model in members
        ],
        dim=1,
    )
    losses = -log_target

    # Same finite-sample convention used in the manuscript: largest ceil(alpha K).
    tail_count = max(1, int(math.ceil(alpha * len(members))))
    worst_tail = torch.topk(losses, k=tail_count, dim=1, largest=True).values
    return worst_tail.mean(dim=1).mean(), tail_count


def train_avcg_cvar(
    cvae,
    train_loader,
    members,
    *,
    alpha: float,
    epochs: int,
    lambda_class: float = 1.0,
    lambda_kl: float = 0.1,
    lambda_prox: float = 0.1,
):
    """Train the existing TabularCVAE with an empirical CVaR validity term."""
    optimizer = torch.optim.Adam(cvae.parameters(), lr=0.001, weight_decay=1e-4)
    device = next(cvae.parameters()).device
    tail_count = max(1, int(math.ceil(alpha * len(members))))

    for epoch in range(epochs):
        cvae.train()
        total_loss = 0.0
        total_cvar = 0.0
        count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            target = 1 - y

            mu, logvar = cvae.encode(x, target)
            kl = -0.5 * torch.sum(
                1 + logvar - mu.square() - logvar.exp(), dim=1
            ).mean()
            z = cvae.reparameterize(mu, logvar)
            cf = cvae.decode(z, target)

            cvar, _ = empirical_cvar_loss(cf, target, members, alpha)
            proximity = F.mse_loss(cf, x, reduction="mean")
            loss = (
                lambda_class * cvar
                + lambda_kl * kl
                + lambda_prox * proximity
            )

            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Nonfinite AVCG-CVaR training loss")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * len(x)
            total_cvar += cvar.detach().item() * len(x)
            count += len(x)

        print(
            f"    alpha={alpha:g} epoch {epoch + 1}/{epochs}: "
            f"loss={total_loss / count:.5f} "
            f"cvar={total_cvar / count:.5f} "
            f"tail={tail_count}/{len(members)}",
            flush=True,
        )


def hypothesis_validity_ratio(cf, target, members):
    """Fraction of the training-distribution hypotheses accepting the CF."""
    votes = 0
    with torch.no_grad():
        for model in members:
            model.eval()
            votes += int(model(cf).argmax(1).item() == int(target))
    return votes / len(members)


def empirical_cvar_at_cf(cf, target, members, alpha: float):
    """CVaR NLL at one generated counterfactual, for reporting only."""
    target_tensor = torch.tensor([int(target)], dtype=torch.long, device=cf.device)
    with torch.no_grad():
        value, tail_count = empirical_cvar_loss(cf, target_tensor, members, alpha)
    return float(value.item()), int(tail_count)


# ---------------------------------------------------------------------------
# Recreate the metric-side models exactly as in the companion AVCG-L script.
# ---------------------------------------------------------------------------
def prepare_original_metrics(
    src, train_ds, train_loader, input_dim, latent_dim, n_classes, device
):
    """Create the class AEs and in-class samples used by the original metrics."""
    # Preserve the original random-number schedule used in the companion script.
    general_ae = src.AutoEncoder(input_dim, latent_dim).to(device)
    src.train_autoencoder(general_ae, train_loader, epochs=40, name="General AE")

    ae_by_class = {}
    for c in range(n_classes):
        indices = (train_ds.labels == c).nonzero(as_tuple=True)[0].tolist()
        if not indices:
            raise ValueError(f"No training examples for class {c}")
        subset = DataLoader(Subset(train_ds, indices), batch_size=32, shuffle=True)
        ae = src.AutoEncoder(input_dim, latent_dim).to(device)
        src.train_autoencoder(ae, subset, epochs=40, name=f"AE class {c}")
        ae.eval()
        ae_by_class[c] = ae

    samples = {
        c: train_ds.features[train_ds.labels == c].cpu()
        for c in range(n_classes)
    }
    return ae_by_class, samples


# ---------------------------------------------------------------------------
# Summaries and plots.
# ---------------------------------------------------------------------------
def _metric_columns(raw: pd.DataFrame):
    """Return numeric instance-level metric columns, excluding metadata."""
    metadata = {
        "Dataset",
        "Seed",
        "Test Index",
        "Alpha",
        "Method",
        "Status",
        "Error",
        "Hypothesis Scope",
        "Hypothesis Epsilon",
        "Metric Epsilon",
        "Hypothesis Count",
        "CVaR Tail Count",
    }
    return [
        col
        for col in raw.columns
        if col not in metadata and pd.api.types.is_numeric_dtype(raw[col])
    ]


def seed_summary_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute one metric mean per independent seed/run.

    Each row in the returned frame corresponds to one
    (dataset, alpha, method, seed) combination. Instance-level metric values are
    averaged *within* that seed first. This frame is the intermediate level used
    for the final mean +/- SD across seeds.
    """
    if raw.empty:
        return pd.DataFrame()

    metrics = _metric_columns(raw)
    rows = []

    for (dataset, alpha, method, seed), group in raw.groupby(
        ["Dataset", "Alpha", "Method", "Seed"], sort=True
    ):
        attempted = len(group)
        succeeded = int((group["Status"] == "ok").sum())

        row = {
            "Dataset": dataset,
            "Alpha": float(alpha),
            "Method": method,
            "Seed": int(seed),
            "N Attempted": attempted,
            "N Succeeded": succeeded,
            "Success Rate": succeeded / attempted if attempted else np.nan,
        }

        # Settings / run-level quantities. Hypothesis Count and CVaR Tail Count
        # can vary by seed for a Rashomon-restricted hypothesis distribution, so
        # retain their seed-specific values here.
        for col in (
            "Hypothesis Count",
            "CVaR Tail Count",
            "Hypothesis Epsilon",
            "Metric Epsilon",
        ):
            if col in group.columns:
                vals = (
                    pd.to_numeric(group[col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                row[col] = float(vals.iloc[0]) if len(vals) else np.nan

        # Stage 1: instance-level observations -> ONE mean for this seed.
        for col in metrics:
            values = (
                pd.to_numeric(group[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            row[col] = float(values.mean()) if len(values) else np.nan
            row[col + "_n_instances"] = int(len(values))

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["Dataset", "Alpha", "Method", "Seed"])
        .reset_index(drop=True)
    )


def summary_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Report mean +/- sample SD across independent seed-level means.

    Two-stage aggregation:
      1. Average instance-level metric values within each seed.
      2. Average those seed-level means and compute sample SD (ddof=1) across seeds.

    N Attempted / N Succeeded remain total counts across seeds so the familiar
    success-count presentation (e.g. 500/500) is preserved.
    """
    if raw.empty:
        return pd.DataFrame()

    per_seed = seed_summary_frame(raw)
    if per_seed.empty:
        return pd.DataFrame()

    metrics = _metric_columns(raw)
    rows = []

    for (dataset, alpha, method), group in per_seed.groupby(
        ["Dataset", "Alpha", "Method"], sort=True
    ):
        row = {
            "Dataset": dataset,
            "Alpha": float(alpha),
            "Method": method,
            "N Seeds": int(group["Seed"].nunique()),
            "N Attempted": int(group["N Attempted"].sum()),
            "N Succeeded": int(group["N Succeeded"].sum()),
        }

        # Success rate is itself a run-level statistic, so report its mean and
        # between-seed sample SD. With equal N per seed, its mean equals the
        # pooled success proportion.
        success_rates = (
            pd.to_numeric(group["Success Rate"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(success_rates):
            row["Success Rate"] = float(success_rates.mean())
            row["Success Rate_std"] = (
                float(success_rates.std(ddof=1)) if len(success_rates) > 1 else 0.0
            )
        else:
            row["Success Rate"] = np.nan
            row["Success Rate_std"] = np.nan

        # Fixed experiment settings.
        for col in ("Hypothesis Epsilon", "Metric Epsilon"):
            if col in group.columns:
                vals = (
                    pd.to_numeric(group[col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                row[col] = float(vals.iloc[0]) if len(vals) else np.nan

        # Run-level hypothesis counts can differ by seed when using a restricted
        # Rashomon set, so aggregate them transparently as mean +/- SD too.
        for col in ("Hypothesis Count", "CVaR Tail Count"):
            if col in group.columns:
                vals = (
                    pd.to_numeric(group[col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if len(vals):
                    row[col] = float(vals.mean())
                    row[col + "_std"] = (
                        float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                    )
                else:
                    row[col] = np.nan
                    row[col + "_std"] = np.nan

        # Stage 2: seed-level means -> overall mean +/- sample SD across seeds.
        for col in metrics:
            values = (
                pd.to_numeric(group[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(values):
                row[col] = float(values.mean())
                row[col + "_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
                row[col + "_n_seeds"] = int(len(values))
            else:
                row[col] = np.nan
                row[col + "_std"] = np.nan
                row[col + "_n_seeds"] = 0

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["Dataset", "Alpha", "Method"])
        .reset_index(drop=True)
    )

def _safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("_") or "metric"


def plot_alpha_sweep(summary: pd.DataFrame, out_dir: Path):
    """Create one plot per mean metric with alpha on the x-axis."""
    if summary.empty:
        return

    plot_dir = out_dir / "metric_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    non_metric = {
        "Alpha",
        "N Seeds",
        "N Attempted",
        "N Succeeded",
        "Success Rate",
        "Hypothesis Count",
        "CVaR Tail Count",
        "Hypothesis Epsilon",
        "Metric Epsilon",
    }
    metric_cols = [
        c
        for c in summary.columns
        if c not in {"Dataset", "Method"}
        and c not in non_metric
        and not c.endswith("_std")
        and not c.endswith("_n_seeds")
        and pd.api.types.is_numeric_dtype(summary[c])
    ]

    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        plotted = False

        for dataset, group in summary.groupby("Dataset", sort=True):
            group = group.sort_values("Alpha")
            y = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            x = group["Alpha"].to_numpy(dtype=float)
            finite = np.isfinite(y)
            if not finite.any():
                continue

            std_col = metric + "_std"
            if std_col in group.columns:
                yerr = pd.to_numeric(group[std_col], errors="coerce").to_numpy(dtype=float)
                yerr = np.where(np.isfinite(yerr), yerr, 0.0)
                ax.errorbar(x[finite], y[finite], yerr=yerr[finite], marker="o", label=dataset)
            else:
                ax.plot(x[finite], y[finite], marker="o", label=dataset)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel(r"CVaR tail fraction $\alpha$")
        ax.set_ylabel(metric)
        ax.set_title(f"AVCG-CVaR: {metric} vs alpha")
        ax.set_xticks(sorted(summary["Alpha"].dropna().unique()))
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{_safe_filename(metric)}_vs_alpha.png", dpi=180)
        plt.close(fig)


def save_progress(rows, out_dir: Path, dataset: str):
    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / f"{dataset}_avcg_cvar_per_instance.csv", index=False)
    seed_summary_frame(raw).to_csv(
        out_dir / f"{dataset}_avcg_cvar_per_seed.csv", index=False
    )
    summary_frame(raw).to_csv(
        out_dir / f"{dataset}_avcg_cvar_summary.csv", index=False
    )


# ---------------------------------------------------------------------------
# Main experiment.
# ---------------------------------------------------------------------------
def run_dataset(src, dataset: str, args, out_dir: Path):
    device = src.DEVICE
    rows = []

    for seed in args.seeds:
        src.set_seed(seed)
        train_ds, val_ds, test_ds, _, _, classes, lower, upper = src.get_dataset(dataset)
        n_classes = len(classes)
        if n_classes != 2:
            raise ValueError("This AVCG-CVaR script targets the binary tabular experiment")

        input_dim = train_ds.features.shape[1]
        latent_dim = max(5, input_dim // 4)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        # Same construction predictor and frozen-mask hypothesis machinery.
        base = src.MCDropoutMLP(input_dim, n_classes).to(device)
        src.train_mc_dropout(base, train_loader, epochs=args.base_epochs)
        base.eval()

        ensemble = src.MCDropoutRashomonSet(base, n_samples=args.n_ensemble)
        ensemble.evaluate_all(val_loader, base_model=base)
        eval_ensemble = src.MCDropoutRashomonSet(
            base, n_samples=max(20, args.n_ensemble // 2)
        )
        eval_ensemble.evaluate_all(val_loader, base_model=base)

        members = select_hypotheses(
            ensemble, args.hypothesis_scope, args.hypothesis_epsilon
        )
        if args.n_train_hypotheses is not None:
            if args.n_train_hypotheses > len(members):
                raise ValueError(
                    f"Requested {args.n_train_hypotheses} training hypotheses but "
                    f"only {len(members)} are available"
                )
            members = members[: args.n_train_hypotheses]

        # Same five independently trained standard MLP surrogates used by the
        # existing Cross-Model Validity metric.
        surrogates = []
        for j in range(5):
            src.set_seed(seed * 100 + j + 1)
            model = src.StandardMLP(input_dim, n_classes).to(device)
            src.train_standard_mlp(
                model, train_loader, epochs=30, name=f"CMV surrogate {j + 1}/5"
            )
            model.eval()
            surrogates.append(model)

        src.set_seed(seed + 1000)
        ae_by_class, class_samples = prepare_original_metrics(
            src, train_ds, train_loader, input_dim, latent_dim, n_classes, device
        )

        # Train one generator per alpha. Resetting the seed ensures that all alpha
        # settings start from the same CVAE initialisation and minibatch RNG state.
        cvae_by_alpha = {}
        for alpha in args.alphas:
            src.set_seed(seed + 3000)
            cvae = src.TabularCVAE(input_dim, n_classes, latent_dim).to(device)
            print(
                f"  Training AVCG-CVaR alpha={alpha:g} "
                f"with {len(members)} {args.hypothesis_scope} hypotheses",
                flush=True,
            )
            train_avcg_cvar(
                cvae,
                train_loader,
                members,
                alpha=alpha,
                epochs=args.cvae_epochs,
                lambda_class=args.lambda_class,
                lambda_kl=args.lambda_kl,
                lambda_prox=args.lambda_prox,
            )
            cvae_by_alpha[alpha] = cvae

        # Same correctly-predicted filter used by the companion script.
        base.train()
        processed = 0
        for test_index, (x, y) in enumerate(test_loader):
            if processed >= args.n_eval:
                break

            x, y = x.to(device), y.to(device)
            base.train()
            with torch.no_grad():
                predicted = (
                    torch.stack([base(x) for _ in range(20)])
                    .mean(0)
                    .argmax(1)
                    .item()
                )
            if predicted != y.item():
                continue

            processed += 1
            target = 1 - predicted

            for alpha in args.alphas:
                cvae = cvae_by_alpha[alpha]
                tail_count = max(1, int(math.ceil(alpha * len(members))))
                row = {
                    "Dataset": dataset,
                    "Seed": seed,
                    "Test Index": test_index,
                    "Alpha": alpha,
                    "Method": "AVCG-CVaR",
                    "Status": "failed",
                    "Error": "",
                    "Hypothesis Scope": args.hypothesis_scope,
                    "Hypothesis Epsilon": (
                        args.hypothesis_epsilon
                        if args.hypothesis_scope == "rashomon"
                        else np.nan
                    ),
                    "Metric Epsilon": args.metric_epsilon,
                    "Hypothesis Count": len(members),
                    "CVaR Tail Count": tail_count,
                }

                try:
                    cf, generation_time, diversity = src.generate_cf_amortized(
                        cvae, x, target, lower, upper
                    )
                    if not bool(torch.isfinite(cf).all()):
                        raise FloatingPointError("Nonfinite AVCG-CVaR counterfactual")

                    def regenerate(perturbed, _cvae=cvae, _target=target):
                        return src.generate_cf_amortized(
                            _cvae, perturbed, _target, lower, upper
                        )[:2]

                    metric = src._collect_metrics(
                        cf,
                        target,
                        predicted,
                        x,
                        base,
                        surrogates,
                        ae_by_class,
                        class_samples,
                        generation_time,
                        ensemble=ensemble,
                        eval_ensemble=eval_ensemble,
                        epsilon=args.metric_epsilon,
                        generator_callable=(
                            regenerate if not args.skip_ric else None
                        ),
                        diversity=diversity,
                    )
                    row.update(metric)

                    cvar_value, _ = empirical_cvar_at_cf(
                        cf, target, members, alpha
                    )
                    row["CVaR NLL"] = cvar_value
                    row["Hypothesis Validity Ratio"] = hypothesis_validity_ratio(
                        cf, target, members
                    )
                    row["L2 Distance"] = torch.linalg.norm(cf - x).item()
                    row["Status"] = "ok"

                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    row["Error"] = f"{type(exc).__name__}: {exc}"

                rows.append(row)

        save_progress(rows, out_dir, dataset)
        print(
            f"{dataset} seed={seed}: evaluated {processed} correctly predicted inputs",
            flush=True,
        )

    raw = pd.DataFrame(rows)
    per_seed = seed_summary_frame(raw)
    summary = summary_frame(raw)
    raw.to_csv(out_dir / f"{dataset}_avcg_cvar_per_instance.csv", index=False)
    per_seed.to_csv(out_dir / f"{dataset}_avcg_cvar_per_seed.csv", index=False)
    summary.to_csv(out_dir / f"{dataset}_avcg_cvar_summary.csv", index=False)
    return raw, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tabular-source",
        default="main.py",
        help="Original AVCG tabular .py or fenced Markdown source",
    )
    parser.add_argument("--out", default="avcg_cvar_results")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--alphas", default=",".join(map(str, ALPHAS)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-ensemble", type=int, default=50)
    parser.add_argument("--base-epochs", type=int, default=50)
    parser.add_argument("--cvae-epochs", type=int, default=50)

    parser.add_argument(
        "--hypothesis-scope",
        choices=("full", "rashomon"),
        default="full",
        help=(
            "full = CVaR over all frozen MC-dropout hypotheses; "
            "rashomon = CVaR over epsilon-retained hypotheses"
        ),
    )
    parser.add_argument(
        "--hypothesis-epsilon",
        type=float,
        default=0.05,
        help="Rashomon epsilon used only when --hypothesis-scope rashomon",
    )
    parser.add_argument(
        "--metric-epsilon",
        type=float,
        default=0.8,
        help=(
            "Epsilon used only by the existing RVR and Validity-After-Training "
            "evaluation metrics; it does not affect full-scope CVaR training"
        ),
    )
    parser.add_argument(
        "--n-train-hypotheses",
        type=int,
        default=None,
        help="Optional fixed prefix of the selected hypothesis set; default uses all",
    )

    parser.add_argument("--lambda-class", type=float, default=1.0)
    parser.add_argument("--lambda-kl", type=float, default=0.1)
    parser.add_argument("--lambda-prox", type=float, default=0.1)
    parser.add_argument("--skip-ric", action="store_true")

    args = parser.parse_args()
    args.datasets = tuple(v.strip() for v in args.datasets.split(",") if v.strip())
    args.alphas = tuple(float(v) for v in args.alphas.split(","))
    args.seeds = tuple(int(v) for v in args.seeds.split(","))

    if not args.datasets or any(d not in DATASETS for d in args.datasets):
        parser.error(f"Datasets must be chosen from {', '.join(DATASETS)}")
    if len(set(args.alphas)) != len(args.alphas) or any(
        a <= 0 or a > 1 for a in args.alphas
    ):
        parser.error("Alphas must be distinct and lie in (0, 1]")
    if not args.seeds:
        parser.error("At least one seed is required")
    if min(args.n_eval, args.n_ensemble, args.base_epochs, args.cvae_epochs) < 1:
        parser.error("Counts and epoch numbers must be positive")
    if args.n_train_hypotheses is not None and args.n_train_hypotheses < 1:
        parser.error("--n-train-hypotheses must be positive")
    if args.hypothesis_epsilon < 0 or args.metric_epsilon < 0:
        parser.error("Epsilon values must be nonnegative")
    if min(args.lambda_class, args.lambda_kl, args.lambda_prox) < 0:
        parser.error("Loss weights must be nonnegative")

    return args


def main():
    args = parse_args()
    src = load_original(Path(args.tabular_source).expanduser().resolve())
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    settings = dict(vars(args))
    settings.update(
        {
            "cvar_definition": "mean of largest ceil(alpha*K) per-hypothesis NLLs per instance",
            "alpha_semantics": "alpha=1 mean risk; smaller alpha emphasizes worse hypotheses",
            "hypothesis_family": "existing frozen MC-dropout masks",
            "metric_models": "existing base predictor, 5 CMV MLPs, class AEs, RVR/VAT ensembles",
            "aggregation": (
                "instance metrics are averaged within each seed first; final values are "
                "mean +/- sample SD (ddof=1) across seed-level means"
            ),
        }
    )
    (output / "avcg_cvar_run_config.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    all_raw = []
    all_summary = []

    for dataset in args.datasets:
        subdir = output / dataset
        subdir.mkdir(parents=True, exist_ok=True)
        raw, summary = run_dataset(src, dataset, args, subdir)
        all_raw.append(raw)
        all_summary.append(summary)
        print(f"Saved AVCG-CVaR results: {subdir}", flush=True)

    raw_all = pd.concat(all_raw, ignore_index=True) if all_raw else pd.DataFrame()
    summary_all = (
        pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    )
    raw_all.to_csv(output / "all_datasets_avcg_cvar_per_instance.csv", index=False)
    seed_summary_frame(raw_all).to_csv(
        output / "all_datasets_avcg_cvar_per_seed.csv", index=False
    )
    summary_all.to_csv(output / "all_datasets_avcg_cvar_summary.csv", index=False)
    plot_alpha_sweep(summary_all, output)

    print(f"\nComplete. Combined summary: {output / 'all_datasets_avcg_cvar_summary.csv'}")
    print(f"Metric plots: {output / 'metric_plots'}")


if __name__ == "__main__":
    main()
