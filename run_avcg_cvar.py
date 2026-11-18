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

The default evaluation epsilons are 0, 0.01, 0.05, 0.1 and 0.5. Each dataset,
seed and alpha uses one trained generator. The same generated counterfactual and
the same two frozen mask banks are used at every evaluation epsilon. Only RVR
and Validity After Training depend on the evaluation epsilon in the tabular protocol.
CSV outputs retain the metric epsilon at instance, seed and summary levels.
The default alpha sweep is 0.1, 0.2, 0.5 and 1.0. Use --alphas 0.2 for the main table.
Evaluation draws are paired by seed and test index, so running alpha 0.2 alone
uses the same eligible inputs and latent draws as running it in the full sweep.

Examples
--------
python run_avcg_cvar.py --tabular-source main.py --alphas 0.2 --out avcg_cvar_sweep
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
METRIC_EPSILONS = (0.0, 0.01, 0.05, 0.1, 0.5)
GROUP_KEYS = ["Dataset", "Method", "Alpha", "Hypothesis Scope", "Hypothesis Epsilon",
              "Metric Epsilon"]
COUNT_COLUMNS = ["Hypothesis Count", "CVaR Tail Count", "RVR Mask Count", "VAT Mask Count",
                 "RVR Raw Mask Count", "VAT Raw Mask Count"]
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
        lines = content.splitlines()
        fence = re.match(r"^`{3,}", lines[0]).group()
        if lines[-1].strip() != fence:
            raise ValueError(f"Expected closing {fence} in {path}")
        content = "\n".join(lines[1:-1])

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
        "DEVICE",
        "rashomon_validity_ratio",
        "validity_after_training",
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
    metadata = {*GROUP_KEYS, *COUNT_COLUMNS, "Seed", "Test Index", "Status", "Error"}
    return [col for col in raw.select_dtypes(include="number") if col not in metadata]


def finite_values(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def seed_summary_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """One mean per metric within each seed and evaluation setting."""
    if raw.empty:
        return pd.DataFrame()

    keys = GROUP_KEYS + ["Seed"]
    metrics = _metric_columns(raw)
    rows = []
    # Full-scope CVaR has a missing training epsilon by design.
    for setting, group in raw.groupby(keys, dropna=False, sort=True):
        ok = group[group["Status"].eq("ok")]
        row = dict(zip(keys, setting))
        row.update({"N Attempted": len(group), "N Succeeded": len(ok),
                    "Success Rate": len(ok) / len(group)})
        for col in COUNT_COLUMNS:
            if col in group:
                row[col] = group[col].iloc[0]
        for col in metrics:
            values = finite_values(ok[col])
            row[col] = values.mean()
            row[col + "_n_instances"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


def summary_frame(raw: pd.DataFrame, per_seed=None) -> pd.DataFrame:
    """Mean and sample SD of finite seed-level means, separately at each epsilon."""
    if raw.empty:
        return pd.DataFrame()

    if per_seed is None:
        per_seed = seed_summary_frame(raw)
    metrics = _metric_columns(raw)
    rows = []
    for setting, group in per_seed.groupby(GROUP_KEYS, dropna=False, sort=True):
        row = dict(zip(GROUP_KEYS, setting))
        row["N Seeds"] = len(group)
        row["N Attempted"] = group["N Attempted"].sum()
        row["N Succeeded"] = group["N Succeeded"].sum()
        for col in ["Success Rate", *COUNT_COLUMNS, *metrics]:
            if col in group:
                values = finite_values(group[col])
                row[col] = values.mean()
                row[col + "_std"] = values.std(ddof=1)
                if col in metrics:
                    row[col + "_n_seeds"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)



def _safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("_") or "metric"


def plot_sweep(summary: pd.DataFrame, out_dir: Path):
    """Keep each plotted series at one fixed alpha or one fixed evaluation epsilon."""
    if summary.empty:
        return
    if summary["Alpha"].nunique() == 1 and summary["Metric Epsilon"].nunique() > 1:
        axis, fixed = "Metric Epsilon", "Alpha"
    else:
        axis, fixed = "Alpha", "Metric Epsilon"
    metadata = {*GROUP_KEYS, *COUNT_COLUMNS, "N Seeds", "N Attempted", "N Succeeded", "Success Rate"}
    metrics = [c for c in summary.select_dtypes(include="number")
               if c not in metadata and not c.endswith(("_std", "_n_seeds"))]

    for value, panel in summary.groupby(fixed, sort=True):
        folder = out_dir / "metric_plots" / f"{_safe_filename(fixed)}_{value:g}"
        folder.mkdir(parents=True, exist_ok=True)
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            for dataset, group in panel.groupby("Dataset", sort=True):
                points = group.sort_values(axis).dropna(subset=[metric])
                if points.empty:
                    continue
                ax.plot(points[axis], points[metric], marker="o", label=dataset)
                errors = points.dropna(subset=[metric + "_std"])
                if not errors.empty:
                    ax.errorbar(errors[axis], errors[metric], yerr=errors[metric + "_std"],
                                fmt="none", color=ax.lines[-1].get_color())
            if ax.lines:
                ax.set(xlabel=axis, ylabel=metric, title=f"AVCG-CVaR, {fixed} = {value:g}")
                ax.set_xticks(sorted(panel[axis].unique()))
                ax.grid(True, alpha=0.25)
                ax.legend()
                fig.tight_layout()
                fig.savefig(folder / f"{_safe_filename(metric)}.png", dpi=180)
            plt.close(fig)


def save_progress(rows, out_dir: Path, dataset: str):
    raw = pd.DataFrame(rows)
    per_seed = seed_summary_frame(raw)
    summary = summary_frame(raw, per_seed)
    raw.to_csv(out_dir / f"{dataset}_avcg_cvar_per_instance.csv", index=False)
    per_seed.to_csv(out_dir / f"{dataset}_avcg_cvar_per_seed.csv", index=False)
    summary.to_csv(out_dir / f"{dataset}_avcg_cvar_summary.csv", index=False)
    return raw, summary


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

        # The original selector may fall back to the best mask at tight thresholds.
        # Record its effective denominator and, when available, the unadjusted count.
        mask_counts = {}
        for epsilon in args.metric_epsilons:
            counts = {}
            for prefix, bank in (("RVR", ensemble), ("VAT", eval_ensemble)):
                counts[f"{prefix} Mask Count"] = len(bank.get_rashomon_models(epsilon))
                if hasattr(bank, "rashomon_size"):
                    counts[f"{prefix} Raw Mask Count"] = bank.rashomon_size(epsilon)
            mask_counts[epsilon] = counts

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
            # A separate stream per factual keeps selection independent of the
            # number of alphas and of metric calls on previous test inputs.
            filter_seed, cf_seed = np.random.SeedSequence(
                [seed, test_index, 4000]
            ).generate_state(2).tolist()
            src.set_seed(filter_seed)
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
                src.set_seed(cf_seed)
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

                    common_metrics = src._collect_metrics(
                        cf,
                        target,
                        predicted,
                        x,
                        base,
                        surrogates,
                        ae_by_class,
                        class_samples,
                        generation_time,
                        ensemble=None,
                        eval_ensemble=None,
                        epsilon=None,
                        generator_callable=(
                            regenerate if not args.skip_ric else None
                        ),
                        diversity=diversity,
                    )

                    cvar_value, _ = empirical_cvar_at_cf(
                        cf, target, members, alpha
                    )
                    common_metrics["CVaR NLL"] = cvar_value
                    common_metrics["Hypothesis Validity Ratio"] = hypothesis_validity_ratio(
                        cf, target, members
                    )
                    common_metrics["L2 Distance"] = torch.linalg.norm(cf - x).item()

                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    row["Error"] = f"{type(exc).__name__}: {exc}"

                # Frozen masks are deterministic. Applying the two original
                # metrics here leaves the stochastic generation and noise draws
                # independent of the number or order of evaluation epsilons.
                for epsilon in args.metric_epsilons:
                    result = {**row, "Metric Epsilon": epsilon, **mask_counts[epsilon]}
                    if not row["Error"]:
                        try:
                            rvr = src.rashomon_validity_ratio(cf, target, ensemble, epsilon)
                            vat = src.validity_after_training(cf, target, eval_ensemble, epsilon)
                            result.update(common_metrics)
                            result.update({"Rashomon Validity Ratio": rvr,
                                           "Validity After Training": vat, "Status": "ok"})
                        except (RuntimeError, ValueError, FloatingPointError) as exc:
                            result["Error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(result)

        if processed == 0:
            raise RuntimeError(f"{dataset} seed={seed}: no eligible test inputs")

        raw, summary = save_progress(rows, out_dir, dataset)
        print(
            f"{dataset} seed={seed}: evaluated {processed} correctly predicted inputs",
            flush=True,
        )

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
    parser.add_argument("--out", default="avcg_cvar_sweep")
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
        "--metric-epsilons", "--epsilons", "--metric-epsilon", dest="metric_epsilons",
        default=",".join(map(str, METRIC_EPSILONS)), metavar="E1,E2,...",
        help="Evaluation thresholds for RVR and Validity After Training (default: %(default)s)",
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
    try:
        args.alphas = tuple(float(v) for v in args.alphas.split(","))
        args.seeds = tuple(int(v) for v in args.seeds.split(","))
        args.metric_epsilons = tuple(float(v) for v in args.metric_epsilons.split(","))
    except ValueError as exc:
        parser.error(f"Invalid comma-separated numeric list: {exc}")

    if len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Datasets and seeds must have distinct entries")
    if not args.datasets or any(d not in DATASETS for d in args.datasets):
        parser.error(f"Datasets must be chosen from {', '.join(DATASETS)}")
    if len(set(args.alphas)) != len(args.alphas) or any(
        not 0 < a <= 1 for a in args.alphas
    ):
        parser.error("Alphas must be distinct and lie in (0, 1]")
    if not args.seeds:
        parser.error("At least one seed is required")
    if min(args.n_eval, args.n_ensemble, args.base_epochs, args.cvae_epochs) < 1:
        parser.error("Counts and epoch numbers must be positive")
    if args.n_train_hypotheses is not None and args.n_train_hypotheses < 1:
        parser.error("--n-train-hypotheses must be positive")
    if len(set(args.metric_epsilons)) != len(args.metric_epsilons):
        parser.error("Evaluation epsilons must be distinct")
    if any(not math.isfinite(e) or e < 0 for e in (*args.metric_epsilons, args.hypothesis_epsilon)):
        parser.error("Epsilon values must be finite and nonnegative")
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
            "evaluation_randomness": "paired by seed and test index across alphas and metric epsilons",
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
    if summary_all.empty or summary_all["N Succeeded"].eq(0).any():
        raise RuntimeError("At least one sweep setting has no successful evaluations. "
                           "See the Error column in the per-instance CSV.")
    plot_sweep(summary_all, output)

    print(f"\nComplete. Combined summary: {output / 'all_datasets_avcg_cvar_summary.csv'}")
    print(f"Metric plots: {output / 'metric_plots'}")


if __name__ == "__main__":
    main()
