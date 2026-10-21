"""Run ElliCE only against the attached experiment protocols.

ElliCE is fitted as a binary logistic surrogate on frozen penultimate
embeddings for each original/target class pair. Its ellipsoidal robustness
certificate applies to that surrogate in embedding space.

Unlike the previous shared sweep, ElliCE epsilon is calibrated to the fitted
pairwise surrogate's own regularized training objective. By default we test
0%, 1%, 2.5%, 5%, and 10% of that training objective.

Example:
    py run_ellice_only.py \
      --kind tabular \
      --tabular-source main.py \
      --methods-path robust_cf_methods_ellice_calibrated.py

Outputs contain both:
  * Epsilon Fraction: the relative tolerance (e.g. 0.05 = 5% of train loss)
  * ElliCE Epsilon: the actual pair-specific epsilon used in the certificate
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Subset


TABULAR_DATASETS = ("breast_cancer", "heart_disease", "spambase", "adult_income")
IMAGE_DATASETS = ("mnist", "cifar10")
DEFAULT_ELLICE_EPSILON_FRACTIONS = (0.001, 0.01, 0.025, 0.05, 0.10)
DEFAULT_SEEDS = (11, 22, 33, 44, 55)
METHOD_NAME = "ElliCE-embedding"


def _locate(path: str, *, base: Path) -> Path:
    requested = Path(path).expanduser()
    candidates = (requested, base / requested, base / "upload" / requested,
                  base / "downloads" / requested)
    for item in candidates:
        if item.is_file():
            return item.resolve()
    raise FileNotFoundError(f"Cannot find {path!r}; pass its path explicitly.")


def load_experiment_source(path: Path) -> ModuleType:
    """Use original definitions, ignoring their main block and fenced Markdown.

    Only execute local source the user provided. AST removal prevents
    rerunning its full experiment; top-level imports and DEVICE assignment run.
    """
    source = path.read_text(encoding="utf-8-sig").strip()
    if source.startswith("```"):
        source = source.split("\n", 1)[1].rsplit("```", 1)[0]
    tree = ast.parse(source, filename=str(path))
    tree.body = [
        node for node in tree.body
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        )
    ]
    module = ModuleType("existing_experiment")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def load_methods(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("robust_recourse_methods", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def feature_map(model, x, kind: str):
    """Frozen penultimate features, with gradients available for input x."""
    if kind == "tabular":
        h = F.relu(model.fc1(x))
        return F.relu(model.fc2(h))
    h = model.pool(F.relu(model.conv1(x)))
    h = model.pool(F.relu(model.conv2(h)))
    return F.relu(model.fc1(h.reshape(len(x), -1)))


def make_ellice_pair(model, train_loader, kind, original, target, reg, methods, device):
    """Fit ElliCE's binary logistic surrogate and calibrate its train-loss scale."""
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for xb, yb in train_loader:
            keep = (yb == original) | (yb == target)
            if not bool(keep.any()):
                continue
            h = feature_map(model, xb[keep].to(device), kind)
            embeddings.append(h.cpu().numpy())
            labels.append((yb[keep] == target).int().numpy())

    X = np.concatenate(embeddings)
    y = np.concatenate(labels)
    if len(np.unique(y)) != 2:
        raise RuntimeError("Both classes required to refit ElliCE's last layer.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        # This matches mean logistic loss + (reg/2)||w||^2.
        # sklearn does not regularize the intercept.
        fit = LogisticRegression(C=1.0 / (len(X) * reg), max_iter=1000)
        fit.fit(X, y)

    w = fit.coef_[0].astype(float)
    b = float(fit.intercept_[0])

    H = methods.ellice_logistic_hessian(
        X, w, b, l2_regularization=reg
    )
    # sklearn's binary LogisticRegression does not regularize the intercept.
    H[-1, -1] -= reg
    H = 0.5 * (H + H.T)
    if np.linalg.eigvalsh(H)[0] <= 0:
        raise RuntimeError("ElliCE Hessian is not positive definite.")

    # Pair-specific regularized training objective used to calibrate epsilon.
    if hasattr(methods, "ellice_logistic_objective"):
        train_loss = methods.ellice_logistic_objective(
            X,
            y,
            w,
            bias=b,
            l2_regularization=reg,
            regularize_intercept=False,
        )
    else:
        logits = X @ w + b
        data_loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
        train_loss = data_loss + 0.5 * reg * float(np.dot(w, w))

    theta = (
        torch.tensor(w, device=device, dtype=torch.float32),
        torch.tensor(b, device=device, dtype=torch.float32),
        torch.tensor(np.linalg.inv(H), device=device, dtype=torch.float32),
    )
    return theta, float(train_loss)

def generate_ellice(model, x0, original, target, theta, epsilon, lower, upper,
                    kind, *, steps=350, lr=0.04, prox=0.002):
    """Appendix B.4 style continuous input search on a frozen feature map."""
    model.eval()
    w, b, H_inv = theta
    xcf = x0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([xcf], lr=lr)
    best, best_distance, best_score = None, float("inf"), float("nan")
    for _ in range(steps):
        opt.zero_grad()
        h = feature_map(model, xcf, kind)[0]
        z = torch.cat([h, h.new_ones(1)])
        nominal = torch.dot(h, w) + b
        q = torch.dot(z, H_inv @ z).clamp_min(1e-12)
        robust = nominal - math.sqrt(2 * epsilon) * torch.sqrt(q)
        loss = F.softplus(-robust) + prox * (xcf - x0).square().sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            xcf.clamp_(lower, upper)
            # Recompute at the projected point before checking the guarantee.
            h = feature_map(model, xcf, kind)[0]
            z = torch.cat([h, h.new_ones(1)])
            robust_value = (torch.dot(h, w) + b
                            - math.sqrt(2 * epsilon)
                            * torch.sqrt(torch.dot(z, H_inv @ z).clamp_min(1e-12))).item()
            if robust_value >= 0 and model(xcf).argmax(1).item() == target:
                d = torch.linalg.norm(xcf - x0).item()
                if d < best_distance:
                    best, best_distance, best_score = xcf.detach().clone(), d, robust_value
                break
    if best is None:
        raise RuntimeError("No ellipsoid-feasible, original-model-valid CF found.")
    return best, best_score


def prepare_ae_metrics(src, kind, train_ds, train_loader, dim, channels, size,
                       num_classes, device):
    ae_by_class, class_samples = {}, {}
    if kind == "tabular":
        features, labels = train_ds.features, train_ds.labels
        class_samples = {c: features[labels == c].cpu() for c in range(num_classes)}
        latent = max(5, dim // 4)
    for c in range(num_classes):
        if kind == "tabular":
            indices = (labels == c).nonzero(as_tuple=True)[0].tolist()
            ae = src.AutoEncoder(dim, latent).to(device)
            epochs = 40
        else:
            indices = [i for i, (_, label) in enumerate(train_ds) if label == c]
            ae = src.ConvAutoEncoder(channels, size, 64).to(device)
            class_samples[c] = torch.stack([train_ds[i][0] for i in indices[:500]])
            epochs = 5
        if not indices:
            raise RuntimeError(f"No training examples for class {c}.")
        subset_loader = DataLoader(Subset(train_ds, indices), batch_size=32, shuffle=True)
        src.train_autoencoder(ae, subset_loader, epochs=epochs, name=f"AE class {c}")
        ae.eval()
        ae_by_class[c] = ae
    return ae_by_class, class_samples


def summarize(rows):
    """Aggregate metrics in two stages: instances -> seed -> across-seed SD.

    Failed generations count toward attempted/success rates but are omitted from
    conditional metric means. Across-seed SD is sample SD (ddof=1). It is NaN
    when only one seed has a finite value, rather than spuriously reporting 0.

    Returns:
        raw:            one record per attempted generation
        summary:        one record per dataset/epsilon/method, with seed-level SD
        seed_summary:   intermediate seed-level metrics for audit/reproduction
    """
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, pd.DataFrame(), pd.DataFrame()

    metadata = {
        "Kind", "Dataset", "Seed", "Test Index", "Epsilon Fraction",
        "Method", "Status", "Error", "Target", "Original",
    }
    metric_cols = [
        col for col in raw.columns
        if col not in metadata and pd.api.types.is_numeric_dtype(raw[col])
    ]
    group_cols = ["Kind", "Dataset", "Epsilon Fraction", "Method"]

    # One observation per independent run, not one per test instance.
    seed_records = []
    for keys, group in raw.groupby(group_cols + ["Seed"], sort=False):
        record = dict(zip(group_cols + ["Seed"], keys))
        success = group["Status"].eq("ok")
        record["N Attempted"] = len(group)
        record["N Succeeded"] = int(success.sum())
        record["Success Rate"] = float(success.mean())

        for col in metric_cols:
            # Metrics for generated CFs are conditional on successful generation.
            # Calibrated epsilons / training objectives are recorded for all
            # attempts and do not depend on successful CF generation.
            is_calibration = col in {"ElliCE Epsilon", "ElliCE Train Objective"}
            values = pd.to_numeric(
                group[col] if is_calibration else group.loc[success, col],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).dropna()
            record[col] = float(values.mean()) if len(values) else np.nan
            record[col + "_n"] = len(values)

        # End-to-end validity, counting unsuccessful generation as invalid.
        if "Validity" in raw.columns:
            validity = pd.to_numeric(
                group["Validity"], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
            record["Attempted Validity"] = float(
                validity.where(success, 0.0).fillna(0.0).mean()
            )

        seed_records.append(record)

    seed_summary = pd.DataFrame(seed_records)
    summary_records = []
    means = ["Success Rate", *metric_cols]
    if "Attempted Validity" in seed_summary.columns:
        means.append("Attempted Validity")

    for keys, group in seed_summary.groupby(group_cols, sort=False):
        record = dict(zip(group_cols, keys))
        record["N Attempted"] = int(group["N Attempted"].sum())
        record["N Succeeded"] = int(group["N Succeeded"].sum())
        record["N Seeds"] = len(group)
        record["N Seeds With Success"] = int(group["N Succeeded"].gt(0).sum())
        record["Pooled Success Rate"] = (
            record["N Succeeded"] / record["N Attempted"]
        )

        for col in means:
            vals = pd.to_numeric(group[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            record[col] = float(vals.mean()) if len(vals) else np.nan
            record[col + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            record[col + "_n_seeds"] = len(vals)

        summary_records.append(record)

    return raw, pd.DataFrame(summary_records), seed_summary

def run_dataset(src, kind, dataset, methods, args, out_dir):
    rows = []
    device = src.DEVICE
    ensemble_size = (
        args.n_ensemble_tabular if kind == "tabular" else args.n_ensemble_image
    )
    eval_n = args.n_eval_tabular if kind == "tabular" else args.n_eval_image
    epochs = args.epochs_tabular if kind == "tabular" else args.epochs_image

    for seed in args.seeds:
        src.set_seed(seed)
        parts = src.get_dataset(dataset)
        train_ds, val_ds, test_ds = parts[:3]
        lower, upper = parts[6:8]
        num_classes = len(parts[5])
        channels, size = (None, None) if kind == "tabular" else parts[8:10]
        dim = train_ds.features.shape[1] if kind == "tabular" else None

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        fit_loader = DataLoader(train_ds, batch_size=256, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        # Base model is still required because ElliCE operates on its frozen
        # penultimate embedding and the experiment metrics evaluate the CF on it.
        base = (
            src.MCDropoutMLP(dim, num_classes)
            if kind == "tabular"
            else src.MCDropoutCNN(channels, num_classes, size)
        ).to(device)
        src.train_mc_dropout(base, train_loader, epochs=epochs)
        base.eval()
        for parameter in base.parameters():
            parameter.requires_grad_(False)

        # Retain these because _collect_metrics uses them for the same robustness
        # metrics as the original experiment. They are evaluation machinery, not
        # ElliCE generation; these objects are retained only for evaluation metrics.
        ensemble = src.MCDropoutRashomonSet(base, n_samples=ensemble_size)
        ensemble.evaluate_all(val_loader, base_model=base)
        eval_ensemble = src.MCDropoutRashomonSet(
            base, n_samples=max(20, ensemble_size // 2)
        )
        eval_ensemble.evaluate_all(val_loader, base_model=base)

        surrogates = []
        for i in range(5 if kind == "tabular" else 3):
            src.set_seed(seed * 100 + i + 1)
            surrogate = (
                src.StandardMLP(dim, num_classes)
                if kind == "tabular"
                else src.StandardCNN(channels, num_classes, size)
            ).to(device)
            src.train_standard_mlp(
                surrogate,
                train_loader,
                epochs=30 if kind == "tabular" else epochs,
                name=f"CMV surrogate {i + 1}",
            )
            surrogate.eval()
            surrogates.append(surrogate)

        src.set_seed(seed + 1000)
        ae_by_class, class_samples = prepare_ae_metrics(
            src,
            kind,
            train_ds,
            train_loader,
            dim,
            channels,
            size,
            num_classes,
            device,
        )

        lower_tensor = torch.as_tensor(lower, device=device)
        upper_tensor = torch.as_tensor(upper, device=device)
        pair_cache = {}

        processed = 0
        for test_index, (x, y) in enumerate(test_loader):
            if processed >= eval_n:
                break

            x = x.to(device)

            # Same correctly-predicted-instance filter as the previous runner.
            base.train()
            with torch.no_grad():
                pred = (
                    torch.stack([base(x) for _ in range(20)])
                    .mean(0)
                    .argmax(1)
                    .item()
                )
            base.eval()

            if pred != int(y.item()):
                continue

            processed += 1
            target = 1 - pred if kind == "tabular" else (pred + 1) % num_classes
            pair = (pred, target)

            pair_error = None
            if pair not in pair_cache:
                try:
                    pair_cache[pair] = make_ellice_pair(
                        base,
                        fit_loader,
                        kind,
                        pred,
                        target,
                        args.ellice_regularization,
                        methods,
                        device,
                    )
                    theta_cached, train_loss_cached = pair_cache[pair]
                    actual_eps = tuple(
                        train_loss_cached * frac
                        for frac in args.ellice_epsilon_fractions
                    )
                    print(
                        f"{dataset} seed={seed} pair={pair}: "
                        f"ElliCE train objective={train_loss_cached:.6g}; "
                        f"epsilons={actual_eps}",
                        flush=True,
                    )
                except (RuntimeError, ValueError) as exc:
                    pair_error = str(exc)
                    pair_cache[pair] = None

            pair_data = pair_cache.get(pair)
            if pair_data is None:
                theta = None
                train_loss = np.nan
            else:
                theta, train_loss = pair_data

            for eps_fraction in args.ellice_epsilon_fractions:
                eps = (
                    float(train_loss * eps_fraction)
                    if np.isfinite(train_loss)
                    else np.nan
                )

                row = {
                    "Kind": kind,
                    "Dataset": dataset,
                    "Seed": seed,
                    "Test Index": test_index,
                    "Epsilon Fraction": eps_fraction,
                    "ElliCE Epsilon": eps,
                    "ElliCE Train Objective": train_loss,
                    "Method": METHOD_NAME,
                    "Original": pred,
                    "Target": target,
                    "Status": "failed",
                    "Error": "",
                }

                t0 = time.perf_counter()
                try:
                    if pair_error:
                        raise RuntimeError(pair_error)
                    if theta is None or not np.isfinite(eps):
                        raise RuntimeError("ElliCE pairwise surrogate was unavailable.")

                    def do_ellice(z):
                        return generate_ellice(
                            base,
                            z,
                            pred,
                            target,
                            theta,
                            eps,
                            lower_tensor,
                            upper_tensor,
                            kind,
                            steps=args.ellice_steps,
                            lr=args.ellice_lr,
                            prox=args.ellice_proximity,
                        )

                    cf, robust_logit = do_ellice(x)
                    generation_seconds = time.perf_counter() - t0

                    row["ElliCE worst logit"] = robust_logit

                    generator = (
                        (lambda z: (do_ellice(z)[0], 0.0))
                        if args.ric
                        else None
                    )

                    if not torch.isfinite(cf).all():
                        raise RuntimeError("Nonfinite counterfactual.")

                    def safe_generator(z):
                        try:
                            return generator(z)
                        except (RuntimeError, ValueError):
                            return None, 0.0

                    metric = src._collect_metrics(
                        cf,
                        target,
                        pred,
                        x,
                        base,
                        surrogates,
                        ae_by_class,
                        class_samples,
                        generation_seconds,
                        ensemble=ensemble,
                        eval_ensemble=eval_ensemble,
                        # Preserve the original experiment metric call.
                        # This is the actual ElliCE tolerance, not a ROAR radius.
                        epsilon=eps,
                        generator_callable=safe_generator if generator else None,
                        diversity=0.0,
                    )
                    row.update(metric)
                    row["L2 Distance"] = torch.linalg.norm(cf - x).item()
                    row["Status"] = "ok"

                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    row["Error"] = f"{type(exc).__name__}: {exc}"

                rows.append(row)

                # Persist progress after every epsilon/input.
                raw, summary, seed_summary = summarize(rows)
                raw.to_csv(
                    out_dir / f"{dataset}_ellice_per_instance.csv",
                    index=False,
                )
                summary.to_csv(
                    out_dir / f"{dataset}_ellice_summary.csv",
                    index=False,
                )
                seed_summary.to_csv(
                    out_dir / f"{dataset}_ellice_per_seed.csv",
                    index=False,
                )

            print(
                f"{dataset} seed={seed}: "
                f"{processed}/{eval_n} correctly predicted inputs",
                flush=True,
            )

        print(
            f"Completed {dataset}, seed {seed}: evaluated {processed} inputs.",
            flush=True,
        )

    return summarize(rows)

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kind", choices=("tabular", "image", "both"), default="both"
    )
    parser.add_argument("--tabular-source", default="main.py")
    parser.add_argument("--image-source", default="main_image.py")
    parser.add_argument(
        "--methods-path",
        default="robust_cf_methods_ellice_calibrated.py",
    )
    parser.add_argument("--out", default="ellice_results")
    parser.add_argument(
        "--tabular-datasets", default=",".join(TABULAR_DATASETS)
    )
    parser.add_argument(
        "--image-datasets", default=",".join(IMAGE_DATASETS)
    )
    parser.add_argument(
        "--ellice-epsilon-fractions",
        default=",".join(map(str, DEFAULT_ELLICE_EPSILON_FRACTIONS)),
        help=(
            "Comma-separated fractions of each pairwise ElliCE surrogate's "
            "regularized training objective. Default: 0,0.01,0.025,0.05,0.10"
        ),
    )
    parser.add_argument(
        "--seeds", default=",".join(map(str, DEFAULT_SEEDS))
    )
    parser.add_argument("--n-eval-tabular", type=int, default=100)
    parser.add_argument("--n-eval-image", type=int, default=100)
    parser.add_argument("--n-ensemble-tabular", type=int, default=50)
    parser.add_argument("--n-ensemble-image", type=int, default=20)
    parser.add_argument("--epochs-tabular", type=int, default=50)
    parser.add_argument("--epochs-image", type=int, default=20)
    parser.add_argument(
        "--ellice-regularization", type=float, default=0.001
    )
    parser.add_argument("--ellice-steps", type=int, default=350)
    parser.add_argument("--ellice-lr", type=float, default=0.04)
    parser.add_argument("--ellice-proximity", type=float, default=0.002)
    parser.add_argument(
        "--skip-ric",
        action="store_true",
        help="Leave R_IC columns as NaN; saves extra ElliCE generations per CF.",
    )

    args = parser.parse_args()
    args.seeds = tuple(int(v) for v in args.seeds.split(","))
    args.ellice_epsilon_fractions = tuple(
        float(v) for v in args.ellice_epsilon_fractions.split(",")
    )
    args.ric = not args.skip_ric

    if any(v < 0 for v in args.ellice_epsilon_fractions):
        parser.error("ElliCE epsilon fractions must be nonnegative.")

    return args

def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    methods = load_methods(_locate(args.methods_path, base=here))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    kinds = ("tabular", "image") if args.kind == "both" else (args.kind,)
    for kind in kinds:
        src_path = args.tabular_source if kind == "tabular" else args.image_source
        src = load_experiment_source(_locate(src_path, base=here))
        datasets = args.tabular_datasets if kind == "tabular" else args.image_datasets
        for dataset in datasets.split(","):
            subdir = out / kind / dataset.strip()
            subdir.mkdir(parents=True, exist_ok=True)
            run_dataset(src, kind, dataset.strip(), methods, args, subdir)
            print(f"Saved ElliCE results: {subdir}", flush=True)


if __name__ == "__main__":
    main()

