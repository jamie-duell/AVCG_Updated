"""Run only BETARCE, ROAR and ElliCE against the attached experiment protocols.

Inputs: the two original experiment scripts (.py or fenced .md), plus the
companion robust_recourse_methods.py from the previous answer. Original
experiment main blocks and CVAE baselines are never executed.

Example:
    py run_robust_recourse_baselines.py \
      --tabular-source main.py \
      --image-source main_image.py \
      --methods-path robust_recourse_methods.py

Dependencies: torch, torchvision (image only), numpy, pandas, scipy,
scikit-learn, ucimlrepo (tabular only), matplotlib (original imports).

Interpretation: BETARCE uses the experiment's filtered frozen-dropout masks as
the admissible-model sample, not retrained models. ROAR uses a first-order,
target-versus-original-class local input linearization. ElliCE refits binary
logistic regression on frozen penultimate embeddings for each class pair.
Its ellipsoid certificate is for that surrogate in feature space; it does not
certify the original CNN/MLP in input space, nor full multiclass validity.
These adaptations are explicit so method names alone do not imply equivalence
to the papers' original binary, linear experimental protocols.
The sweep epsilon is a validation-loss tolerance for the frozen-mask ensemble
and a training-loss tolerance for ElliCE's pairwise surrogate; these numbers
are shown on the same x axis but are not calibrated to identical uncertainty
sets. ROAR uses a separate --roar-radius that does not vary with epsilon.

Statistical reporting: instance-level metrics are first averaged within each
random seed. Final summaries report the mean +/- sample standard deviation
(ddof=1) across seed-level means.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Subset


TABULAR_DATASETS = ("breast_cancer", "heart_disease", "spambase", "adult_income")
IMAGE_DATASETS = ("mnist", "cifar10")
DEFAULT_EPSILONS = (0.0, 0.01, 0.05, 0.1, 0.5)
DEFAULT_SEEDS = (11, 22, 33, 44, 55)
METHOD_NAMES = ("BETARCE", "ROAR-local", "ElliCE-embedding")


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


def predict_labels(models, arrays, shape, device, batch_size=128):
    models.eval()
    X = np.asarray(arrays, dtype=np.float32).reshape(-1, *shape)
    output = []
    for first in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[first:first + batch_size]).to(device)
        with torch.no_grad():
            output.append(models(xb).argmax(1).cpu().numpy())
    return np.concatenate(output)


def simple_base_cf(model, x0, target, lower, upper, *,
                   steps=350, lr=0.05, proximity=0.002):
    """Separate per-instance base CFE for BETARCE; no prior baseline rerun."""
    model.eval()
    cf = x0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([cf], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            loss = -model(cf)[0, target] + proximity * (cf - x0).square().sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                cf.clamp_(lower, upper)
                if model(cf).argmax(1).item() == target:
                    return cf.detach()
    raise RuntimeError("Base CF failed to reach the target class.")


def make_ellice_pair(model, train_loader, kind, original, target, reg, methods, device):
    """Fit binary logistic ERM on frozen embeddings for this class pair."""
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
        fit = LogisticRegression(C=1.0 / (len(X) * reg), max_iter=1000)
        fit.fit(X, y)
    w = fit.coef_[0].astype(float)
    b = float(fit.intercept_[0])
    H = methods.ellice_logistic_hessian(X, w, b, l2_regularization=reg)
    # sklearn's binary LogisticRegression does not regularize the intercept.
    H[-1, -1] -= reg
    H = 0.5 * (H + H.T)
    if np.linalg.eigvalsh(H)[0] <= 0:
        raise RuntimeError("ElliCE Hessian is not positive definite.")
    return (torch.tensor(w, device=device, dtype=torch.float32),
            torch.tensor(b, device=device, dtype=torch.float32),
            torch.tensor(np.linalg.inv(H), device=device, dtype=torch.float32))


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


def generate_roar(model, x, original, target, lower, upper, methods, args, device):
    """ROAR on local target-versus-original log-odds input linearization."""
    model.eval()
    xp = x.detach().clone().requires_grad_(True)
    lp = model(xp)
    margin = lp[0, target] - lp[0, original]
    gradient = torch.autograd.grad(margin, xp)[0].flatten().detach()
    x_flat = x.flatten().detach()
    bias = float(margin.detach() - torch.dot(gradient, x_flat))
    config = methods.ROARConfig(
        epsilon=args.roar_radius, perturbation="l2",
        lambda_cost=args.roar_cost, cost="l2", outer_lr=args.roar_lr,
        inner_lr=args.roar_inner_lr, outer_steps=args.roar_steps,
        inner_steps=args.roar_inner_steps, patience=args.roar_steps + 1,
    )
    result = methods.roar_linear(
        x_flat.cpu().numpy(), gradient.cpu().numpy(), bias=bias,
        y_target=1, config=config,
        feature_min=np.broadcast_to(np.asarray(lower.cpu()), x_flat.shape),
        feature_max=np.broadcast_to(np.asarray(upper.cpu()), x_flat.shape),
        device=str(device),
    )
    return torch.as_tensor(result["x_cf"], device=device).reshape_as(x), result


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


def seed_frame(rows):
    """
    Stage 1 aggregation: average instance-level metrics within each random seed.

    One row is produced for every
        Kind x Dataset x Epsilon x Method x Seed
    experimental condition.
    """
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    metadata = {
        "Kind", "Dataset", "Seed", "Test Index", "Epsilon", "Method",
        "Status", "Error", "Target", "Original"
    }

    metrics = [
        col for col in raw.columns
        if col not in metadata
        and pd.api.types.is_numeric_dtype(raw[col])
    ]

    out = []

    for keys, group in raw.groupby(
        ["Kind", "Dataset", "Epsilon", "Method", "Seed"],
        sort=False,
    ):
        record = dict(
            zip(
                ("Kind", "Dataset", "Epsilon", "Method", "Seed"),
                keys,
            )
        )

        record["N Attempted"] = int(len(group))
        record["N Succeeded"] = int((group.Status == "ok").sum())
        record["Success Rate"] = (
            record["N Succeeded"] / record["N Attempted"]
            if record["N Attempted"] else np.nan
        )

        for col in metrics:
            vals = (
                pd.to_numeric(group[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            record[col] = float(vals.mean()) if len(vals) else np.nan
            record[col + "_n_instances"] = int(len(vals))

        out.append(record)

    return pd.DataFrame(out)


def summarize(rows):
    """
    Two-stage aggregation for paper reporting.

    Stage 1:
        Average each metric over evaluated instances WITHIN each seed.

    Stage 2:
        Compute the mean and sample standard deviation (ddof=1) across the
        seed-level means.

    Therefore each ``*_std`` field in the final summary measures variability
    BETWEEN independent random-seed runs, rather than variability between
    individual counterfactual instances pooled across runs.

    Returns
    -------
    raw : DataFrame
        One row per attempted method/instance/epsilon evaluation.
    per_seed : DataFrame
        One row per Kind/Dataset/Epsilon/Method/Seed.
    summary : DataFrame
        Final paper-facing mean +/- SD across seed-level means.
    """
    raw = pd.DataFrame(rows)

    if raw.empty:
        return raw, raw.copy(), raw.copy()

    per_seed = seed_frame(rows)

    # Metrics available after within-seed aggregation. Helper count columns are
    # deliberately excluded from the mean/SD calculation.
    non_metrics = {
        "Kind", "Dataset", "Epsilon", "Method", "Seed",
        "N Attempted", "N Succeeded", "Success Rate",
    }

    metrics = [
        col for col in per_seed.columns
        if col not in non_metrics
        and not col.endswith("_n_instances")
        and pd.api.types.is_numeric_dtype(per_seed[col])
    ]

    out = []

    for keys, group in per_seed.groupby(
        ["Kind", "Dataset", "Epsilon", "Method"],
        sort=False,
    ):
        record = dict(
            zip(
                ("Kind", "Dataset", "Epsilon", "Method"),
                keys,
            )
        )

        record["N Seeds"] = int(group["Seed"].nunique())
        record["N Attempted"] = int(group["N Attempted"].sum())
        record["N Succeeded"] = int(group["N Succeeded"].sum())

        # Keep the pooled count-based rate as a diagnostic.
        record["Overall Success Rate"] = (
            record["N Succeeded"] / record["N Attempted"]
            if record["N Attempted"] else np.nan
        )

        # For paper-facing uncertainty, success rate is treated in exactly the
        # same way as the other metrics: first calculate a rate within each
        # seed, then mean +/- sample SD across seeds.
        seed_success_rates = (
            pd.to_numeric(group["Success Rate"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(seed_success_rates):
            record["Success Rate"] = float(seed_success_rates.mean())
            record["Success Rate_std"] = (
                float(seed_success_rates.std(ddof=1))
                if len(seed_success_rates) > 1
                else 0.0
            )
            record["Success Rate_n_seeds"] = int(len(seed_success_rates))
        else:
            record["Success Rate"] = np.nan
            record["Success Rate_std"] = np.nan
            record["Success Rate_n_seeds"] = 0

        for col in metrics:
            seed_vals = (
                pd.to_numeric(group[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if len(seed_vals):
                record[col] = float(seed_vals.mean())
                record[col + "_std"] = (
                    float(seed_vals.std(ddof=1))
                    if len(seed_vals) > 1
                    else 0.0
                )
                record[col + "_n_seeds"] = int(len(seed_vals))
            else:
                record[col] = np.nan
                record[col + "_std"] = np.nan
                record[col + "_n_seeds"] = 0

        out.append(record)

    summary = pd.DataFrame(out)

    return raw, per_seed, summary


def save_results(rows, out_dir, dataset):
    """
    Persist all three levels of the experiment:

      * per-instance observations;
      * per-seed means;
      * final mean +/- sample SD across seed-level means.
    """
    raw, per_seed, summary = summarize(rows)

    raw.to_csv(
        out_dir / f"{dataset}_new_methods_per_instance.csv",
        index=False,
    )
    per_seed.to_csv(
        out_dir / f"{dataset}_new_methods_per_seed.csv",
        index=False,
    )
    summary.to_csv(
        out_dir / f"{dataset}_new_methods_summary.csv",
        index=False,
    )

    return raw, per_seed, summary


def run_dataset(src, kind, dataset, methods, args, out_dir):
    rows = []
    device = src.DEVICE
    ensemble_size = args.n_ensemble_tabular if kind == "tabular" else args.n_ensemble_image
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

        base = (src.MCDropoutMLP(dim, num_classes) if kind == "tabular"
                else src.MCDropoutCNN(channels, num_classes, size)).to(device)
        src.train_mc_dropout(base, train_loader, epochs=epochs)
        base.eval()
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        ensemble = src.MCDropoutRashomonSet(base, n_samples=ensemble_size)
        ensemble.evaluate_all(val_loader, base_model=base)
        eval_ensemble = src.MCDropoutRashomonSet(
            base, n_samples=max(20, ensemble_size // 2))
        eval_ensemble.evaluate_all(val_loader, base_model=base)

        surrogates = []
        for i in range(5 if kind == "tabular" else 3):
            src.set_seed(seed * 100 + i + 1)
            s = (src.StandardMLP(dim, num_classes) if kind == "tabular"
                 else src.StandardCNN(channels, num_classes, size)).to(device)
            src.train_standard_mlp(s, train_loader,
                                   epochs=30 if kind == "tabular" else epochs,
                                   name=f"CMV surrogate {i+1}")
            s.eval()
            surrogates.append(s)
        src.set_seed(seed + 1000)
        ae_by_class, class_samples = prepare_ae_metrics(
            src, kind, train_ds, train_loader, dim, channels, size,
            num_classes, device)
        model_shape = (dim,) if kind == "tabular" else (channels, size, size)
        lower_tensor = torch.as_tensor(lower, device=device)
        upper_tensor = torch.as_tensor(upper, device=device)
        pair_cache = {}
        raw_members = {
            eps: [model for model, loss in zip(ensemble.models, ensemble.val_losses)
                  if loss <= ensemble.base_loss + eps]
            for eps in args.epsilons
        }
        # Do not count the experiment's single-best-mask fallback as multiple
        # admissible draws for BETARCE's posterior.
        def predict_fn(model, X):
            return predict_labels(model, X, model_shape, device)

        processed = 0
        for test_index, (x, y) in enumerate(test_loader):
            if processed >= eval_n:
                break
            x = x.to(device)
            base.eval()
            # Same correctly-predicted-instance filter as attached experiments.
            # Their averaging of 20 stochastic log-probabilities is reproduced.
            base.train()
            with torch.no_grad():
                pred = torch.stack([base(x) for _ in range(20)]).mean(0).argmax(1).item()
            base.eval()
            if pred != int(y.item()):
                continue
            processed += 1
            target = 1 - pred if kind == "tabular" else (pred + 1) % num_classes
            base_cf = None
            base_error = None
            base_start = time.perf_counter()
            try:
                base_cf = simple_base_cf(
                    base, x, target, lower_tensor, upper_tensor,
                    steps=args.base_cf_steps)
            except RuntimeError as exc:
                base_error = str(exc)
            base_seconds = time.perf_counter() - base_start

            # ROAR does not depend on the outer Rashomon epsilon; generate it
            # once and report the same counterfactual across sweep rows.
            roar_start = time.perf_counter()
            try:
                roar_cf, roar_result = generate_roar(
                    base, x, pred, target, lower_tensor, upper_tensor,
                    methods, args, device)
                roar_error = None
            except (RuntimeError, ValueError) as exc:
                roar_cf, roar_result, roar_error = None, None, str(exc)
            roar_seconds = time.perf_counter() - roar_start

            pair = (pred, target)
            pair_error = None
            if pair not in pair_cache:
                try:
                    pair_cache[pair] = make_ellice_pair(
                        base, fit_loader, kind, pred, target,
                        args.ellice_regularization, methods, device)
                except (RuntimeError, ValueError) as exc:
                    pair_error = str(exc)
            theta = pair_cache.get(pair)

            def do_roar(z):
                cf, _ = generate_roar(
                    base, z, pred, target, lower_tensor, upper_tensor,
                    methods, args, device)
                return cf, 0.0

            for eps in args.epsilons:
                members = raw_members[eps]
                for name in METHOD_NAMES:
                    row = {
                        "Kind": kind, "Dataset": dataset, "Seed": seed,
                        "Test Index": test_index, "Epsilon": eps,
                        "Method": name, "Original": pred, "Target": target,
                        "Status": "failed", "Error": "",
                        "Beta k": len(members) if name == "BETARCE" else np.nan,
                    }
                    t0 = time.perf_counter()
                    try:
                        if name == "BETARCE":
                            if base_error is not None:
                                raise RuntimeError(base_error)
                            # Even all k successful trials cannot meet this
                            # posterior condition for a small admissible set.
                            best_bound = methods.beta_robustness_lower_bound(
                                len(members), len(members), args.beta_alpha
                            ) if members else -np.inf
                            if best_bound < args.beta_delta:
                                raise RuntimeError(
                                    f"Only {len(members)} admissible masks; "
                                    "desired Beta certificate is unattainable.")
                            flat = x.detach().cpu().numpy().reshape(-1)
                            def base_fn(_model, _orig, _y):
                                # Reuse the separately generated base CF.
                                return base_cf.detach().cpu().numpy().reshape(-1)
                            def do_beta(z):
                                result = methods.betarce(
                                    x_orig=z.detach().cpu().numpy().reshape(-1),
                                    base_model=base, base_cf_fn=base_fn,
                                    admissible_models=members,
                                    delta=args.beta_delta, alpha=args.beta_alpha,
                                    eta=args.beta_eta_image if kind == "image" else args.beta_eta_tabular,
                                    n=args.beta_candidates,
                                    max_search_steps=args.beta_search_steps,
                                    max_warmup_steps=15,
                                    feature_min=np.broadcast_to(
                                        np.asarray(lower_tensor.cpu()), flat.shape),
                                    feature_max=np.broadcast_to(
                                        np.asarray(upper_tensor.cpu()), flat.shape),
                                    predict_fn=predict_fn,
                                    random_state=None,
                                )
                                return (torch.from_numpy(result["x_cf"].copy())
                                        .to(device).reshape_as(x), result)
                            cf, beta_result = do_beta(x)
                            row["Beta lower"] = beta_result["beta_lower_bound"]
                            generator = (lambda z: (do_beta(z)[0], 0.0)) if args.ric else None
                        elif name == "ROAR-local":
                            if roar_error:
                                raise RuntimeError(roar_error)
                            cf = roar_cf
                            row["ROAR worst p"] = roar_result["p_target_worst_shift"]
                            generator = do_roar if args.ric else None
                        else:
                            if pair_error:
                                raise RuntimeError(pair_error)
                            def do_ellice(z):
                                return generate_ellice(
                                    base, z, pred, target, theta, eps,
                                    lower_tensor, upper_tensor, kind,
                                    steps=args.ellice_steps, lr=args.ellice_lr,
                                    prox=args.ellice_proximity)
                            cf, robust_logit = do_ellice(x)
                            row["ElliCE worst logit"] = robust_logit
                            generator = (lambda z: (do_ellice(z)[0], 0.0)) if args.ric else None

                        generation_seconds = (
                            time.perf_counter() - t0 + base_seconds
                            if name == "BETARCE" else
                            roar_seconds if name == "ROAR-local" else
                            time.perf_counter() - t0
                        )
                        if not torch.isfinite(cf).all():
                            raise RuntimeError("Nonfinite counterfactual.")
                        def safe_generator(z):
                            try:
                                return generator(z)
                            except (RuntimeError, ValueError):
                                return None, 0.0

                        metric = src._collect_metrics(
                            cf, target, pred, x, base, surrogates, ae_by_class,
                            class_samples, generation_seconds,
                            ensemble=ensemble, eval_ensemble=eval_ensemble,
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
                # Persist progress at each completed input in case of interruption.
                save_results(rows, out_dir, dataset)
            print(f"{dataset} seed={seed}: {processed}/{eval_n} correctly predicted inputs", flush=True)
        print(f"Completed {dataset}, seed {seed}: evaluated {processed} inputs.", flush=True)
    return save_results(rows, out_dir, dataset)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("tabular", "image", "both"), default="both")
    parser.add_argument("--tabular-source", default="main.py")
    parser.add_argument("--image-source", default="main_image.py")
    parser.add_argument("--methods-path", default="robust_recourse_methods.py")
    parser.add_argument("--out", default="robust_baseline_results")
    parser.add_argument("--tabular-datasets", default=",".join(TABULAR_DATASETS))
    parser.add_argument("--image-datasets", default=",".join(IMAGE_DATASETS))
    parser.add_argument("--epsilons", default=",".join(map(str, DEFAULT_EPSILONS)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--n-eval-tabular", type=int, default=100)
    parser.add_argument("--n-eval-image", type=int, default=100)
    parser.add_argument("--n-ensemble-tabular", type=int, default=50)
    parser.add_argument("--n-ensemble-image", type=int, default=20)
    parser.add_argument("--epochs-tabular", type=int, default=50)
    parser.add_argument("--epochs-image", type=int, default=20)
    parser.add_argument("--base-cf-steps", type=int, default=350)
    parser.add_argument("--beta-delta", type=float, default=0.8)
    parser.add_argument("--beta-alpha", type=float, default=0.9)
    parser.add_argument("--beta-candidates", type=int, default=64)
    parser.add_argument("--beta-search-steps", type=int, default=25)
    parser.add_argument("--beta-eta-tabular", type=float, default=0.1)
    parser.add_argument("--beta-eta-image", type=float, default=1.0)
    parser.add_argument("--roar-radius", type=float, default=0.05)
    parser.add_argument("--roar-cost", type=float, default=0.03)
    parser.add_argument("--roar-lr", type=float, default=0.05)
    parser.add_argument("--roar-inner-lr", type=float, default=0.05)
    parser.add_argument("--roar-steps", type=int, default=150)
    parser.add_argument("--roar-inner-steps", type=int, default=5)
    parser.add_argument("--ellice-regularization", type=float, default=0.001)
    parser.add_argument("--ellice-steps", type=int, default=350)
    parser.add_argument("--ellice-lr", type=float, default=0.04)
    parser.add_argument("--ellice-proximity", type=float, default=0.002)
    parser.add_argument("--skip-ric", action="store_true",
                        help="Leave R_IC columns as NaN; saves two extra generations per CF.")
    args = parser.parse_args()
    args.seeds = tuple(int(v) for v in args.seeds.split(","))
    args.epsilons = tuple(float(v) for v in args.epsilons.split(","))
    args.ric = not args.skip_ric
    if any(e < 0 for e in args.epsilons):
        parser.error("Epsilons must be nonnegative.")
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
            print(f"Saved new baselines only: {subdir}", flush=True)


if __name__ == "__main__":
    main()
