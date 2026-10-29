"""Run AVCG-L on the tabular AVCG experiment without rerunning its baselines.

AVCG-L uses the attached CVAE architecture and objective, replacing the
Rashomon MC-dropout training ensemble with a last-layer Laplace approximation.
The feature extractor is frozen and the two-class softmax head is represented
by the difference of its two weight vectors. Its precision is the observed
binary-logistic Hessian on the training set plus an isotropic prior precision.
The centre is the trained head, an approximate MAP point because the original
network was trained end-to-end with dropout. This is NOT a full-network Laplace
approximation, and it does not provide a certified recourse guarantee.

For each epsilon, keep posterior draws whose validation NLL is at most the
undropped base model's validation NLL + epsilon. The trained central head is
always included. This uses the same validation-loss units and threshold rule
as the attached AVCG-R experiment; the two posterior families still differ.
Evaluation uses the ORIGINAL MC-dropout ensembles for RVR and VAT, so those
metrics remain directly defined as in the attached experiment. A separate
Laplace Validity Ratio is recorded for the selected Laplace heads.

Example:
    python run_avcg_l.py --tabular-source main.py --out avcg_l_results
    python run_avcg_l.py --tabular-source 'Pasted markdown(20260926-105957).md'

Requires the original experiment's dependencies: torch, numpy, pandas,
scipy, scikit-learn, ucimlrepo, matplotlib. Downloads UCI datasets as its
original get_dataset function does. Writes AVCG-L results only.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


DATASETS = ("breast_cancer", "heart_disease", "spambase", "adult_income")
EPSILONS = (0.0, 0.01, 0.05, 0.1, 0.5)
SEEDS = (11, 22, 33, 44, 55)


def load_original(path: Path) -> ModuleType:
    """Load definitions from the original .py or fenced .md, excluding main."""
    if not path.is_file():
        raise FileNotFoundError(f"Original tabular experiment not found: {path}")
    content = path.read_text(encoding="utf-8-sig").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    tree = ast.parse(content, filename=str(path))
    tree.body = [
        node for node in tree.body
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
    for name in ("MCDropoutMLP", "MCDropoutRashomonSet", "StandardMLP",
                 "TabularCVAE", "AutoEncoder", "_collect_metrics",
                 "generate_cf_amortized"):
        if not hasattr(source, name):
            raise ValueError(f"Original experiment is missing {name}")
    return source


def frozen_features(model, x):
    """The original MLP's deterministic penultimate activations."""
    h = F.relu(model.fc1(x))
    return F.relu(model.fc2(h))


def augment(h):
    return torch.cat((h, torch.ones((len(h), 1), dtype=h.dtype, device=h.device)), 1)


@dataclass
class LaplaceHeads:
    model: torch.nn.Module
    heads: torch.Tensor                  # (K, hidden_dim + 1), row 0 = centre
    validation_losses: torch.Tensor      # NLL per head, same order as heads
    precision_cholesky: torch.Tensor

    def indices(self, epsilon: float) -> torch.Tensor:
        if epsilon < 0:
            raise ValueError("Epsilon must be nonnegative")
        tolerance = 1e-9
        selected = torch.where(
            self.validation_losses <= self.validation_losses[0] + epsilon + tolerance
        )[0]
        # The central, undropped head must never disappear due to roundoff.
        if not bool((selected == 0).any()):
            selected = torch.cat((selected.new_zeros(1), selected))
        return selected

    def expected_log_prob(self, x, targets, selected):
        """Mean of log p(y_target | x, theta) over selected classifier heads."""
        # Calling the original evaluator toggles train mode for MC dropout.
        # Restore deterministic features for the Laplace posterior each time.
        self.model.eval()
        z = augment(frozen_features(self.model, x))
        scores = z @ self.heads[selected].T            # [batch, heads]
        signs = (2 * targets.float() - 1).unsqueeze(1)
        return F.logsigmoid(signs * scores).mean()

    def target_vote_fraction(self, x, targets, selected):
        self.model.eval()
        with torch.no_grad():
            z = augment(frozen_features(self.model, x))
            scores = z @ self.heads[selected].T
            prediction = scores >= 0
            return (prediction == targets.bool().unsqueeze(1)).float().mean().item()


def fit_laplace_heads(model, fit_loader, val_loader, *, prior_precision,
                      posterior_scale, n_heads, seed):
    """Full (d+1)x(d+1) last-layer Hessian; Gaussian heads from its inverse."""
    if n_heads < 1 or prior_precision <= 0 or posterior_scale <= 0:
        raise ValueError("n_heads, prior_precision, and posterior_scale must be > 0")
    model.eval()
    if model.fc3.out_features != 2:
        raise ValueError("AVCG-L's binary last-layer formulation requires two classes")
    device = next(model.parameters()).device
    weights = model.fc3.weight.detach()
    biases = model.fc3.bias.detach()
    centre = torch.cat((weights[1] - weights[0], (biases[1] - biases[0]).view(1)))
    dim = centre.numel()

    # NLL is a sum, so prior_precision is in matching sum-loss units.
    precision = torch.eye(dim, device=device, dtype=torch.float64) * prior_precision
    with torch.no_grad():
        for x, _ in fit_loader:
            z = augment(frozen_features(model, x.to(device))).double()
            p = torch.sigmoid(z @ centre.double())
            variance = p * (1 - p)
            precision += z.T @ (z * variance[:, None])
    chol = torch.linalg.cholesky(precision)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    noise = torch.randn((n_heads - 1, dim), device=device,
                        dtype=torch.float64, generator=generator)
    # L L^T = precision; L^{-T} xi has covariance precision^{-1}.
    offsets = (torch.linalg.solve_triangular(chol.T, noise.T, upper=True).T
               if n_heads > 1 else noise)
    heads = torch.cat((centre[None].double(),
                       centre[None].double() + posterior_scale * offsets), dim=0)
    heads = heads.to(dtype=centre.dtype).detach()

    losses = torch.zeros(n_heads, device=device, dtype=torch.float64)
    total = 0
    with torch.no_grad():
        for x, labels in val_loader:
            z = augment(frozen_features(model, x.to(device)))
            logits = z @ heads.T
            signs = (2 * labels.to(device).float() - 1)[:, None]
            losses += F.softplus(-signs * logits).double().sum(dim=0)
            total += len(labels)
    if total == 0:
        raise ValueError("Empty validation set")
    losses /= total
    return LaplaceHeads(model, heads, losses, chol)


def train_avcg_l(cvae, train_loader, posterior, selected, *, epochs,
                 lambda_class=1.0, lambda_kl=0.1, lambda_prox=0.1):
    """The attached AVCG-R objective with a Laplace posterior expectation."""
    opt = torch.optim.Adam(cvae.parameters(), lr=0.001, weight_decay=1e-4)
    device = next(cvae.parameters()).device
    for epoch in range(epochs):
        cvae.train()
        total_loss, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            target = 1 - y
            mu, logvar = cvae.encode(x, target)
            kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1).mean()
            cf = cvae.decode(cvae.reparameterize(mu, logvar), target)
            log_prob = posterior.expected_log_prob(cf, target, selected)
            proximity = F.mse_loss(cf, x, reduction="mean")
            loss = -lambda_class * log_prob + lambda_kl * kl + lambda_prox * proximity
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Nonfinite AVCG-L training loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.detach().item() * len(x)
            count += len(x)
        print(f"    CVAE epoch {epoch + 1}/{epochs}: loss={total_loss / count:.5f}",
              flush=True)


def prepare_original_metrics(src, train_ds, train_loader, input_dim,
                             latent_dim, n_classes, device):
    """Recreate the original's 5 CMV models and class AEs in its seed order."""
    # Caller creates the surrogates separately to preserve their original seed
    # schedule. The otherwise unused general AE keeps the class AE RNG order.
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
    samples = {c: train_ds.features[train_ds.labels == c].cpu()
               for c in range(n_classes)}
    return ae_by_class, samples


def seed_frame(raw):
    """
    Aggregate instance-level metrics within each seed.

    The experimental unit for uncertainty reporting is the random seed/run.
    For every Dataset x Epsilon x Method x Seed combination, each metric is
    averaged over the successfully available finite instance-level values.
    """
    if raw.empty:
        return pd.DataFrame()

    metadata = {
        "Dataset", "Seed", "Test Index", "Epsilon",
        "Method", "Status", "Error"
    }

    metrics = [
        col for col in raw.columns
        if col not in metadata and pd.api.types.is_numeric_dtype(raw[col])
    ]

    rows = []

    for (dataset, eps, method, seed), group in raw.groupby(
        ["Dataset", "Epsilon", "Method", "Seed"], sort=True
    ):
        row = {
            "Dataset": dataset,
            "Epsilon": float(eps),
            "Method": method,
            "Seed": int(seed),
            "N Attempted": len(group),
            "N Succeeded": int((group.Status == "ok").sum()),
        }

        row["Success Rate"] = (
            row["N Succeeded"] / row["N Attempted"]
            if row["N Attempted"] else np.nan
        )

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
        .sort_values(["Dataset", "Epsilon", "Method", "Seed"])
        .reset_index(drop=True)
    )


def summary_frame(raw):
    """
    Two-stage aggregation for paper reporting.

    Stage 1:
        Average each metric over evaluated instances WITHIN each seed.

    Stage 2:
        Report the mean and sample standard deviation (ddof=1) of those
        seed-level means across independent seeds.

    Therefore every ``*_std`` column in the returned summary is a
    BETWEEN-SEED standard deviation, not a pooled instance-level standard
    deviation.
    """
    if raw.empty:
        return pd.DataFrame()

    per_seed = seed_frame(raw)
    if per_seed.empty:
        return pd.DataFrame()

    non_metrics = {
        "Dataset", "Epsilon", "Method", "Seed",
        "N Attempted", "N Succeeded", "Success Rate",
    }

    metrics = [
        col for col in per_seed.columns
        if col not in non_metrics
        and not col.endswith("_n_instances")
        and pd.api.types.is_numeric_dtype(per_seed[col])
    ]

    summary = []

    for (dataset, eps, method), group in per_seed.groupby(
        ["Dataset", "Epsilon", "Method"], sort=True
    ):
        row = {
            "Dataset": dataset,
            "Epsilon": float(eps),
            "Method": method,
            "N Seeds": int(group["Seed"].nunique()),
            "N Attempted": int(group["N Attempted"].sum()),
            "N Succeeded": int(group["N Succeeded"].sum()),
        }

        row["Success Rate"] = (
            row["N Succeeded"] / row["N Attempted"]
            if row["N Attempted"] else np.nan
        )

        for col in metrics:
            seed_values = (
                pd.to_numeric(group[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if len(seed_values):
                row[col] = float(seed_values.mean())
                row[col + "_std"] = (
                    float(seed_values.std(ddof=1))
                    if len(seed_values) > 1
                    else 0.0
                )
                row[col + "_n_seeds"] = int(len(seed_values))
            else:
                row[col] = np.nan
                row[col + "_std"] = np.nan
                row[col + "_n_seeds"] = 0

        summary.append(row)

    return (
        pd.DataFrame(summary)
        .sort_values(["Dataset", "Epsilon", "Method"])
        .reset_index(drop=True)
    )


def save_progress(rows, out_dir, dataset):
    """
    Save all three useful result levels:

    1. per-instance observations;
    2. per-seed means;
    3. final mean +/- sample SD across seed-level means.
    """
    raw = pd.DataFrame(rows)

    raw.to_csv(
        out_dir / f"{dataset}_avcg_l_per_instance.csv",
        index=False
    )

    per_seed = seed_frame(raw)
    per_seed.to_csv(
        out_dir / f"{dataset}_avcg_l_per_seed.csv",
        index=False
    )

    summary = summary_frame(raw)
    summary.to_csv(
        out_dir / f"{dataset}_avcg_l_summary.csv",
        index=False
    )



def run_dataset(src, dataset, args, out_dir):
    device = src.DEVICE
    rows = []
    for seed in args.seeds:
        src.set_seed(seed)
        train_ds, val_ds, test_ds, _, _, classes, lower, upper = src.get_dataset(dataset)
        n_classes = len(classes)
        if n_classes != 2:
            raise ValueError("This AVCG-L version supports the binary tabular datasets")
        input_dim = train_ds.features.shape[1]
        latent_dim = max(5, input_dim // 4)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        fit_loader = DataLoader(train_ds, batch_size=256, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        base = src.MCDropoutMLP(input_dim, n_classes).to(device)
        src.train_mc_dropout(base, train_loader, epochs=args.base_epochs)
        base.eval()
        ensemble = src.MCDropoutRashomonSet(base, n_samples=args.n_ensemble)
        ensemble.evaluate_all(val_loader, base_model=base)
        eval_ensemble = src.MCDropoutRashomonSet(
            base, n_samples=max(20, args.n_ensemble // 2))
        eval_ensemble.evaluate_all(val_loader, base_model=base)

        surrogates = []
        for j in range(5):
            src.set_seed(seed * 100 + j + 1)
            model = src.StandardMLP(input_dim, n_classes).to(device)
            src.train_standard_mlp(model, train_loader, epochs=30,
                                   name=f"CMV surrogate {j + 1}/5")
            model.eval()
            surrogates.append(model)
        src.set_seed(seed + 1000)
        ae_by_class, class_samples = prepare_original_metrics(
            src, train_ds, train_loader, input_dim, latent_dim, n_classes, device)

        for parameter in base.parameters():
            parameter.requires_grad_(False)
        posterior = fit_laplace_heads(
            base, fit_loader, val_loader, prior_precision=args.prior_precision,
            posterior_scale=args.posterior_scale, n_heads=args.n_posterior_heads,
            seed=seed + 2000)
        selected = {eps: posterior.indices(eps) for eps in args.epsilons}
        print(f"{dataset} seed={seed}: Laplace heads by epsilon: "
              + ", ".join(f"{eps:g}: {len(selected[eps])}"
                          for eps in args.epsilons), flush=True)

        cvae_by_epsilon = {}
        for eps in args.epsilons:
            # Match initialisation and minibatch order across epsilon settings;
            # only the selected Laplace heads change the training objective.
            src.set_seed(seed + 3000)
            cvae = src.TabularCVAE(input_dim, n_classes, latent_dim).to(device)
            print(f"  Training AVCG-L at epsilon={eps:g}", flush=True)
            train_avcg_l(cvae, train_loader, posterior, selected[eps],
                         epochs=args.cvae_epochs)
            cvae_by_epsilon[eps] = cvae

        base.train()  # same 20-pass correctly-predicted filter as original
        processed = 0
        for test_index, (x, y) in enumerate(test_loader):
            if processed >= args.n_eval:
                break
            x, y = x.to(device), y.to(device)
            base.train()
            with torch.no_grad():
                predicted = torch.stack([base(x) for _ in range(20)]).mean(0).argmax(1).item()
            if predicted != y.item():
                continue
            processed += 1
            target = 1 - predicted
            target_tensor = torch.tensor([target], device=device)
            for eps in args.epsilons:
                cvae = cvae_by_epsilon[eps]
                row = {"Dataset": dataset, "Seed": seed, "Test Index": test_index,
                       "Epsilon": eps, "Method": "AVCG-L", "Status": "failed", "Error": "",
                       "Laplace Set Size": len(selected[eps]),
                       "Laplace Centre Val NLL": posterior.validation_losses[0].item()}
                try:
                    cf, generation_time, diversity = src.generate_cf_amortized(
                        cvae, x, target, lower, upper)
                    if not bool(torch.isfinite(cf).all()):
                        raise FloatingPointError("Nonfinite AVCG-L counterfactual")
                    def regenerate(perturbed, _cvae=cvae, _target=target):
                        return src.generate_cf_amortized(
                            _cvae, perturbed, _target, lower, upper)[:2]
                    metric = src._collect_metrics(
                        cf, target, predicted, x, base, surrogates,
                        ae_by_class, class_samples, generation_time,
                        ensemble=ensemble, eval_ensemble=eval_ensemble,
                        epsilon=eps,
                        generator_callable=regenerate if not args.skip_ric else None,
                        diversity=diversity)
                    row.update(metric)
                    row["Laplace Validity Ratio"] = posterior.target_vote_fraction(
                        cf, target_tensor, selected[eps])
                    row["L2 Distance"] = torch.linalg.norm(cf - x).item()
                    row["Status"] = "ok"
                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    row["Error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
        save_progress(rows, out_dir, dataset)
        print(f"{dataset} seed={seed}: evaluated {processed} correctly predicted inputs",
              flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tabular-source", default="main.py",
                   help="Original AVCG tabular .py or fenced Markdown source")
    p.add_argument("--out", default="avcg_l_results")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--epsilons", default=",".join(map(str, EPSILONS)))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--n-ensemble", type=int, default=50)
    p.add_argument("--base-epochs", type=int, default=50)
    p.add_argument("--cvae-epochs", type=int, default=50)
    p.add_argument("--n-posterior-heads", type=int, default=50,
                   help="Total Laplace heads, including the central model")
    p.add_argument("--prior-precision", type=float, default=1.0,
                   help="Isotropic precision added to sum-NLL Hessian")
    p.add_argument("--posterior-scale", type=float, default=1.0,
                   help="Scale of sampled offsets from the centre")
    p.add_argument("--skip-ric", action="store_true")
    args = p.parse_args()
    args.datasets = tuple(v.strip() for v in args.datasets.split(",") if v.strip())
    args.epsilons = tuple(float(v) for v in args.epsilons.split(","))
    args.seeds = tuple(int(v) for v in args.seeds.split(","))
    if len(set(args.epsilons)) != len(args.epsilons) or any(e < 0 for e in args.epsilons):
        p.error("Epsilons must be distinct and nonnegative")
    if not args.datasets or any(d not in DATASETS for d in args.datasets):
        p.error(f"Datasets must be chosen from {', '.join(DATASETS)}")
    if not args.seeds or min(args.n_eval, args.n_ensemble, args.base_epochs,
                             args.cvae_epochs, args.n_posterior_heads) < 1:
        p.error("Counts and epoch numbers must be positive")
    if args.prior_precision <= 0 or args.posterior_scale <= 0:
        p.error("Prior precision and posterior scale must be positive")
    return args


def main():
    args = parse_args()
    src = load_original(Path(args.tabular_source).expanduser().resolve())
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items()}
    settings.update({
        "laplace_scope": "binary last layer, frozen feature extractor",
        "posterior_center": "original trained classifier head (approximate MAP)",
        "epsilon_definition": "mean validation NLL <= centre NLL + epsilon",
        "metric_ensemble": "original MC dropout Rashomon set",
        "aggregation": (
            "instance metrics are averaged within each seed first; "
            "final metrics are reported as mean +/- sample SD (ddof=1) "
            "across seed-level means"
        ),
    })
    (output / "avcg_l_run_config.json").write_text(json.dumps(settings, indent=2))
    for dataset in args.datasets:
        subdir = output / dataset
        subdir.mkdir(parents=True, exist_ok=True)
        run_dataset(src, dataset, args, subdir)
        print(f"Saved AVCG-L only: {subdir}", flush=True)


if __name__ == "__main__":
    main()
