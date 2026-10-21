#!/usr/bin/env python3
"""
AVCG cross-family transfer experiment (binary tabular datasets).

Purpose
-------
Do counterfactuals constructed against one MC-dropout neural predictor remain
valid for INDEPENDENTLY trained predictors of different model families?

Reuses the ORIGINAL AVCG tabular experiment's:
    get_dataset, MCDropoutMLP, MCDropoutRashomonSet, TabularCVAE,
    train_mc_dropout, generate_cf_amortized, set_seed, DEVICE.

Trains these generators with identical CVAE architecture:
    Point-CVAE  : average target NLL on the fixed original predictor
    AVCG-B      : average target NLL over live MC-dropout predictions
    AVCG-R      : average target NLL over frozen Rashomon-retained masks
    AVCG-CVaR   : mean worst-alpha target NLL over frozen hypotheses
    AVCG-Mixed  : configurable mass on a mixture of dropout masks and
                  independently bootstrapped logistic regression hypotheses

The HELD-OUT evaluation families never train on generated counterfactuals:
    logistic regression, random forest, extra trees, histogram gradient
    boosting, shallow MLP, deep MLP
Each can be fit to independent bootstrap samples for multiple replicates.

Main evaluation:
    target_success : held-out model assigns requested target to x'
    flip_success   : held-out model assigns original label to x,
                     and target label to x'
    factual_agree  : held-out model agrees with the construction predictor at x
    target_prob    : held-out target-class probability at x'
    target_nll     : held-out -log p(target | x')
    target_gain    : target_prob(x') - target_prob(x)
    source_success : original stochastic predictor assigns target to x'
    l2_distance    : ||x' - x||_2

Two evaluation panels are ALWAYS reported:
    all held-out models (unfiltered)
    eligible held-out models selected by PRESET validation thresholds only
No test-set data or counterfactual is used to fit/select an evaluation model.

Results (CSV):
    per_instance.csv               one row per (seed, method, instance, heldout)
    heldout_models.csv              validation/test accuracy, log-loss, eligibility
    mixed_hypotheses.csv            train-time mixture membership and validation NLL
    failures.csv                    counterfactual generation failures, if any
    per_seed_family.csv             within-seed family averages
    across_seed_family.csv          mean +/- sample SD across independent seeds
    per_seed_macro.csv              family-balanced seed-level averages
    across_seed_macro.csv           macro mean +/- sample SD across seeds
    cross_family_success.png        summary figure, where available

IMPORTANT
---------
- This is a NEW controlled experiment, not a reproduction of a result from
  the AVCG manuscript.
- "AVCG-B" here is trained with the LIVE MC dropout objective; "AVCG-R"
  uses frozen masks; CVaR uses frozen selected hypotheses.
- This experiment uses SOURCE get_dataset preprocessing and exact generator
  implementation. Compare generators under the SAME train/val/test split,
  factual instances, and test models.
- Unlike a worst-case proof, observed validity across finite held-out models
  is EMPIRICAL transfer evidence only.
- Results with --quick are a smoke test; use 5 seeds, full epochs, full
  hypotheses and sufficient independent held-out replicas for a paper.
- It does NOT include AVCG-L, since your Laplace script already trains it.
  AVCG-L exports can be added to this evaluation using the same interface.

Examples
--------
python run_avcg_cross_family.py --tabular-source main.py --quick
python run_avcg_cross_family.py --tabular-source main.py \
  --datasets adult_income,breast_cancer,heart_disease,spambase \
  --seeds 11,22,33,44,55 --eval-replicates 3 --base-epochs 50 \
  --cvae-epochs 50 --n-hypotheses 50 --mixed-selection source-rashomon \
  --mixed-logistic-members 10 --mixed-logistic-weight 0.5 --out avcg_cross_family

Requires
--------
numpy, pandas, torch, scikit-learn, matplotlib, plus dependencies of main.py.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import math
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

DATASETS = ("breast_cancer", "heart_disease", "spambase", "adult_income")
SEEDS = (11, 22, 33, 44, 55)
METHODS = ("Point-CVAE", "AVCG-B", "AVCG-R", "AVCG-CVaR", "AVCG-Mixed")


def load_original(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Could not find your AVCG tabular source: {path}")
    content = path.read_text(encoding="utf-8-sig").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    tree = ast.parse(content, filename=str(path))
    # Import definitions without running the source's own experiment main.
    tree.body = [
        n for n in tree.body
        if not (
            isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Name)
            and n.test.left.id == "__name__"
        )
    ]
    module = ModuleType("avcg_cross_family_original")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    required = (
        "get_dataset", "MCDropoutMLP", "MCDropoutRashomonSet",
        "TabularCVAE", "set_seed", "train_mc_dropout",
        "generate_cf_amortized", "DEVICE",
    )
    missing = [s for s in required if not hasattr(module, s)]
    if missing:
        raise ValueError(
            "Your main.py uses a different interface. Missing definitions: "
            + ", ".join(missing)
        )
    return module


def unpack_data(src, name: str):
    data = src.get_dataset(name)
    if not isinstance(data, (tuple, list)) or len(data) != 8:
        raise ValueError("Expected get_dataset(name) to return 8 fields, as in AVCG.")
    train_ds, val_ds, test_ds, _, _, classes, lower, upper = data
    if len(classes) != 2:
        raise ValueError("This experiment currently supports binary datasets only.")
    for ds in (train_ds, val_ds, test_ds):
        if not hasattr(ds, "features") or not hasattr(ds, "labels"):
            raise TypeError("AVCG dataset needs .features and .labels tensors.")
    return train_ds, val_ds, test_ds, lower, upper


def arrays(ds):
    x = ds.features.detach().cpu().numpy().astype(np.float64, copy=True)
    y = ds.labels.detach().cpu().numpy().astype(int, copy=True).reshape(-1)
    if x.ndim != 2 or x.shape[0] != len(y):
        raise ValueError("Input arrays have incorrect shapes.")
    if not np.isfinite(x).all():
        raise ValueError("Dataset contains NaN/Inf; use your main.py preprocessing.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected binary class labels 0 and 1.")
    return x, y


def set_frozen(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def select_rashomon(ensemble, epsilon: float, max_members: int):
    members = list(ensemble.get_rashomon_models(float(epsilon)))
    if not members:
        raise ValueError(
            f"No hypotheses satisfy epsilon={epsilon:g}. "
            "Increase --rashomon-epsilon or inspect validation losses."
        )
    if max_members > 0:
        members = members[:max_members]
    for model in members:
        set_frozen(model)
    return members



class TorchLogisticPredictor(torch.nn.Module):
    """Differentiable view of a FITTED sklearn StandardScaler+LogisticRegression.

    Its parameters are frozen; only gradients with respect to x' are used.
    """
    def __init__(self, fitted_pipeline, device):
        super().__init__()
        scaler = fitted_pipeline.named_steps["standardscaler"]
        lr = fitted_pipeline.named_steps["logisticregression"]
        if list(lr.classes_) != [0, 1]:
            raise ValueError("Expected sklearn logistic classes [0, 1].")
        self.register_buffer("mean", torch.tensor(scaler.mean_,dtype=torch.float32,device=device))
        self.register_buffer("scale", torch.tensor(scaler.scale_,dtype=torch.float32,device=device))
        self.register_buffer("weight", torch.tensor(lr.coef_[0],dtype=torch.float32,device=device))
        self.register_buffer("bias", torch.tensor(float(lr.intercept_[0]),dtype=torch.float32,device=device))

    def forward(self, x):
        logit = ((x-self.mean)/self.scale * self.weight).sum(dim=1)+self.bias
        return torch.stack((torch.zeros_like(logit), logit),dim=1).log_softmax(1)


def train_mixed_hypotheses(x_train, y_train, seed, members, device):
    """Create DIFFERENT logistic regressions for the construction distribution.

    These are NOT the same models as the held-out logistic regressions used for
    evaluation. That prevents evaluating on hypotheses the generator trained on.
    """
    result=[]
    for k in range(members):
        rng=np.random.default_rng(seed*101001+k*11+755)
        inds=rng.choice(len(y_train),len(y_train),replace=True)
        if np.unique(y_train[inds]).size<2:
            raise RuntimeError("Mixed logistic bootstrap yielded one class")
        est=make_pipeline(StandardScaler(), LogisticRegression(max_iter=800))
        est.fit(x_train[inds],y_train[inds])
        net=TorchLogisticPredictor(est,device)
        set_frozen(net)
        result.append(net)
    return result


def validation_target_nll(model, x_val, y_val, device, batch_size=512):
    """Empirical NLL on the same validation partition for every model family.

    This can be used to construct a COMMON empirical Rashomon set over the
    candidate dropout and logistic predictor families, rather than equating
    unrelated family-specific epsilon values.
    """
    model.eval()
    nll_sum = 0.0
    n = len(y_val)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            x = torch.as_tensor(
                x_val[start:start + batch_size], dtype=torch.float32, device=device
            )
            targets = torch.as_tensor(
                y_val[start:start + batch_size], dtype=torch.long, device=device
            )
            nll_sum += float((-log_target_p(model(x), targets)).sum().item())
    return nll_sum / n


def select_mixed_hypotheses(
    all_dropout, source_rashomon, logistic_models, x_val, y_val,
    device, seed, dataset, epsilon, selection,
):
    """Return two family-specific lists and auditable validation NLL records.

    source-rashomon: dropout = selected via original Rashomon estimator;
                     LR = all bootstrap logistic models (which may NOT belong
                     to any cross-family Rashomon set).
    cross-family-rashomon: both families meet common validation NLL threshold
                     min(all candidate validation NLL) + epsilon.

    This finite empirical set is NOT the complete Rashomon set of classifiers.
    """
    if not all_dropout or not logistic_models:
        raise ValueError("Mixed AVCG requires both hypothesis families")
    candidates = (
        [("MC-dropout", i+1, m) for i, m in enumerate(all_dropout)]
        + [("LogisticRegression", i+1, m) for i, m in enumerate(logistic_models)]
    )
    records = []
    scores = []
    for family, index, model in candidates:
        loss = validation_target_nll(model, x_val, y_val, device)
        scores.append(loss)
        records.append({
            "Dataset": dataset, "Seed": seed, "Family": family,
            "Candidate": index, "Validation NLL": loss,
            "Selection Protocol": selection,
        })
    best = float(min(scores))
    threshold = best + epsilon
    accepted_dropout = {id(h) for h in source_rashomon}
    dropout_selected, logistic_selected = [], []
    for (family, index, model), rec, loss in zip(candidates, records, scores):
        if selection == "cross-family-rashomon":
            selected = loss <= threshold + 1e-12
        else:
            selected = (id(model) in accepted_dropout if family == "MC-dropout" else True)
        rec["Common Best NLL"] = best
        rec["Common NLL Threshold"] = threshold
        rec["In Common Empirical Rashomon"] = int(loss <= threshold + 1e-12)
        rec["Selected For AVCG Mixed"] = int(selected)
        if selected and family == "MC-dropout":
            dropout_selected.append(model)
        if selected and family == "LogisticRegression":
            logistic_selected.append(model)
    if not dropout_selected or not logistic_selected:
        raise ValueError(
            f"Mixed selection '{selection}' retained {len(dropout_selected)} "
            f"dropout and {len(logistic_selected)} logistic models at "
            f"epsilon={epsilon:g}. Increase --rashomon-epsilon or "
            "use --mixed-selection source-rashomon."
        )
    print(
        f"    AVCG-Mixed selection: {len(dropout_selected)} MC-dropout + "
        f"{len(logistic_selected)} bootstrap logistic models "
        f"(common min val NLL={best:.4f}; threshold={threshold:.4f})",
        flush=True,
    )
    return dropout_selected, logistic_selected, records


def log_target_p(log_probs, targets):
    """Predictor returns log-softmax in the original AVCG implementation."""
    return log_probs.gather(1, targets.view(-1, 1)).squeeze(1)


def train_generator(
    model, train_ds, *,
    base, members, mixed_members, method, alpha, mixed_logistic_weight, epochs, batch_size,
    lambda_class, lambda_kl, lambda_prox, n_mc, device, seed,
):
    # The generator architecture, initialization and optimizer are identical
    # across treatments. ONLY the predictive objective differs.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(seed + 222)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=loader_gen)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        epoch_losses = []
        for x, y in loader:
            x = x.to(device)
            target = (1 - y.long()).to(device)
            mu, logvar = model.encode(x, target)
            z = model.reparameterize(mu, logvar)
            cf = model.decode(z, target)
            kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())
                  .sum(dim=1).mean())
            prox = F.mse_loss(cf, x)
            if method == "Point-CVAE":
                base.eval()
                nll = -log_target_p(base(cf), target).mean()
            elif method == "AVCG-B":
                # Training mode activates dropout, without updating base weights.
                base.train()
                nll = -torch.stack(
                    [log_target_p(base(cf), target) for _ in range(n_mc)],
                    dim=0,
                ).mean()
            else:
                if not members:
                    raise ValueError("AVCG-R and AVCG-CVaR need frozen hypotheses.")
                losses = torch.stack(
                    [-log_target_p(h(cf), target) for h in members], dim=1
                )  # [batch, hypotheses]
                if method == "AVCG-R":
                    nll = losses.mean()
                elif method == "AVCG-CVaR":
                    tail = max(1, math.ceil(alpha * len(members)))
                    nll = losses.topk(tail, dim=1, largest=True).values.mean()
                elif method == "AVCG-Mixed":
                    if not mixed_members:
                        raise ValueError("AVCG-Mixed needs logistic training hypotheses")
                    logistic_losses = torch.stack(
                        [-log_target_p(h(cf), target) for h in mixed_members], dim=1
                    )
                    # Equal total probability mass for the two model FAMILIES,
                    # regardless of how many individual models each contains.
                    nll = ((1.0 - mixed_logistic_weight) * losses.mean()
                           + mixed_logistic_weight * logistic_losses.mean())
                else:
                    raise ValueError(method)
            loss = lambda_class*nll + lambda_kl*kl + lambda_prox*prox
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"{method}: nonfinite loss at epoch {ep+1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))
        if (ep+1) == epochs or (ep+1) == 1 or (ep+1) % max(1, epochs//5) == 0:
            print(f"    {method} {ep+1:>3}/{epochs}: loss={np.mean(epoch_losses):.5f}",
                  flush=True)
    model.eval()
    base.eval()
    return model


def model_specs(seed: int, mlp_max_iter: int):
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=800, random_state=seed),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, random_state=seed, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=150, min_samples_leaf=2, random_state=seed, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=15, learning_rate=0.08,
            random_state=seed,
        ),
        "MLP_shallow": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32,), alpha=1e-3, random_state=seed,
                early_stopping=True, max_iter=mlp_max_iter, n_iter_no_change=12,
            ),
        ),
        "MLP_deep": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32), alpha=1e-3, random_state=seed,
                early_stopping=True, max_iter=mlp_max_iter, n_iter_no_change=12,
            ),
        ),
    }


def train_heldouts(
    x_train, y_train, x_val, y_val, *,
    dataset, seed, eval_replicates, bootstrap_fraction,
    mlp_max_iter, base_val_acc, base_val_nll, max_acc_gap, max_nll_gap,
):
    fitted, records = [], []
    for j in range(eval_replicates):
        for family, estimator in model_specs(seed*10000 + j*101+7, mlp_max_iter).items():
            # Bootstrap training samples create independently learned hypotheses.
            # Fitting sees neither X_test nor generated counterfactuals.
            rng = np.random.default_rng(seed*100000 + j*1009 + sum(map(ord, family)))
            if bootstrap_fraction > 0:
                n = max(20, round(bootstrap_fraction * len(y_train)))
                indices = rng.choice(len(y_train), size=n, replace=True)
                if len(np.unique(y_train[indices])) < 2:
                    raise RuntimeError("Bootstrapping drew only one class.")
                xt, yt = x_train[indices], y_train[indices]
            else:
                xt, yt = x_train, y_train
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                estimator.fit(xt, yt)
            pred = estimator.predict(x_val)
            prob = estimator.predict_proba(x_val)
            val_acc = float(accuracy_score(y_val, pred))
            val_nll = float(log_loss(y_val, prob, labels=[0, 1]))
            eligible = (
                val_acc >= base_val_acc - max_acc_gap
                and val_nll <= base_val_nll + max_nll_gap
            )
            model_id = f"{family}__rep{j+1}"
            fitted.append((model_id, family, j+1, estimator, bool(eligible)))
            records.append({
                "Dataset": dataset, "Seed": seed, "Model ID": model_id,
                "Family": family, "Replicate": j+1,
                "Val Accuracy": val_acc, "Val NLL": val_nll,
                "Base Val Accuracy": base_val_acc, "Base Val NLL": base_val_nll,
                "Eligible": int(eligible),
                "Train Bootstrap Fraction": bootstrap_fraction,
            })
            print(f"    holdout {model_id:29s} accuracy={val_acc:.3f} "
                  f"NLL={val_nll:.3f} eligible={eligible}", flush=True)
    return fitted, records


@torch.no_grad()
def source_probs(base, x, n_samples: int):
    was_train = base.training
    try:
        base.train()
        p = torch.stack(
            [base(x).exp() for _ in range(n_samples)], dim=0
        ).mean(dim=0)
        return p
    finally:
        base.train(was_train)


def across_seeds(df: pd.DataFrame, group_cols, metrics):
    if df.empty:
        return pd.DataFrame()
    out = []
    for key, g in df.groupby(group_cols, sort=True, dropna=False):
        key = (key,) if not isinstance(key, tuple) else key
        row = dict(zip(group_cols, key))
        row["N seeds"] = int(g["Seed"].nunique())
        for metric in metrics:
            a = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(float)
            row[metric + " mean"] = float(np.mean(a)) if len(a) else np.nan
            row[metric + " SD"] = float(np.std(a, ddof=1)) if len(a)>1 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def summarize(raw: pd.DataFrame, out: Path):
    numeric = [
        "Target Success", "Flip Success", "Factual Agreement",
        "Target Probability", "Target NLL", "Target Gain",
        "Source Success", "L2 Distance", "Generation Time",
    ]
    if raw.empty:
        print("No successful counterfactuals; cannot summarize.")
        return
    seed_family_rows = []
    seed_macro_rows = []
    for panel, data in [
        ("All", raw),
        ("Validation eligible", raw.loc[raw["Eligible"]==1]),
    ]:
        if data.empty:
            print(f"WARNING: panel '{panel}' has no eligible model predictions.")
            continue

        # Each seed/family uses same set of test instances across methods.
        for (dataset, seed, method, family), group in data.groupby(
            ["Dataset","Seed","Method","Family"], sort=True
        ):
            row = {
                "Panel":panel,"Dataset":dataset,"Seed":seed,
                "Method":method,"Family":family,
                "N model-instance evaluations":len(group),
                "N models":group["Model ID"].nunique(),
                "N factual instances":group["Test Index"].nunique(),
            }
            for m in numeric:
                row[m] = float(group[m].mean())
            # Among cases where heldout factual label matches construction label.
            valid = group.loc[group["Factual Agreement"] == 1]
            row["Conditional Target Success"] = (
                float(valid["Target Success"].mean()) if not valid.empty else np.nan
            )
            seed_family_rows.append(row)

    seed_family = pd.DataFrame(seed_family_rows)
    seed_family.to_csv(out/"per_seed_family.csv",index=False)
    metrics = numeric + ["Conditional Target Success"]
    across_seeds(
        seed_family, ["Panel","Dataset","Method","Family"], metrics
    ).to_csv(out/"across_seed_family.csv", index=False)

    # Equal family weights, so a method cannot benefit from more replicas of a
    # particularly easy family. Summary is THEN aggregated across experiment seeds.
    for (panel, dataset, seed, method), g in seed_family.groupby(
        ["Panel","Dataset","Seed","Method"], sort=True
    ):
        row = {
            "Panel":panel, "Dataset":dataset, "Seed":seed, "Method":method,
            "N families":len(g),
        }
        for metric in metrics:
            row[metric] = float(g[metric].mean())
        row["Worst Family Target Success"] = float(g["Target Success"].min())
        seed_macro_rows.append(row)
    seed_macro = pd.DataFrame(seed_macro_rows)
    seed_macro.to_csv(out/"per_seed_macro.csv",index=False)
    across_seeds(
        seed_macro, ["Panel","Dataset","Method"],
        metrics+["Worst Family Target Success"]
    ).to_csv(out/"across_seed_macro.csv",index=False)
    make_plot(seed_family, out)
    print("\nMacro summary (across seeds):")
    cols = ["Panel","Dataset","Method","N seeds",
            "Target Success mean","Target Success SD",
            "Worst Family Target Success mean","Target Probability mean",
            "L2 Distance mean"]
    display = pd.read_csv(out/"across_seed_macro.csv")
    print(display[cols].round(4).to_string(index=False))


def make_plot(seed_family, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plot.")
        return
    sub = seed_family[seed_family["Panel"] == "All"]
    if sub.empty:
        return
    for dataset, g in sub.groupby("Dataset"):
        means = g.groupby(["Family", "Method"])["Target Success"].mean().unstack()
        means = means.reindex(columns=[m for m in METHODS if m in means])
        if means.empty: continue
        ax = means.plot(kind="bar", figsize=(11, 5.5), ylim=(0,1.05))
        ax.set_title(f"{dataset}: transfer to independently trained model families")
        ax.set_ylabel("Held-out target validity (higher is better)")
        ax.set_xlabel("Model family")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Generator", loc="lower right")
        ax.figure.tight_layout()
        ax.figure.savefig(out/f"{dataset}_cross_family_success.png",dpi=170)
        plt.close(ax.figure)


def run_dataset(src, name, args, out):
    raw_rows, model_rows, error_rows, mixed_model_rows = [], [], [], []
    device = src.DEVICE
    for seed in args.seeds:
        print(f"\n=== DATASET {name} | SEED {seed} ===", flush=True)
        src.set_seed(seed)
        train_ds, val_ds, test_ds, lower, upper = unpack_data(src, name)
        X_train,y_train = arrays(train_ds)
        X_val,y_val = arrays(val_ds)
        X_test,y_test = arrays(test_ds)
        in_dim = X_train.shape[1]
        latent_dim = max(5, in_dim//4)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        base = src.MCDropoutMLP(in_dim, 2).to(device)
        src.train_mc_dropout(base, train_loader, epochs=args.base_epochs)
        base.eval()

        with torch.no_grad():
            val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=device)
            p_val = base(val_tensor).exp().detach().cpu().numpy()
            base_val_acc = float(np.mean(p_val.argmax(1)==y_val))
            base_val_nll = float(log_loss(y_val, p_val, labels=[0,1]))

        ens = src.MCDropoutRashomonSet(base, n_samples=args.n_hypotheses)
        ens.evaluate_all(val_loader, base_model=base)
        rash_members = select_rashomon(
            ens, args.rashomon_epsilon, args.max_train_members
        )
        cvar_members = (
            rash_members
            if args.cvar_hypothesis_scope == "rashomon"
            else list(ens.models)
        )
        if args.max_train_members > 0:
            cvar_members = cvar_members[:args.max_train_members]
        for h in cvar_members:
            set_frozen(h)
        set_frozen(base)
        mixed_logistic_candidates = train_mixed_hypotheses(
            X_train, y_train, seed, args.mixed_logistic_members, device
        )
        mixed_dropout, mixed_members, mixture_records = select_mixed_hypotheses(
            list(ens.models), rash_members, mixed_logistic_candidates,
            X_val, y_val, device, seed, name,
            args.rashomon_epsilon, args.mixed_selection,
        )
        mixed_model_rows.extend(mixture_records)

        heldouts, model_record = train_heldouts(
            X_train,y_train,X_val,y_val,
            dataset=name,seed=seed,eval_replicates=args.eval_replicates,
            bootstrap_fraction=args.bootstrap_fraction,
            mlp_max_iter=args.mlp_max_iter,
            base_val_acc=base_val_acc,base_val_nll=base_val_nll,
            max_acc_gap=args.max_val_accuracy_gap,
            max_nll_gap=args.max_val_nll_gap,
        )
        model_rows.extend(model_record)

        # Evaluate heldout test accuracy only for reporting AFTER fit and selection.
        for entry, rec in zip(heldouts, model_record):
            _,_,_,estimator,_ = entry
            rec["Test Accuracy"] = float(accuracy_score(y_test, estimator.predict(X_test)))
            rec["Test NLL"] = float(log_loss(y_test, estimator.predict_proba(X_test),
                                             labels=[0, 1]))

        generators = {}
        for method in METHODS:
            print(f"  Training {method} ...", flush=True)
            src.set_seed(seed+12345)
            generator = src.TabularCVAE(in_dim, 2, latent_dim).to(device)
            generators[method] = train_generator(
                generator, train_ds, base=base,
                members=(
                    mixed_dropout if method == "AVCG-Mixed" else
                    rash_members if method == "AVCG-R" else cvar_members
                ),
                mixed_members=(mixed_members if method=="AVCG-Mixed" else None),
                method=method, alpha=args.alpha,
                mixed_logistic_weight=args.mixed_logistic_weight,
                epochs=args.cvae_epochs,
                batch_size=args.batch_size,
                lambda_class=args.lambda_class,
                lambda_kl=args.lambda_kl,lambda_prox=args.lambda_prox,
                n_mc=args.mc_train_draws,device=device,seed=seed+991,
            )

        # Select one identical set of factuals for all methods (and families).
        selected=[]
        for idx in range(len(test_ds)):
            if len(selected)>=args.n_eval:
                break
            x = test_ds.features[idx:idx+1].to(device)
            y = int(test_ds.labels[idx])
            with torch.no_grad():
                p = source_probs(base, x, args.mc_eval_draws)
            prediction = int(p.argmax(1).item())
            if prediction == y:
                selected.append((idx, x, prediction, 1-prediction))
        if not selected:
            raise RuntimeError(f"{name} seed={seed}: no correctly predicted factuals.")

        for idx,x,source_label,target in selected:
            source_np = X_test[idx:idx+1]
            tensor_target = torch.tensor([target], dtype=torch.long, device=device)
            # Precompute heldout factual predictions and probabilities once.
            factual_by_model = {}
            for model_id,family,rep,estimator,eligible in heldouts:
                factual_by_model[model_id] = (
                    int(estimator.predict(source_np)[0]),
                    float(estimator.predict_proba(source_np)[0,target]),
                )
            for method, generator in generators.items():
                src.set_seed(seed*100000+idx*10+444)
                try:
                    cf, elapsed, _diversity = src.generate_cf_amortized(
                        generator, x, target, lower, upper
                    )
                    cf_np = cf.detach().cpu().numpy().astype(np.float64)
                    if cf_np.shape != source_np.shape or not np.isfinite(cf_np).all():
                        raise ValueError("Invalid counterfactual shape or NaN/Inf.")
                    l2 = float(np.linalg.norm(cf_np - source_np))
                    with torch.no_grad():
                        src_success = int(source_probs(
                            base, cf, args.mc_eval_draws
                        ).argmax(1).item() == target)
                except (ValueError,RuntimeError,FloatingPointError,TypeError) as e:
                    error_rows.append({
                        "Dataset":name, "Seed":seed,"Test Index":idx,
                        "Method":method,"Error":f"{type(e).__name__}: {e}"
                    })
                    continue

                for model_id,family,rep,estimator,eligible in heldouts:
                    prediction_x, target_prob_x = factual_by_model[model_id]
                    cf_pred = int(estimator.predict(cf_np)[0])
                    target_prob_cf = float(estimator.predict_proba(cf_np)[0,target])
                    target_prob_cf = float(np.clip(target_prob_cf,1e-12,1.0))
                    agreed = int(prediction_x == source_label)
                    valid = int(cf_pred == target)
                    raw_rows.append({
                        "Dataset":name,"Seed":seed,"Test Index":idx,
                        "Method":method,"Family":family,
                        "Replicate":rep,"Model ID":model_id,
                        "Eligible":int(eligible),
                        "Ground Truth":int(y_test[idx]),
                        "Source Original Label":source_label,
                        "Requested Target":target,
                        "Heldout Factual Label":prediction_x,
                        "Heldout CF Label":cf_pred,
                        "Factual Agreement":agreed,
                        "Target Success":valid,
                        "Flip Success":int(agreed and valid),
                        "Target Probability":target_prob_cf,
                        "Target NLL":float(-math.log(target_prob_cf)),
                        "Target Gain":target_prob_cf-target_prob_x,
                        "Source Success":src_success,
                        "L2 Distance":l2,
                        "Generation Time":float(elapsed),
                        "Rashomon Epsilon":args.rashomon_epsilon,
                        "CVaR Alpha":args.alpha,
                    })
        # Save partial progress for long-running multi-seed experiments.
        pd.DataFrame(raw_rows).to_csv(out/"per_instance.csv",index=False)
        pd.DataFrame(model_rows).to_csv(out/"heldout_models.csv",index=False)
        pd.DataFrame(mixed_model_rows).to_csv(out/"mixed_hypotheses.csv",index=False)
        pd.DataFrame(error_rows).to_csv(out/"failures.csv",index=False)
        print(f"  {name} seed={seed}: {len(selected)} selected factuals; "
              f"{len(error_rows)} total generation failures",flush=True)
    return raw_rows, model_rows, error_rows, mixed_model_rows


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tabular-source",default="main.py")
    ap.add_argument("--out",default="avcg_cross_family")
    ap.add_argument("--datasets",default=",".join(DATASETS))
    ap.add_argument("--seeds",default=",".join(map(str,SEEDS)))
    ap.add_argument("--n-eval",type=int,default=100)
    ap.add_argument("--n-hypotheses",type=int,default=50)
    ap.add_argument("--max-train-members",type=int,default=0,
                    help="0 = all selected; e.g. 6 for smoke testing only.")
    ap.add_argument("--rashomon-epsilon",type=float,default=0.8)
    ap.add_argument("--cvar-hypothesis-scope",choices=["full","rashomon"],default="full")
    ap.add_argument("--alpha",type=float,default=0.2)
    ap.add_argument("--mixed-logistic-members",type=int,default=10)
    ap.add_argument("--mixed-logistic-weight",type=float,default=0.5,
                    help="Total probability mass on logistic family in AVCG-Mixed; 0.5 means 50/50.")
    ap.add_argument("--mixed-selection",choices=["source-rashomon", "cross-family-rashomon"],
                    default="source-rashomon",
                    help="How to select training hypotheses for AVCG-Mixed.")
    ap.add_argument("--base-epochs",type=int,default=50)
    ap.add_argument("--cvae-epochs",type=int,default=50)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--mc-train-draws",type=int,default=10)
    ap.add_argument("--mc-eval-draws",type=int,default=20)
    ap.add_argument("--eval-replicates",type=int,default=3)
    ap.add_argument("--bootstrap-fraction",type=float,default=1.0,
                    help="1.0 bootstrap with replacement; 0 disables.")
    ap.add_argument("--mlp-max-iter",type=int,default=150)
    ap.add_argument("--max-val-accuracy-gap",type=float,default=0.05)
    ap.add_argument("--max-val-nll-gap",type=float,default=0.25)
    ap.add_argument("--lambda-class",type=float,default=1.0)
    ap.add_argument("--lambda-kl",type=float,default=0.1)
    ap.add_argument("--lambda-prox",type=float,default=0.1)
    ap.add_argument("--quick",action="store_true",
                    help="Small smoke test, NOT publication-scale.")
    args=ap.parse_args()
    args.datasets=tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    args.seeds=tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    unknown=set(args.datasets)-set(DATASETS)
    if unknown:
        ap.error(f"Unknown datasets: {sorted(unknown)}")
    if not 0.0 < args.alpha <= 1.0:
        ap.error("--alpha must lie in (0,1].")
    if not 0 <= args.bootstrap_fraction <= 1.5:
        ap.error("--bootstrap-fraction must be between 0 and 1.5.")
    if args.mixed_logistic_members<1:
        ap.error("--mixed-logistic-members must be >=1")
    if not 0.0 < args.mixed_logistic_weight < 1.0:
        ap.error("--mixed-logistic-weight must be strictly between 0 and 1")
    if args.quick:
        args.datasets=("breast_cancer",)
        args.seeds=(11,)
        args.n_eval=12
        args.n_hypotheses=10
        args.max_train_members=5
        args.mixed_logistic_members=2
        args.base_epochs=2
        args.cvae_epochs=2
        args.mc_train_draws=2
        args.mc_eval_draws=3
        args.eval_replicates=1
        args.mlp_max_iter=15
        args.batch_size=64
    return args


def main():
    args=parse_args()
    out=Path(args.out).expanduser()
    out.mkdir(parents=True,exist_ok=True)
    src=load_original(Path(args.tabular_source))
    everything=[]
    model_reports=[]
    failures=[]
    mixture_reports=[]
    for dataset in args.datasets:
        data, reports, errors, mixture = run_dataset(src, dataset, args, out)
        mixture_reports.extend(mixture)
        everything.extend(data)
        model_reports.extend(reports)
        failures.extend(errors)
        pd.DataFrame(everything).to_csv(out/"per_instance.csv",index=False)
        pd.DataFrame(model_reports).to_csv(out/"heldout_models.csv",index=False)
        pd.DataFrame(failures).to_csv(out/"failures.csv",index=False)
        pd.DataFrame(mixture_reports).to_csv(out/"mixed_hypotheses.csv",index=False)
    summarize(pd.DataFrame(everything),out)
    print(f"\nFiles written to: {out.resolve()}")


if __name__=="__main__":
    main()
