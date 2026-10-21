"""Reference implementations of BETARCE, ROAR, and ElliCE.

BETARCE: Stepka, Stefanowski, Lango (KDD 2025 / arXiv:2408.04842v4)
ROAR: Upadhyay, Joshi, Lakkaraju (NeurIPS 2021)
ElliCE: Turbal, Voitsitska, Semenova (NeurIPS 2025 / arXiv:2602.07674v1)

These implementations expose the algorithms in a compact, reusable form. They are
not copies of the authors' repositories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Dict, Any

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import expit

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # BETARCE itself does not require torch
    torch = None
    F = None


Array = np.ndarray
PredictFn = Callable[[Any, Array], Array]


def default_predict(model: Any, X: Array) -> Array:
    """Return class labels for sklearn-like or PyTorch models."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]

    if hasattr(model, "predict"):
        y = np.asarray(model.predict(X))
        return y.reshape(-1)

    if torch is None:
        raise TypeError("Model has no .predict() and PyTorch is unavailable.")

    with torch.no_grad():
        xt = torch.as_tensor(X, dtype=torch.float32)
        out = model(xt)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 1 or out.shape[-1] == 1:
            z = out.reshape(-1)
            # Treat values already in [0, 1] as probabilities, otherwise logits.
            if torch.all((z >= 0) & (z <= 1)):
                p = z
            else:
                p = torch.sigmoid(z)
            return (p >= 0.5).long().cpu().numpy()
        return out.argmax(dim=-1).cpu().numpy()


def beta_robustness_lower_bound(
    successes: Array | int,
    k: int,
    alpha: float,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
) -> Array:
    """Lower (1-alpha) quantile of the Beta posterior used by BETARCE."""
    s = np.asarray(successes)
    a_post = prior_a + s
    b_post = prior_b + (k - s)
    return beta_distribution.ppf(1.0 - alpha, a_post, b_post)


def verify_betarce_robustness(
    X: Array,
    desired_class: int,
    admissible_models: Sequence[Any],
    delta: float,
    alpha: float,
    predict_fn: PredictFn = default_predict,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
) -> Tuple[Array, Array, Array]:
    """Algorithm 1: Bootstrap verification of (delta, alpha)-robustness.

    Returns
    -------
    robust : bool array, shape (n_candidates,)
    lower_bounds : float array, Beta posterior lower bounds
    successes : int array, number of admissible models preserving desired class
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    k = len(admissible_models)
    if k == 0:
        raise ValueError("admissible_models must contain at least one sampled model.")

    successes = np.zeros(X.shape[0], dtype=np.int64)
    for model in admissible_models:
        pred = np.asarray(predict_fn(model, X)).reshape(-1)
        successes += (pred == desired_class)

    lower = beta_robustness_lower_bound(
        successes, k=k, alpha=alpha, prior_a=prior_a, prior_b=prior_b
    )
    return lower >= delta, lower, successes


def sample_uniform_annulus(
    center: Array,
    r_inner: float,
    r_outer: float,
    n: int,
    rng: np.random.Generator,
    mutable_mask: Optional[Array] = None,
) -> Array:
    """Uniformly sample n points by volume in a d-dimensional annulus.

    Sampling is performed only in mutable dimensions; immutable coordinates stay
    equal to the center.
    """
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    if not (0.0 <= r_inner < r_outer):
        raise ValueError("Require 0 <= r_inner < r_outer.")

    if mutable_mask is None:
        mutable_mask = np.ones(center.size, dtype=bool)
    else:
        mutable_mask = np.asarray(mutable_mask, dtype=bool).reshape(-1)
        if mutable_mask.size != center.size:
            raise ValueError("mutable_mask has wrong dimensionality.")

    idx = np.flatnonzero(mutable_mask)
    d = len(idx)
    if d == 0:
        return np.repeat(center[None, :], n, axis=0)

    direction = rng.normal(size=(n, d))
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    direction /= norms

    # Uniform by volume between radii r_inner and r_outer.
    u = rng.random(n)
    radius = (r_inner ** d + u * (r_outer ** d - r_inner ** d)) ** (1.0 / d)

    X = np.repeat(center[None, :], n, axis=0)
    X[:, idx] += direction * radius[:, None]
    return X


def _project_numpy_candidates(
    X: Array,
    x_reference: Array,
    mutable_mask: Optional[Array],
    feature_min: Optional[Array],
    feature_max: Optional[Array],
) -> Array:
    X = np.asarray(X, dtype=np.float32).copy()
    x_reference = np.asarray(x_reference, dtype=np.float32).reshape(-1)

    if feature_min is not None:
        X = np.maximum(X, np.asarray(feature_min, dtype=np.float32).reshape(1, -1))
    if feature_max is not None:
        X = np.minimum(X, np.asarray(feature_max, dtype=np.float32).reshape(1, -1))

    if mutable_mask is not None:
        m = np.asarray(mutable_mask, dtype=bool).reshape(-1)
        X[:, ~m] = x_reference[~m]
    return X


def betarce(
    x_orig: Array,
    base_model: Any,
    base_cf_fn: Callable[[Any, Array, int], Array],
    admissible_models: Sequence[Any],
    *,
    delta: float = 0.9,
    alpha: float = 0.9,
    eta: float = 0.1,
    n: int = 1000,
    predict_fn: PredictFn = default_predict,
    mutable_mask: Optional[Array] = None,
    feature_min: Optional[Array] = None,
    feature_max: Optional[Array] = None,
    distance: str = "l2",
    max_warmup_steps: int = 50,
    max_search_steps: int = 500,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Algorithm 2: BETARCE post-hoc robustification.

    Parameters
    ----------
    base_cf_fn:
        Callable with signature base_cf_fn(base_model, x_orig, y_orig) -> x_cf.
        This can wrap DiCE, GrowingSpheres, Wachter, etc.
    admissible_models:
        A pre-sampled finite approximation to the admissible model space M_M.
        Its length is k in Algorithm 1.
    """
    if not (0.0 < delta < 1.0 and 0.0 < alpha < 1.0):
        raise ValueError("delta and alpha must lie in (0, 1).")
    if eta <= 0 or n <= 0:
        raise ValueError("eta and n must be positive.")

    rng = np.random.default_rng(random_state)
    x_orig = np.asarray(x_orig, dtype=np.float32).reshape(-1)
    y_orig = int(np.asarray(predict_fn(base_model, x_orig[None, :])).reshape(-1)[0])

    x_cf = np.asarray(base_cf_fn(base_model, x_orig.copy(), y_orig), dtype=np.float32).reshape(-1)
    y_cf = int(np.asarray(predict_fn(base_model, x_cf[None, :])).reshape(-1)[0])
    if y_cf == y_orig:
        raise ValueError("base_cf_fn did not return a valid counterfactual.")

    def dist_to_base(X: Array) -> Array:
        D = X - x_cf[None, :]
        if distance.lower() == "l1":
            return np.abs(D).sum(axis=1)
        if distance.lower() == "l2":
            return np.linalg.norm(D, axis=1)
        raise ValueError("distance must be 'l1' or 'l2'.")

    def valid_and_robust(X: Array):
        pred_base = np.asarray(predict_fn(base_model, X)).reshape(-1)
        valid = pred_base == y_cf
        robust, lower, successes = verify_betarce_robustness(
            X,
            desired_class=y_cf,
            admissible_models=admissible_models,
            delta=delta,
            alpha=alpha,
            predict_fn=predict_fn,
        )
        return valid & robust, lower, successes

    # If the base CFE already satisfies the constraint, it is the minimum-distance
    # solution to the BETARCE post-hoc objective (distance 0).
    base_mask, base_lb, base_s = valid_and_robust(x_cf[None, :])
    if bool(base_mask[0]):
        return {
            "x_cf": x_cf.copy(),
            "x_base": x_cf.copy(),
            "y_orig": y_orig,
            "y_cf": y_cf,
            "beta_lower_bound": float(base_lb[0]),
            "successes": int(base_s[0]),
            "k": len(admissible_models),
            "distance_to_base": 0.0,
            "eta_final": eta,
            "search_steps": 0,
        }

    eta_current = float(eta)

    # GrowingSpheres warmup: shrink while the ball still contains a valid+robust CFE.
    for _ in range(max_warmup_steps):
        Z = sample_uniform_annulus(x_cf, 0.0, eta_current, n, rng, mutable_mask)
        Z = _project_numpy_candidates(Z, x_orig, mutable_mask, feature_min, feature_max)
        mask, _, _ = valid_and_robust(Z)
        if not np.any(mask):
            break
        eta_current *= 0.5
    else:
        raise RuntimeError("BETARCE warmup did not converge; increase max_warmup_steps.")

    # GrowingSpheres outward search.
    a0, a1 = eta_current, 2.0 * eta_current
    for step in range(1, max_search_steps + 1):
        Z = sample_uniform_annulus(x_cf, a0, a1, n, rng, mutable_mask)
        Z = _project_numpy_candidates(Z, x_orig, mutable_mask, feature_min, feature_max)
        mask, lower, successes = valid_and_robust(Z)
        if np.any(mask):
            idx = np.flatnonzero(mask)
            best_local = idx[np.argmin(dist_to_base(Z[idx]))]
            return {
                "x_cf": Z[best_local].copy(),
                "x_base": x_cf.copy(),
                "y_orig": y_orig,
                "y_cf": y_cf,
                "beta_lower_bound": float(lower[best_local]),
                "successes": int(successes[best_local]),
                "k": len(admissible_models),
                "distance_to_base": float(dist_to_base(Z[best_local : best_local + 1])[0]),
                "eta_final": eta_current,
                "search_steps": step,
            }
        a0 = a1
        a1 = a1 + eta_current

    raise RuntimeError("BETARCE search failed; increase max_search_steps/eta/n.")


# -----------------------------------------------------------------------------
# ROAR
# -----------------------------------------------------------------------------

@dataclass
class ROARConfig:
    lambda_cost: float = 0.1
    outer_lr: float = 0.05
    inner_lr: float = 0.05
    outer_steps: int = 500
    inner_steps: int = 20
    perturbation: str = "linf"  # 'linf', 'l2', or 'box'
    epsilon: float = 0.1
    delta_min: Optional[Array] = None
    delta_max: Optional[Array] = None
    cost: str = "l1"  # 'l1' or 'l2'
    tolerance: float = 1e-7
    patience: int = 20


def _require_torch():
    if torch is None:
        raise ImportError("ROAR implementation requires PyTorch.")


def _project_delta(delta, cfg: ROARConfig):
    with torch.no_grad():
        if cfg.perturbation == "linf":
            delta.clamp_(-cfg.epsilon, cfg.epsilon)
        elif cfg.perturbation == "l2":
            norm = torch.linalg.vector_norm(delta)
            if norm > cfg.epsilon:
                delta.mul_(cfg.epsilon / (norm + 1e-12))
        elif cfg.perturbation == "box":
            if cfg.delta_min is None or cfg.delta_max is None:
                raise ValueError("box perturbation requires delta_min and delta_max.")
            dmin = torch.as_tensor(cfg.delta_min, dtype=delta.dtype, device=delta.device)
            dmax = torch.as_tensor(cfg.delta_max, dtype=delta.dtype, device=delta.device)
            delta.copy_(torch.maximum(torch.minimum(delta, dmax), dmin))
        else:
            raise ValueError("perturbation must be 'linf', 'l2', or 'box'.")
    return delta


def _roar_inner_adversary(x_cf, w, bias, y_target: int, cfg: ROARConfig):
    """Approximate argmax_delta loss(f_{w+delta}(x_cf), y_target) by PGD."""
    x_fixed = x_cf.detach()
    delta = torch.zeros_like(w, requires_grad=True)
    target = torch.tensor(float(y_target), dtype=x_cf.dtype, device=x_cf.device)

    for _ in range(cfg.inner_steps):
        logit = torch.dot(w + delta, x_fixed) + bias
        loss = F.binary_cross_entropy_with_logits(logit, target)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        with torch.no_grad():
            delta.add_(cfg.inner_lr * grad)
            _project_delta(delta, cfg)
        delta.requires_grad_(True)

    return delta.detach()


def roar_linear(
    x: Array,
    w: Array,
    bias: float = 0.0,
    *,
    y_target: int = 1,
    config: Optional[ROARConfig] = None,
    mutable_mask: Optional[Array] = None,
    feature_min: Optional[Array] = None,
    feature_max: Optional[Array] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """ROAR for the paper's linear approximation f_w.

    Solves, approximately:
        min_xcf max_delta  loss(f_{w+delta}(xcf), y_target)
                           + lambda * c(xcf, x)

    using projected gradient ascent for the inner model-shift problem and
    gradient descent for the outer recourse problem.
    """
    _require_torch()
    cfg = config or ROARConfig()

    x0 = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1)
    wt = torch.as_tensor(w, dtype=torch.float32, device=device).reshape(-1)
    if x0.numel() != wt.numel():
        raise ValueError("x and w must have the same dimensionality.")
    bt = torch.tensor(float(bias), dtype=torch.float32, device=device)

    if mutable_mask is None:
        mutable = torch.ones_like(x0, dtype=torch.bool)
    else:
        mutable = torch.as_tensor(mutable_mask, dtype=torch.bool, device=device).reshape(-1)

    fmin = None if feature_min is None else torch.as_tensor(feature_min, dtype=x0.dtype, device=device).reshape(-1)
    fmax = None if feature_max is None else torch.as_tensor(feature_max, dtype=x0.dtype, device=device).reshape(-1)

    x_cf = x0.clone().detach().requires_grad_(True)
    target = torch.tensor(float(y_target), dtype=x0.dtype, device=device)

    history = []
    best_obj = float("inf")
    stale = 0
    final_delta = torch.zeros_like(wt)

    for step in range(cfg.outer_steps):
        delta_hat = _roar_inner_adversary(x_cf, wt, bt, y_target, cfg)
        final_delta = delta_hat

        logit = torch.dot(wt + delta_hat, x_cf) + bt
        pred_loss = F.binary_cross_entropy_with_logits(logit, target)

        if cfg.cost == "l1":
            recourse_cost = torch.abs(x_cf - x0).sum()
        elif cfg.cost == "l2":
            recourse_cost = torch.linalg.vector_norm(x_cf - x0)
        else:
            raise ValueError("cost must be 'l1' or 'l2'.")

        objective = pred_loss + cfg.lambda_cost * recourse_cost
        obj_value = float(objective.detach().cpu())
        history.append(obj_value)

        grad = torch.autograd.grad(objective, x_cf, only_inputs=True)[0]
        with torch.no_grad():
            x_cf -= cfg.outer_lr * grad
            if fmin is not None:
                x_cf.copy_(torch.maximum(x_cf, fmin))
            if fmax is not None:
                x_cf.copy_(torch.minimum(x_cf, fmax))
            x_cf[~mutable] = x0[~mutable]

        x_cf = x_cf.detach().requires_grad_(True)

        if best_obj - obj_value > cfg.tolerance:
            best_obj = obj_value
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    # Recompute final worst-case perturbation at returned recourse.
    final_delta = _roar_inner_adversary(x_cf, wt, bt, y_target, cfg)
    with torch.no_grad():
        original_logit = torch.dot(wt, x_cf) + bt
        worst_logit = torch.dot(wt + final_delta, x_cf) + bt
        p_original = torch.sigmoid(original_logit)
        p_worst = torch.sigmoid(worst_logit)

    return {
        "x_cf": x_cf.detach().cpu().numpy(),
        "delta_hat": final_delta.detach().cpu().numpy(),
        "p_target_original_model": float(p_original.cpu()) if y_target == 1 else float((1 - p_original).cpu()),
        "p_target_worst_shift": float(p_worst.cpu()) if y_target == 1 else float((1 - p_worst).cpu()),
        "objective_history": history,
        "steps": len(history),
    }


def sklearn_logistic_parameters(model: Any) -> Tuple[Array, float]:
    """Extract w,b from a fitted binary sklearn LogisticRegression model."""
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise TypeError("Expected a fitted sklearn-style linear classifier.")
    coef = np.asarray(model.coef_, dtype=np.float32)
    if coef.shape[0] != 1:
        raise ValueError("This helper supports binary logistic regression only.")
    w = coef[0]
    b = float(np.asarray(model.intercept_).reshape(-1)[0])
    return w, b


# -----------------------------------------------------------------------------
# ElliCE: ellipsoid-approximated Rashomon set, binary target class 1
# -----------------------------------------------------------------------------

def ellice_logistic_hessian(
    X_train: Array,
    w: Array,
    bias: float = 0.0,
    *,
    l2_regularization: float = 1e-3,
    include_intercept: bool = True,
) -> Array:
    """Hessian of mean logistic loss + (lambda/2)*||theta||^2.

    X_train must be in the *same feature space and scaling* as w and the
    counterfactuals. With include_intercept=True, theta=[w,b] and the last
    column of the design matrix is one. This regularizes the intercept too,
    matching the displayed ellipsoid H = E.T W E / n + lambda I in the
    paper's Appendix B.4. If your fitted objective excludes the intercept
    from regularization, construct and supply its exact Hessian instead.
    """
    X = np.asarray(X_train, dtype=float)
    weights = np.asarray(w, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != weights.size:
        raise ValueError("X_train must have shape (n, len(w)), n > 0.")
    if l2_regularization <= 0:
        raise ValueError("Use positive regularization for a bounded ellipsoid.")
    E = np.column_stack((X, np.ones(len(X)))) if include_intercept else X
    logits = X @ weights + float(bias)
    p = expit(logits)
    curvature = p * (1 - p)
    return (E.T * curvature) @ E / len(E) + l2_regularization * np.eye(E.shape[1])


def _ellice_context(w: Array, bias: float, hessian: Array) -> Tuple[Array, Array]:
    weights = np.asarray(w, dtype=float).reshape(-1)
    H = np.asarray(hessian, dtype=float)
    d = weights.size
    if H.shape == (d + 1, d + 1):
        theta = np.r_[weights, float(bias)]
    elif H.shape == (d, d):
        # A fixed intercept is excluded from the uncertain parameter vector.
        theta = weights
    else:
        raise ValueError("Hessian must be d x d (fixed bias) or (d+1) x (d+1).")
    if not np.all(np.isfinite(H)) or not np.allclose(H, H.T, atol=1e-9):
        raise ValueError("Hessian must be finite and symmetric.")
    try:
        chol = np.linalg.cholesky(H)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Hessian must be positive definite; add regularization.") from exc
    return theta, chol


def _ellice_scores_and_grad(
    X: Array, w: Array, bias: float, theta: Array, chol: Array, epsilon: float
) -> Tuple[Array, Array]:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    augmented = np.column_stack((X, np.ones(len(X)))) if theta.size == X.shape[1] + 1 else X
    v = solve_triangular(chol, augmented.T, lower=True, check_finite=False)
    norms = np.linalg.norm(v, axis=0)
    scores = X @ w + bias - np.sqrt(2 * epsilon) * norms
    if epsilon == 0:
        return scores, np.broadcast_to(w, X.shape).copy()
    inv_z = solve_triangular(chol.T, v, lower=False, check_finite=False).T
    grad = w[None, :] - np.sqrt(2 * epsilon) * inv_z[:, :X.shape[1]] / np.maximum(norms[:, None], 1e-14)
    return scores, grad


def ellice_worst_case_logit(
    X: Array,
    w: Array,
    hessian: Array,
    epsilon: float,
    *,
    bias: float = 0.0,
) -> Array:
    """Minimum logit over {(theta-theta_hat)^T H (theta-theta_hat)/2 <= epsilon}.

    Returns one score per row; a score >= target_logit verifies robustness to
    every linear model in this *approximating ellipsoid* (up to roundoff).
    """
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative.")
    weights = np.asarray(w, dtype=float).reshape(-1)
    theta, chol = _ellice_context(weights, bias, hessian)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[1] != weights.size:
        raise ValueError("X must have len(w) columns.")
    return _ellice_scores_and_grad(X, weights, bias, theta, chol, epsilon)[0]


def _ellice_bounds(
    x: Array, mutable_mask: Optional[Array],
    feature_min: Optional[Array], feature_max: Optional[Array]
) -> Tuple[Array, Array]:
    d = len(x)
    lower = np.full(d, -np.inf) if feature_min is None else np.broadcast_to(np.asarray(feature_min, dtype=float), (d,)).copy()
    upper = np.full(d, np.inf) if feature_max is None else np.broadcast_to(np.asarray(feature_max, dtype=float), (d,)).copy()
    if np.any(lower > upper):
        raise ValueError("feature_min exceeds feature_max.")
    if mutable_mask is not None:
        mutable = np.asarray(mutable_mask, dtype=bool).reshape(-1)
        if mutable.size != d:
            raise ValueError("mutable_mask has wrong dimensionality.")
        if np.any((x[~mutable] < lower[~mutable]) | (x[~mutable] > upper[~mutable])):
            raise ValueError("An immutable original feature violates its bounds.")
        lower[~mutable] = upper[~mutable] = x[~mutable]
    return lower, upper


def ellice_linear(
    x: Array,
    w: Array,
    hessian: Array,
    epsilon: float,
    *,
    bias: float = 0.0,
    target_logit: float = 0.0,
    mutable_mask: Optional[Array] = None,
    feature_min: Optional[Array] = None,
    feature_max: Optional[Array] = None,
    maxiter: int = 1000,
    feasibility_tol: float = 1e-6,
) -> Dict[str, Any]:
    """Continuous ElliCE: min ||x_cf-x||^2, robust logit >= target_logit.

    Solves the paper's convex formulation (2) via SLSQP, with analytic
    constraint Jacobian and explicit feasibility check. A robust logit of
    zero corresponds to a probability threshold of 0.5 for target class 1.
    For a probability threshold p, set target_logit=log(p/(1-p)).
    Immutable and box constraints keep the program convex; categorical
    one-hot and sparsity constraints require a separate treatment.
    """
    x0 = np.asarray(x, dtype=float).reshape(-1)
    weights = np.asarray(w, dtype=float).reshape(-1)
    if x0.size != weights.size or epsilon < 0:
        raise ValueError("x and w must have equal dimension; epsilon >= 0.")
    theta, chol = _ellice_context(weights, bias, hessian)
    lower, upper = _ellice_bounds(x0, mutable_mask, feature_min, feature_max)
    if not np.all(np.isfinite(x0)):
        raise ValueError("x must be finite.")
    start = np.maximum(lower, np.minimum(upper, x0))

    def constraint(z):
        return float(_ellice_scores_and_grad(z, weights, bias, theta, chol, epsilon)[0][0] - target_logit)

    def constraint_jac(z):
        return _ellice_scores_and_grad(z, weights, bias, theta, chol, epsilon)[1][0]

    result = minimize(
        lambda z: float(np.dot(z - x0, z - x0)),
        start,
        jac=lambda z: 2 * (z - x0),
        bounds=list(zip(lower, upper)),
        constraints=[{"type": "ineq", "fun": constraint, "jac": constraint_jac}],
        method="SLSQP",
        options={"maxiter": maxiter, "ftol": 1e-10},
    )
    cf = np.asarray(result.x, dtype=float)
    score = float(_ellice_scores_and_grad(cf, weights, bias, theta, chol, epsilon)[0][0])
    feasible = bool(score >= target_logit - feasibility_tol and
                    np.all(cf >= lower - feasibility_tol) and
                    np.all(cf <= upper + feasibility_tol))
    if not feasible:
        raise RuntimeError(
            f"ElliCE found no verified feasible solution: {result.message}; "
            f"worst-case logit={score:.6g}, threshold={target_logit:.6g}."
        )
    return {
        "x_cf": cf, "worst_case_logit": score,
        "central_logit": float(cf @ weights + bias),
        "distance_l2": float(np.linalg.norm(cf - x0)),
        "solver_success": bool(result.success), "solver_message": str(result.message),
        "iterations": int(result.nit),
    }


def ellice_data_supported(
    x: Array,
    X_train: Array,
    w: Array,
    hessian: Array,
    epsilon: float,
    *,
    bias: float = 0.0,
    target_logit: float = 0.0,
    mutable_mask: Optional[Array] = None,
    feature_min: Optional[Array] = None,
    feature_max: Optional[Array] = None,
    immutable_atol: float = 1e-8,
) -> Dict[str, Any]:
    """Data-supported ElliCE: nearest feasible robust training example.

    Exact scan over the supplied training rows; a k-d tree is an optional
    speedup, not needed for equivalent nearest-neighbor selection.
    """
    x0 = np.asarray(x, dtype=float).reshape(-1)
    X = np.asarray(X_train, dtype=float)
    if X.ndim != 2 or X.shape[1] != x0.size or X.shape[0] == 0:
        raise ValueError("X_train must have shape (n, len(x)), n > 0.")
    lower, upper = _ellice_bounds(x0, mutable_mask, feature_min, feature_max)
    scores = ellice_worst_case_logit(X, w, hessian, epsilon, bias=bias)
    feasible = (scores >= target_logit) & np.all(
        (X >= lower - immutable_atol) & (X <= upper + immutable_atol), axis=1
    )
    if mutable_mask is not None:
        fixed = ~np.asarray(mutable_mask, dtype=bool).reshape(-1)
        feasible &= np.all(np.isclose(X[:, fixed], x0[fixed], atol=immutable_atol, rtol=0), axis=1)
    indices = np.flatnonzero(feasible)
    if len(indices) == 0:
        raise RuntimeError("No robust, actionable training example meets the threshold.")
    distances = np.linalg.norm(X[indices] - x0, axis=1)
    idx = int(indices[np.argmin(distances)])
    return {
        "x_cf": X[idx].copy(), "training_index": idx,
        "worst_case_logit": float(scores[idx]),
        "central_logit": float(X[idx] @ np.asarray(w).reshape(-1) + bias),
        "distance_l2": float(np.min(distances)),
        "n_feasible": int(len(indices)),
    }
