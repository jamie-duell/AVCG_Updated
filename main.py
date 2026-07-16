import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import random
import time
import copy
from collections import defaultdict
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_bayesian_prediction_stats(model, x, target_class, n_samples=100):
    """MC Dropout forward pass — returns (mean_prob, variance, mean_log_prob)."""
    model.train()
    x          = x.to(DEVICE)
    batch_size = x.shape[0]
    x_rep      = x.repeat(n_samples, 1)
    preds      = model(x_rep).view(n_samples, batch_size, -1)
    log_probs      = preds[:, :, target_class]
    mean_log_prob  = log_probs.mean()
    mean_prob      = torch.exp(log_probs).mean()
    variance       = torch.exp(log_probs).var()
    return mean_prob, variance, mean_log_prob


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.long)
    def __len__(self):          return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]


def get_dataset(name):
    print(f"\n--- Loading '{name}' ---")
    if name == 'adult_income':
        ds = fetch_ucirepo(id=2)
        X = ds.data.features
        y = ds.data.targets.iloc[:, 0].astype(str).str.strip().str.replace('.', '', regex=False)
        y = y.map({'<=50K': 0, '>50K': 1})
        class_names = ['<=50K', '>50K']
    elif name == 'spambase':
        ds = fetch_ucirepo(id=94)
        X = ds.data.features
        y = ds.data.targets.iloc[:, 0]
        class_names = ['Not Spam', 'Spam']
    elif name == 'breast_cancer':
        ds = fetch_ucirepo(id=17) # Breast Cancer Wisconsin (Diagnostic)
        X = ds.data.features
        # Diagnosis: M = malignant, B = benign
        y = ds.data.targets.iloc[:, 0].astype(str).str.upper().map({'M': 1, 'B': 0})
        class_names = ['Benign', 'Malignant']
    elif name == 'heart_disease':
        ds = fetch_ucirepo(id=45) # Heart Disease (Cleveland)
        X = ds.data.features
        # num: 0 = no disease, 1-4 = disease
        y = ds.data.targets.iloc[:, 0].apply(lambda x: 1 if x > 0 else 0)
        class_names = ['No Disease', 'Disease']
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Preprocessing
    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    # Fill NaNs with column median, then 0 for any leftover (e.g. all-NaN columns)
    X = X.fillna(X.median()).fillna(0)
    
    # Ensure y is numeric and non-NaN
    y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15, random_state=42, stratify=y_tv)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    lower = torch.tensor(X_train.min(axis=0), dtype=torch.float32).to(DEVICE)
    upper = torch.tensor(X_train.max(axis=0), dtype=torch.float32).to(DEVICE)

    return (TabularDataset(X_train, y_train),
            TabularDataset(X_val,   y_val),
            TabularDataset(X_test,  y_test),
            scaler, X.columns.tolist(), class_names, lower, upper)


# ---------------------------------------------------------------------------
# 2. Model Definitions
# ---------------------------------------------------------------------------
class MCDropoutMLP(nn.Module):
    """MC Dropout model used as base model and for NERCE training."""
    def __init__(self, input_dim, num_classes, hidden_dim1=64,
                 hidden_dim2=32, dropout_rate=0.2):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim1)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.fc2   = nn.Linear(hidden_dim1, hidden_dim2)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3   = nn.Linear(hidden_dim2, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x));  x = self.drop1(x)
        x = F.relu(self.fc2(x));  x = self.drop2(x)
        return F.log_softmax(self.fc3(x), dim=-1)


class StandardMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes), nn.LogSoftmax(dim=-1)
        )
    def forward(self, x): return self.net(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dim=40):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x): return self.decoder(self.encoder(x))


class TabularCVAE(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim=10, hidden_dim=64):
        super().__init__()
        self.input_dim   = input_dim
        self.num_classes = num_classes
        self.latent_dim  = latent_dim
        self.label_emb   = nn.Embedding(num_classes, num_classes)
        self.enc_fc1  = nn.Linear(input_dim + num_classes, hidden_dim)
        self.enc_mu   = nn.Linear(hidden_dim, latent_dim)
        self.enc_logv = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc1  = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.dec_fc2  = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, y):
        h = F.relu(self.enc_fc1(torch.cat([x, self.label_emb(y)], dim=1)))
        return self.enc_mu(h), self.enc_logv(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def decode(self, z, y):
        h = F.relu(self.dec_fc1(torch.cat([z, self.label_emb(y)], dim=1)))
        return self.dec_fc2(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar


# ---------------------------------------------------------------------------
# 3. Rashomon Ensemble (MC Dropout Frozen Masks)
# ---------------------------------------------------------------------------
class FrozenMaskMLP(nn.Module):
    """A deterministic sub-network sampled from the MC Dropout posterior."""
    def __init__(self, base_model, p=0.2):
        super().__init__()
        # Deep copy to safely freeze weights
        self.base = copy.deepcopy(base_model)
        self.base.eval() # Disable native PyTorch dropout
        
        # Pre-sample binary masks, scaled for inference parity
        self.mask1 = (torch.rand(self.base.fc1.out_features) > p).float() / (1.0 - p)
        self.mask2 = (torch.rand(self.base.fc2.out_features) > p).float() / (1.0 - p)

    def forward(self, x):
        x = F.relu(self.base.fc1(x))
        x = x * self.mask1.to(x.device)
        x = F.relu(self.base.fc2(x))
        x = x * self.mask2.to(x.device)
        return F.log_softmax(self.base.fc3(x), dim=-1)


class MCDropoutRashomonSet:
    """Samples N models from MC Dropout and filters them via the epsilon threshold."""
    def __init__(self, base_model, n_samples=50):
        self.models = [FrozenMaskMLP(base_model).to(DEVICE) for _ in range(n_samples)]
        self.val_losses = np.full(n_samples, np.inf)
        self.base_loss = None

    def evaluate_all(self, val_loader, base_model=None):
        print(f"  Evaluating {len(self.models)} MC Dropout masks on validation set...")
        ce = nn.NLLLoss()
        
        if base_model is not None:
            base_model.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    total += ce(base_model(x), y).item() * len(y)
                    n += len(y)
            self.base_loss = total / n
            print(f"  Base model undropped val loss: {self.base_loss:.4f}")

        for i, model in enumerate(self.models):
            total, n = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y  = x.to(DEVICE), y.to(DEVICE)
                    total += ce(model(x), y).item() * len(y)
                    n     += len(y)
            self.val_losses[i] = total / n
        print(f"  Sampled val losses  min={self.val_losses.min():.4f}  "
              f"max={self.val_losses.max():.4f}  mean={self.val_losses.mean():.4f}")

    def get_rashomon_models(self, epsilon):
        L_star = self.base_loss if self.base_loss is not None else self.val_losses.min()
        in_set = [m for m, l in zip(self.models, self.val_losses)
                  if l <= L_star + epsilon]
        # Fallback to the single best mask if epsilon is extremely tight
        return in_set if in_set else [self.models[int(np.argmin(self.val_losses))]]

    def rashomon_size(self, epsilon):
        L_star = self.base_loss if self.base_loss is not None else self.val_losses.min()
        return int((self.val_losses <= L_star + epsilon).sum())

    def expected_log_prob_rashomon(self, x_prime, target_cf, epsilon):
        """Approximates E_{theta~P_R}[log P_theta] using ONLY accepted masks."""
        members      = self.get_rashomon_models(epsilon)
        log_prob_sum = 0.0
        for m in members:
            lp = m(x_prime).gather(1, target_cf.view(-1, 1)).squeeze()
            log_prob_sum = log_prob_sum + lp
        return (log_prob_sum / len(members)).mean()


# ---------------------------------------------------------------------------
# 4. Training Helpers
# ---------------------------------------------------------------------------
def train_mc_dropout(model, train_loader, epochs, name="MC Dropout"):
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    ce  = nn.NLLLoss()
    print(f"  Training {name}...")
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); ce(model(x), y).backward(); opt.step()


def train_standard_mlp(model, train_loader, epochs, name="MLP"):
    opt = optim.Adam(model.parameters(), lr=0.001)
    ce  = nn.NLLLoss()
    print(f"  Training {name}...")
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); ce(model(x), y).backward(); opt.step()


def train_autoencoder(model, loader, epochs, name="AE"):
    opt = optim.Adam(model.parameters(), lr=0.001)
    print(f"  Training {name}...")
    model.train()
    for _ in range(epochs):
        for x, _ in loader:
            x = x.to(DEVICE)
            opt.zero_grad(); F.mse_loss(model(x), x).backward(); opt.step()


def train_cvae_rashomon(cvae, train_loader, ensemble, epsilon, epochs=50,
                        lambda_class=1.0, lambda_prox=0.1, lambda_kl=0.1,
                        name="CVAE"):
    """Trains CVAE using the Rashomon-restricted ELBO objective."""
    opt = optim.Adam(cvae.parameters(), lr=0.001, weight_decay=1e-4)
    print(f"  Training {name} (eps={epsilon})...")
    for _ in range(epochs):
        cvae.train()
        for x, y in train_loader:
            x, y       = x.to(DEVICE), y.to(DEVICE)
            target_cf  = 1 - y
            mu, logvar = cvae.encode(x, target_cf)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            z          = cvae.reparameterize(mu, logvar)
            x_prime    = cvae.decode(z, target_cf)
            exp_lp     = ensemble.expected_log_prob_rashomon(x_prime, target_cf, epsilon)
            proximity  = F.mse_loss(x_prime, x, reduction='mean')
            loss       = -(lambda_class * exp_lp) + (lambda_kl * kl_div) + (lambda_prox * proximity)
            opt.zero_grad(); loss.backward(); opt.step()


def train_cvae_standard(cvae, train_loader, epochs=50, name="CF-VAE"):
    """Plain CVAE — no Rashomon weighting."""
    opt = optim.Adam(cvae.parameters(), lr=0.001)
    print(f"  Training {name}...")
    for _ in range(epochs):
        cvae.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            recon, mu, logvar = cvae(x, y)
            mse = F.mse_loss(recon, x, reduction='sum')
            kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            (mse + kl).backward()
            opt.step()


def train_cvae_nerce(cvae, train_loader, base_model, epochs=50,
                     lambda_class=1.0, lambda_prox=0.1, lambda_kl=0.1,
                     name="MM-CF-CVAE"):
    """
    Trains the CVAE using MC Dropout to approximate E_{theta~P(theta|D)}.
    Original NERCE formulation — no epsilon restriction on the posterior.
    """
    opt = optim.Adam(cvae.parameters(), lr=0.001, weight_decay=1e-4)
    print(f"  Training {name}...")
    for _ in range(epochs):
        cvae.train()
        for x, y in train_loader:
            x, y       = x.to(DEVICE), y.to(DEVICE)
            target_cf  = 1 - y
            mu, logvar = cvae.encode(x, target_cf)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            z          = cvae.reparameterize(mu, logvar)
            x_prime    = cvae.decode(z, target_cf)
            base_model.train()
            M          = 10
            exp_lp_sum = 0.0
            for _ in range(M):
                # Ensure we get different dropout masks
                base_model.train()
                lp = base_model(x_prime).gather(1, target_cf.view(-1, 1)).squeeze()
                exp_lp_sum = exp_lp_sum + lp
            exp_lp = (exp_lp_sum / M).mean()
            proximity = F.mse_loss(x_prime, x, reduction='mean')
            loss      = -(lambda_class * exp_lp) + (lambda_kl * kl_div) + (lambda_prox * proximity)
            opt.zero_grad(); loss.backward(); opt.step()


# ---------------------------------------------------------------------------
# 5. Counterfactual Generation
# ---------------------------------------------------------------------------
def pairwise_diversity(candidates):
    """Calculates average pairwise L2 distance between candidate CFs."""
    if len(candidates) < 2:
        return 0.0
    dists = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            dists.append(torch.linalg.norm(candidates[i] - candidates[j]).item())
    return float(np.mean(dists))


def generate_cf_amortized(cvae, x_original, target_class, lower, upper, n_samples=5):
    """Single forward pass through trained CVAE; returns best CF, time, and diversity."""
    cvae.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        y_t        = torch.tensor([target_class], device=DEVICE)
        mu, logvar = cvae.encode(x_original, y_t)
        candidates = [cvae.decode(cvae.reparameterize(mu, logvar), y_t).clamp(lower, upper)
                      for _ in range(n_samples)]
        dists   = [torch.linalg.norm(c - x_original).item() for c in candidates]
        x_prime = candidates[int(np.argmin(dists))]
        div     = pairwise_diversity(candidates)
    return x_prime, time.perf_counter() - t0, div


def generate_cf_paper_bnn(model, x_original, target_class, lower, upper):
    """Per-instance gradient descent using MC Dropout."""
    t0      = time.perf_counter()
    x_prime = x_original.clone().detach().requires_grad_(True)
    opt     = optim.Adam([x_prime], lr=0.1)
    for _ in range(1001): # Reduced steps for faster evaluation
        opt.zero_grad()
        model.train()
        _, _, mean_lp = get_bayesian_prediction_stats(model, x_prime, target_class)
        loss = -mean_lp + 0.001 * torch.linalg.norm(x_prime - x_original)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x_prime.clamp_(lower, upper)
        model.train()
        with torch.no_grad():
            outs = torch.stack([model(x_prime) for _ in range(20)]).mean(0)
            if outs.max(1)[1].item() == target_class:
                break
    return x_prime.detach(), time.perf_counter() - t0


def generate_cf_greedy(model, x_original, target_class, lower, upper):
    """Feature-wise greedy gradient baseline."""
    t0               = time.perf_counter()
    x_prime          = x_original.clone().detach()
    altered_features = torch.zeros_like(x_prime, device=DEVICE)
    for _ in range(1000): # Reduced steps
        x_prime.requires_grad_(True)
        model.train()
        x_rep    = x_prime.repeat(100, 1)
        logits   = model(x_rep)
        tgt_rep  = torch.tensor([target_class] * 100, device=DEVICE)
        avg_loss = F.nll_loss(logits, tgt_rep, reduction='none').mean()
        avg_loss.backward()
        grad    = x_prime.grad.clone().detach()
        x_prime = x_prime.detach()
        with torch.no_grad():
            avg_probs = torch.exp(logits).mean(dim=0)
            conf, pred = torch.max(avg_probs, dim=0)
            if pred.item() == target_class and conf.item() >= 0.90:
                break
            grad_flat    = grad.view(-1)
            altered_flat = altered_features.view(-1)
            grad_flat[altered_flat >= 5] = 0.0
            if grad_flat.abs().max() == 0:
                break
            max_idx = torch.argmax(grad_flat.abs())
            x_prime.view(-1)[max_idx] -= 0.1 * grad_flat[max_idx].sign()
            altered_features.view(-1)[max_idx] += 1
            x_prime.clamp_(lower, upper)
    return x_prime.detach(), time.perf_counter() - t0


def generate_cf_quce(model, ae, x_original, target_class, lower, upper):
    """Latent-space gradient optimisation through a general autoencoder."""
    t0 = time.perf_counter()
    ae.eval()
    with torch.no_grad():
        z0 = ae.encoder(x_original)
    z_prime = z0.clone().detach().requires_grad_(True)
    opt     = optim.Adam([z_prime], lr=0.1)
    for _ in range(1001): # Reduced steps
        opt.zero_grad()
        x_prime   = ae.decoder(z_prime)
        _, _, mlp = get_bayesian_prediction_stats(model, x_prime, target_class)
        recon_l   = F.mse_loss(ae(x_prime), x_prime)
        prox_l    = F.mse_loss(x_prime, x_original)
        (0.6 * (-mlp) + 0.2 * recon_l + 0.2 * prox_l).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            if model(x_prime).max(1)[1].item() == target_class:
                break
    final = ae.decoder(z_prime).detach().clamp(lower, upper)
    return final, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------
def validity(cf, target_class, model, n_mc=20):
    model.train()
    with torch.no_grad():
        outs = torch.stack([model(cf) for _ in range(n_mc)]).mean(0)
    return int(outs.max(1)[1].item() == target_class)


def l2_distance(cf, x_orig):
    return torch.linalg.norm(cf - x_orig).item()


def im1_score(cf, target_class, original_class, ae_by_class):
    ae_t = ae_by_class.get(target_class)
    ae_o = ae_by_class.get(original_class)
    if not ae_t or not ae_o:
        return np.inf
    ae_t.eval(); ae_o.eval()
    with torch.no_grad():
        err_t = torch.linalg.norm(cf - ae_t(cf)).item() ** 2
        err_o = torch.linalg.norm(cf - ae_o(cf)).item() ** 2
    return err_t / (err_o + 1e-8)


def lof_score(cf, lof_model):
    return -lof_model.score_samples(
        cf.detach().cpu().numpy().reshape(1, -1)).mean()


def implausibility(cf, in_class_samples):
    cf_cpu = cf.detach().cpu()
    if cf_cpu.dim() == 1:
        cf_cpu = cf_cpu.unsqueeze(0)
    return torch.cdist(cf_cpu, in_class_samples.cpu()).mean().item()


def cross_model_validity(cf, target_class, surrogates):
    """Returns fraction of surrogate ensemble that predicts the target class."""
    votes = 0
    for s in surrogates:
        s.eval()
        with torch.no_grad():
            if s(cf).max(1)[1].item() == target_class:
                votes += 1
    return votes / len(surrogates)


def validity_after_training(cf, target_class, eval_ensemble, epsilon):
    """
    Evaluates validity against a set of 'freshly trained' models (eval_ensemble)
    that are also within the Rashomon set (L <= L* + epsilon).
    """
    members = eval_ensemble.get_rashomon_models(epsilon)
    votes = 0
    for m in members:
        m.eval()
        with torch.no_grad():
            votes += int(m(cf).max(1)[1].item() == target_class)
    return votes / len(members)


def robustness_ne(cf, target_class, model, noise=0.01, n_mc=20, n_noise=20):
    model.train()
    with torch.no_grad():
        base = torch.exp(torch.stack([model(cf) for _ in range(n_mc)]).mean(0))
    diffs = []
    for _ in range(n_noise):
        noisy = cf + torch.randn_like(cf) * noise
        with torch.no_grad():
            pert = torch.exp(torch.stack([model(noisy) for _ in range(n_mc)]).mean(0))
        diffs.append(torch.linalg.norm(pert - base).item() ** 2)
    return float(np.mean(diffs))


def robustness_ic(x_original, cf, generator_callable, noise_level=0.01):
    x_pert = x_original + (torch.rand_like(x_original) - 0.5) * noise_level
    cf_pert, _ = generator_callable(x_pert)
    if cf_pert is None or not torch.all(torch.isfinite(cf_pert)):
        return np.nan
    l2_cf   = torch.linalg.norm(cf_pert - cf).item() ** 2
    l2_orig = torch.linalg.norm(cf - x_original).item() ** 2
    return l2_cf / (l2_orig + 1e-6)


def rashomon_validity_ratio(cf, target_class, ensemble, epsilon):
    members = ensemble.get_rashomon_models(epsilon)
    votes = 0
    for m in members:
        m.eval()
        with torch.no_grad():
            votes += int(m(cf).max(1)[1].item() == target_class)
    return votes / len(members)


def _collect_metrics(cf, target_class, pred_label, x_original,
                     base_model, surrogates,
                     ae_by_class, in_class_samples,
                     exec_time, ensemble=None, eval_ensemble=None, epsilon=None,
                     generator_callable=None, diversity=0.0):
    m = {}
    m["Validity"]               = validity(cf, target_class, base_model)
    m["IM1"]                    = im1_score(cf, target_class, pred_label, ae_by_class)
    m["Implausibility"]         = implausibility(cf, in_class_samples[target_class])
    m["Cross-Model Validity"]   = cross_model_validity(cf, target_class, surrogates)
    m["Diversity"]              = diversity
    
    if eval_ensemble is not None and epsilon is not None:
        m["Validity After Training"] = validity_after_training(cf, target_class, eval_ensemble, epsilon)
    else:
        m["Validity After Training"] = np.nan

    for _nl in [0.01, 0.001]:
        m[f"NE Robustness (s={_nl})"] = robustness_ne(cf, target_class, base_model, noise=_nl)
    m["Inference Time (s)"]     = exec_time
    if generator_callable is not None:
        for _nl in [0.01, 0.001]:
            m[f"R_IC (s={_nl})"] = robustness_ic(x_original, cf, generator_callable, noise_level=_nl)
    else:
        for _nl in [0.01, 0.001]:
            m[f"R_IC (s={_nl})"] = np.nan
    if ensemble is not None and epsilon is not None:
        m["Rashomon Validity Ratio"] = rashomon_validity_ratio(
            cf, target_class, ensemble, epsilon)
    else:
        m["Rashomon Validity Ratio"] = np.nan
    return m


# ---------------------------------------------------------------------------
# 7. Epsilon Sweep Experiment
# ---------------------------------------------------------------------------
def run_epsilon_experiment(dataset_name='credit',
                           epsilons=(0.05, 0.1, 0.2, 0.4, 0.8),
                           seeds=(11, 22, 33),
                           n_ensemble=50,
                           n_samples_eval=50,
                           cvae_epochs=50,
                           ensemble_epochs=50):
    results = {eps: defaultdict(lambda: defaultdict(list))
               for eps in epsilons}

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'='*60}\nDATASET: {dataset_name.upper()} | SEED {seed}\n{'='*60}")

        (train_ds, val_ds, test_ds,
         scaler, feat_names, class_names,
         lower, upper) = get_dataset(dataset_name)

        INPUT_DIM   = train_ds.features.shape[1]
        NUM_CLASSES = len(class_names)
        LATENT_DIM  = max(5, INPUT_DIM // 4)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=1,   shuffle=False)

        print("\n[1/7] MC Dropout base model...")
        base_model = MCDropoutMLP(INPUT_DIM, NUM_CLASSES).to(DEVICE)
        train_mc_dropout(base_model, train_loader, epochs=50)

        print("\n[2/7] Generating Restricted Rashomon Sets...")
        # Main ensemble for training CVAE expectation
        ensemble = MCDropoutRashomonSet(base_model, n_samples=n_ensemble)
        ensemble.evaluate_all(val_loader, base_model=base_model)
        
        # Second, freshly sampled ensemble for VAT metric evaluation
        # This approximates 'freshly trained models of similar performance'
        eval_ensemble = MCDropoutRashomonSet(base_model, n_samples=max(20, n_ensemble // 2))
        eval_ensemble.evaluate_all(val_loader, base_model=base_model)

        print("\n[3/7] Surrogate MLP Ensemble (CMV)...")
        surrogates = []
        for i in range(5):
            # Use distinct seeds for each surrogate to ensure maximum diversity and true independence
            surr_seed = seed * 100 + i + 1
            set_seed(surr_seed)
            s = StandardMLP(INPUT_DIM, NUM_CLASSES).to(DEVICE)
            train_standard_mlp(s, train_loader, epochs=30, name=f"CMV surrogate {i+1}/5")
            surrogates.append(s)
        
        # Restore a deterministic state for the rest of the experiment (AE, CVAEs, Evaluation)
        # to ensure they remain consistent regardless of the surrogate ensemble size.
        set_seed(seed + 1000) 

        print("\n[4/7] Autoencoders...")
        ae_general = AutoEncoder(INPUT_DIM, LATENT_DIM).to(DEVICE)
        train_autoencoder(ae_general, train_loader, epochs=40, name="General AE")
        ae_by_class = {}
        for c in range(NUM_CLASSES):
            idxs     = [i for i, lbl in enumerate(train_ds.labels) if lbl == c]
            if not idxs: continue
            c_loader = DataLoader(Subset(train_ds, idxs), batch_size=32, shuffle=True)
            ae_c     = AutoEncoder(INPUT_DIM, LATENT_DIM).to(DEVICE)
            train_autoencoder(ae_c, c_loader, epochs=40, name=f"AE class {c}")
            ae_by_class[c] = ae_c

        train_feats  = train_ds.features.cpu()
        train_labels = train_ds.labels.cpu()
        in_class_samp = {c: train_feats[train_labels == c] for c in range(NUM_CLASSES)}

        print("\n[5/7] CF-VAE baseline...")
        cvae_std = TabularCVAE(INPUT_DIM, NUM_CLASSES, LATENT_DIM).to(DEVICE)
        train_cvae_standard(cvae_std, train_loader, epochs=cvae_epochs)

        print("\n[6/7] MM-CF-CVAE baseline...")
        cvae_nerce = TabularCVAE(INPUT_DIM, NUM_CLASSES, LATENT_DIM).to(DEVICE)
        train_cvae_nerce(cvae_nerce, train_loader, base_model, epochs=cvae_epochs)

        print("\n[7/7] Rashomon CVAEs — one per epsilon...")
        cvae_by_eps = {}
        for eps in epsilons:
            cvae_r = TabularCVAE(INPUT_DIM, NUM_CLASSES, LATENT_DIM).to(DEVICE)
            train_cvae_rashomon(cvae_r, train_loader, ensemble, eps,
                                epochs=cvae_epochs,
                                name=f"Rashomon CVAE (eps={eps})")
            cvae_by_eps[eps] = cvae_r

        print("\n--- Evaluating ---")
        base_model.train()
        processed = 0

        for x, y in test_loader:
            if processed >= n_samples_eval:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                outs       = torch.stack([base_model(x) for _ in range(20)]).mean(0)
                pred_label = outs.max(1)[1].item()
            if pred_label != y.item():
                continue
            processed += 1
            target_cf = 1 - pred_label

            cf_std,   t_std,   div_std   = generate_cf_amortized(cvae_std,   x, target_cf, lower, upper)
            cf_nerce, t_nerce, div_nerce = generate_cf_amortized(cvae_nerce, x, target_cf, lower, upper)
            cf_bnn,   t_bnn   = generate_cf_paper_bnn(base_model,  x, target_cf, lower, upper)
            cf_quce,  t_quce  = generate_cf_quce(base_model, ae_general, x, target_cf, lower, upper)
            cf_grdy,  t_grdy  = generate_cf_greedy(base_model,    x, target_cf, lower, upper)

            gen_std   = lambda xp: generate_cf_amortized(cvae_std,   xp, target_cf, lower, upper)[:2]
            gen_nerce = lambda xp: generate_cf_amortized(cvae_nerce, xp, target_cf, lower, upper)[:2]
            gen_bnn   = lambda xp: generate_cf_paper_bnn(base_model,  xp, target_cf, lower, upper)
            gen_quce  = lambda xp: generate_cf_quce(base_model, ae_general, xp, target_cf, lower, upper)
            gen_grdy  = lambda xp: generate_cf_greedy(base_model,    xp, target_cf, lower, upper)

            baseline_cfs = [
                ("CF-VAE",        cf_std,   t_std,   gen_std,   div_std),
                ("MM-CF-CVAE",    cf_nerce, t_nerce, gen_nerce, div_nerce),
                ("Batten et al.", cf_bnn,   t_bnn,   gen_bnn,   0.0),
                ("QUCE",          cf_quce,  t_quce,  gen_quce,  0.0),
                ("Schut et al.",  cf_grdy,  t_grdy,  gen_grdy,  0.0),
            ]

            for eps in epsilons:
                for mname, cf, et, gen_fn, div in baseline_cfs:
                    if cf is None or not torch.all(torch.isfinite(cf)): continue
                    m = _collect_metrics(cf, target_cf, pred_label, x, base_model, surrogates,
                                       ae_by_class, in_class_samp, et, 
                                       ensemble=ensemble, eval_ensemble=eval_ensemble, epsilon=eps,
                                       generator_callable=gen_fn, diversity=div)
                    for k, v in m.items(): results[eps][mname][k].append(v)

                _cvae_r  = cvae_by_eps[eps]
                gen_rash = lambda xp, _c=_cvae_r: generate_cf_amortized(_c, xp, target_cf, lower, upper)[:2]
                cf_r, t_r, div_r = generate_cf_amortized(_cvae_r, x, target_cf, lower, upper)
                if cf_r is not None and torch.all(torch.isfinite(cf_r)):
                    rname = f"Rashomon CVAE (eps={eps})"
                    m = _collect_metrics(cf_r, target_cf, pred_label, x, base_model, surrogates,
                                       ae_by_class, in_class_samp, t_r, 
                                       ensemble=ensemble, eval_ensemble=eval_ensemble, epsilon=eps,
                                       generator_callable=gen_rash, diversity=div_r)
                    for k, v in m.items(): results[eps][rname][k].append(v)

        print(f"    Evaluated on {processed} instances.")

    summary_rows = []
    for eps in epsilons:
        for method, metric_dict in results[eps].items():
            row = {"Epsilon": eps, "Method": method}
            for metric, vals in metric_dict.items():
                finite = [v for v in vals if np.isfinite(v)]
                row[metric]          = np.mean(finite) if finite else np.nan
                row[f"{metric}_std"] = np.std(finite)  if finite else np.nan
            summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------
METRICS_TO_PLOT = [
    ("Validity",                 "higher is better", True),
    ("IM1",                      "lower is better",  False),
    ("Implausibility",           "lower is better",  False),
    ("Diversity",                "higher is better", True),
    ("Cross-Model Validity",     "higher is better", True),
    ("NE Robustness (s=0.01)",   "lower is better",  False),
    ("NE Robustness (s=0.001)",  "lower is better",  False),
    ("R_IC (s=0.01)",            "lower is better",  False),
    ("R_IC (s=0.001)",           "lower is better",  False),
    ("Rashomon Validity Ratio",  "higher is better", True),
    ("Inference Time (s)",       "lower is better",  False),
]

BASELINE_STYLES = {
    "CF-VAE":     dict(color="#5e81ac", linestyle="--"),
    "MM-CF-CVAE": dict(color="#a3be8c", linestyle="--"),
    "Batten et al.":     dict(color="#ebcb8b", linestyle=":"),
    "QUCE":            dict(color="#d08770", linestyle=":"),
    "Schut et al.":          dict(color="#b48ead", linestyle="-."),
}


def plot_epsilon_sweep(df, dataset_name, out_dir="."):
    epsilons    = sorted(df["Epsilon"].unique())
    all_methods = df["Method"].unique().tolist()
    rash_meths  = sorted([m for m in all_methods if "Rashomon CVAE" in m],
                         key=lambda m: float(m.split("eps=")[1].rstrip(")")))
    base_meths  = [m for m in all_methods if "Rashomon CVAE" not in m]

    ncols = 3
    nrows = (len(METRICS_TO_PLOT) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), facecolor="white")
    fig.suptitle(f"Rashomon Sweep - {dataset_name.replace('_', ' ').title()}",
                 fontsize=18, fontweight="bold", color="black", y=1.01)

    for ax in axes.flat:
        ax.set_facecolor("white"); ax.tick_params(colors="black")
        for s in ax.spines.values(): s.set_edgecolor("black")

    cmap = plt.get_cmap("viridis")
    eps_color = {e: cmap(i / max(len(epsilons) - 1, 1)) for i, e in enumerate(sorted(epsilons))}

    for idx, (metric, label, _) in enumerate(METRICS_TO_PLOT):
        ax = axes.flat[idx]
        ax.set_title(metric, color="black", fontsize=11, pad=6)
        ax.set_xlabel("eps", color="black", fontsize=10); ax.set_ylabel(label, color="black", fontsize=9)

        rash_pts = []
        for meth in rash_meths:
            eps_val = float(meth.split("eps=")[1].rstrip(")"))
            sub = df[df["Method"] == meth]
            if sub.empty or metric not in sub.columns: continue
            yval, ystd = sub[metric].values[0], sub[f"{metric}_std"].values[0]
            rash_pts.append((eps_val, yval, ystd))
            ax.errorbar([eps_val], [yval], yerr=[[ystd]] if np.isfinite(ystd) and ystd > 0 else None,
                        fmt="o", color=eps_color[eps_val], capsize=3, markersize=6)
        if rash_pts:
            xs, ys, _ = zip(*sorted(rash_pts))
            ax.plot(xs, ys, color="#1f77b4", linewidth=1.5)

        # Baselines omitted from sweep plots as per request
        # for bm in base_meths:
        #    ...

        ax.grid(True, color="lightgray", linestyle=":", linewidth=0.7)

    for ax in axes.flat[len(METRICS_TO_PLOT):]: ax.set_visible(False)

    handles = [Line2D([0], [0], color="#1f77b4", linewidth=2, marker="o")]
    labels  = ["Rashomon CVAE (sweep)"]
    # Baseline handles omitted from sweep legend

    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, framealpha=0.9,
               facecolor="white", edgecolor="black", labelcolor="black", fontsize=9, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    path = os.path.join(out_dir, f"epsilon_sweep_{dataset_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_summary_comparison_table(df, dataset_name, out_dir="."):
    """
    Creates a summary table comparing specific Rashomon CVAE epsilon points (0.0, 0.5, 0.8)
    against all baselines.
    """
    target_eps = [0.0, 0.5, 0.8]
    all_methods = df["Method"].unique()
    base_methods = [m for m in all_methods if "Rashomon CVAE" not in m]
    
    # Extract rows for comparison
    comparison_rows = []
    
    # Add specific Rashomon CVAE versions
    for eps in target_eps:
        rname = f"Rashomon CVAE (eps={eps})"
        row = df[(df["Method"] == rname) & (df["Epsilon"] == eps)]
        if not row.empty:
            comparison_rows.append(row.iloc[0])
    
    # Baselines (use epsilon=first_eps row which is identical for baselines)
    first_eps = df["Epsilon"].unique()[0]
    for bm in base_methods:
        row = df[(df["Method"] == bm) & (df["Epsilon"] == first_eps)]
        if not row.empty:
            comparison_rows.append(row.iloc[0])
            
    if not comparison_rows:
        return
        
    comp_df = pd.DataFrame(comparison_rows)
    
    # Clean and format
    cols_to_keep = ["Method"]
    for m, _, _ in METRICS_TO_PLOT:
        if m in comp_df.columns:
            cols_to_keep.append(m)
        if f"{m}_std" in comp_df.columns:
            cols_to_keep.append(f"{m}_std")
    
    comp_df = comp_df[cols_to_keep]
    
    path = os.path.join(out_dir, f"summary_vs_baselines_{dataset_name}.csv")
    comp_df.to_csv(path, index=False)
    
    # Create a pretty-printed version with mean ± std
    display_df = comp_df.copy()
    for m, _, _ in METRICS_TO_PLOT:
        if m in display_df.columns and f"{m}_std" in display_df.columns:
            display_df[m] = display_df.apply(
                lambda row: f"{row[m]:.4f} ± {row[f'{m}_std']:.4f}", axis=1
            )
    
    # Drop the _std columns for the text display
    cols_for_text = ["Method"] + [m for m, _, _ in METRICS_TO_PLOT if m in display_df.columns]
    display_df = display_df[cols_for_text]

    txt_path = os.path.join(out_dir, f"summary_vs_baselines_{dataset_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"RASHOMON CVAE (eps=0, 0.5, 0.8) VS BASELINES - {dataset_name.upper()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(display_df.to_string(index=False))
    print(f"Summary comparison saved to {txt_path}")


def save_results_table(df, dataset_name, out_dir="."):
    csv_path = os.path.join(out_dir, f"results_{dataset_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    save_summary_comparison_table(df, dataset_name, out_dir)


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATASETS     = ["breast_cancer", "heart_disease", "spambase", "adult_income"]
    EPSILONS     = [0.0, 0.1, 0.2, 0.5, 0.8]
    SEEDS        = [11, 22, 33, 44, 55]
    N_ENSEMBLE   = 50
    N_EVAL       = 100
    OUT_DIR      = "./rashomon_results"

    os.makedirs(OUT_DIR, exist_ok=True)

    for dataset in DATASETS:
        dataset_out_dir = os.path.join(OUT_DIR, dataset)
        os.makedirs(dataset_out_dir, exist_ok=True)
        
        try:
            results_df = run_epsilon_experiment(
                dataset_name    = dataset,
                epsilons        = EPSILONS,
                seeds           = SEEDS,
                n_ensemble      = N_ENSEMBLE,
                n_samples_eval  = N_EVAL,
            )
            save_results_table(results_df, dataset, dataset_out_dir)
            plot_epsilon_sweep(results_df, dataset, dataset_out_dir)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")

    print("\nAll done. Results in", OUT_DIR)
