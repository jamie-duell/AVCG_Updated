"""Build the manuscript's tabular evaluations and mask retention from saved seed means."""

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_tables"
DATASETS = {
    "adult_income": "Adult Income", "breast_cancer": "Breast Cancer",
    "heart_disease": "Heart Disease", "spambase": "Spambase",
}
EPSILONS = (0.0, 0.01, 0.05, 0.1, 0.5)
SEEDS = (11, 22, 33, 44, 55)
TRAIN = "Rashomon Validity Ratio"
HELD = "Validity After Training"
# Source column, manuscript heading, larger is better.
METRICS = [
    ("Validity", "Val", True), ("IM1", "IM1", False),
    ("Implausibility", "Imp", False), ("Diversity", "Div", True),
    ("Cross-Model Validity", "CMV", True), ("NE Robustness (s=0.01)", "NE", False),
    ("R_IC (s=0.01)", r"R$_{\mathrm{IC}}$", False),
    (TRAIN, r"RVR$_{\rm tr}$", True), (HELD, r"RVR$_{\rm ho}$", True),
    ("Inference Time (s)", "Time (s)", False),
]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_seed_data(dataset, sweep):
    sources = [
        ROOT / "rashomon_results" / dataset / f"seed_level_results_{dataset}.csv",
        ROOT / "avcg_l_results" / dataset / f"{dataset}_avcg_l_per_seed.csv",
        ROOT / "robust_baseline_results/tabular" / dataset / f"{dataset}_new_methods_per_seed.csv",
    ]
    rows = [row for path in sources for row in read_csv(path)]
    counts = {}
    for row in sweep:
        if row["Dataset"] != dataset or float(row["Alpha"]) != 0.2 or row["Hypothesis Scope"] != "full":
            continue
        eps, seed = float(row["Metric Epsilon"]), int(row["Seed"])
        counts[eps, seed] = int(row["RVR Raw Mask Count"]), int(row["VAT Raw Mask Count"])
        rows.append({**row, "Epsilon": eps})
    expected = {(eps, seed) for eps in EPSILONS for seed in SEEDS}
    if counts.keys() != expected:
        raise ValueError(f"Incomplete mask counts for {dataset}")
    indexed = {}
    for row in rows:
        key = (float(row["Epsilon"]), row["Method"], int(row["Seed"]))
        if key in indexed:
            raise ValueError(f"Duplicate seed result for {dataset}: {key}")
        indexed[key] = row
    return indexed, counts


def method_labels(eps):
    return [
        ("Batten et al.", "Batten et al."), ("QUCE", "QUCE"), ("Schut et al.", "Schut et al."),
        ("MM-CF-CVAE", r"\textbf{AVCG-B}"),
        (f"Rashomon CVAE (eps={eps})", rf"\textbf{{AVCG-R ($\epsilon={eps}$)}}"),
        ("AVCG-L", rf"\textbf{{AVCG-L ($\epsilon={eps}$)}}"),
        ("AVCG-CVaR", r"\textbf{AVCG-CVaR ($\alpha=0.2$)}"),
        ("BETARCE", "BetaRCE"), ("ROAR-local", "ROAR"), ("ElliCE-embedding", "ElliCE"),
    ]


def summarize(rows, counts, eps, method, metric):
    values = []
    for seed in SEEDS:
        row = rows[eps, method, seed]
        n_tr, n_ho = counts[eps, seed]
        if (metric == HELD and n_ho == 0) or (metric == TRAIN and n_tr == 0):
            continue  # A vote by the best rejected mask does not define a retained-set ratio.
        value = row[metric]
        if value and math.isfinite(float(value)):
            values.append(float(value))
    mean = f"{statistics.mean(values):.3f}" if values else None
    sd = f"{statistics.stdev(values):.3f}" if len(values) > 1 else None
    return mean, sd, len(values)


def format_cell(cell, best):
    mean, sd, n = cell
    if mean is None:
        return r"\textemdash{}"
    if float(mean) == best:
        mean = rf"\textbf{{{mean}}}"
        if sd is not None:
            sd = rf"\textbf{{{sd}}}"
    text = mean if sd is None else rf"{mean} $\pm$ {sd}"
    return text if n == len(SEEDS) else text + rf"$^{{({n})}}$"


def evaluation_float(dataset, pair, label, continued, rows, counts):
    title = " and ".join(DATASETS[key] for key in pair)
    caption = (
        f"Counterfactual evaluation on {title}" + (" (continued). " if continued else ". ")
        + r"CMV averages target votes from five $(64,64)$ MLPs fitted separately on the full training partition. "
        r"RVR$_{\rm tr}$ and RVR$_{\rm ho}$ average target votes from masks retained at the indicated "
        r"$\epsilon$ in the construction and held-out banks. AVCG-R trains on the retained construction masks. "
        r"AVCG-CVaR ($\alpha=0.2$) uses all 50 construction masks. "
        r"Entries are means and sample SDs of finite seed means. Superscripts indicate fewer than five seeds. "
        r"Single-seed entries give only the mean. An empty retained set gives an undefined ratio. "
        r"A row of dashes denotes no successful generation. NE and R$_{\rm IC}$ use $s=0.01$. "
        r"Bold marks the best displayed means, including ties."
    )
    lines = [r"\begin{tableorg}[!htbp]"]
    if continued:
        lines.append(r"\ContinuedFloat")
    lines += [r"\centering", (r"\caption[]{" if continued else r"\caption{") + caption + "}"]
    if not continued:
        lines.append(rf"\label{{{label}}}")
    lines += [r"\begingroup", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.08}",
              r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l*{10}{c}}", r"\toprule"]
    headers = [rf"\textbf{{{name}}} " + (r"$\uparrow$" if up else r"$\downarrow$")
               for _, name, up in METRICS]
    lines += [" & ".join([r"\textbf{Method}", *headers]) + r" \\", r"\midrule",
              rf"\multicolumn{{11}}{{c}}{{\textit{{{DATASETS[dataset]}}}}} \\"]
    for eps in EPSILONS:
        lines += [r"\midrule", rf"\multicolumn{{11}}{{l}}{{\textbf{{Environment $\epsilon={eps}$}}}} \\",
                  r"\midrule"]
        block = [
            (name, [summarize(rows, counts, eps, method, metric) for metric, _, _ in METRICS])
            for method, name in method_labels(eps)
        ]
        best = []
        for i, (_, _, higher) in enumerate(METRICS):
            means = [float(cells[i][0]) for _, cells in block if cells[i][0] is not None]
            best.append((max if higher else min)(means, default=None))
        for name, cells in block:
            lines.append(" & ".join([name, *(format_cell(c, b) for c, b in zip(cells, best))]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\endgroup", r"\end{tableorg}"]
    return "\n".join(lines) + "\n"


def retention_table(data):
    lines = [
        r"\begin{tableorg}[!htbp]", r"\centering\small",
        r"\caption[Rashomon retention]{Mean retained construction / held-out mask counts across five seeds. "
        r"Each seed has 50 construction candidates and 25 independently drawn held-out candidates, "
        r"all using the same fitted base weights. Each bank retains masks with validation NLL at most "
        r"the undropped base model's validation NLL plus $\epsilon$. A seed with zero retained masks "
        r"contributes zero to the mean count and has an undefined ratio for that bank.}",
        r"\label{m:tab:rho_main}", r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}",
        r"\toprule", "Dataset & " + " & ".join(rf"$\epsilon={eps:.2f}$" for eps in EPSILONS) + r" \\",
        r"\midrule",
    ]
    for dataset, (_, counts) in data.items():
        cells = []
        for eps in EPSILONS:
            train, held = zip(*(counts[eps, seed] for seed in SEEDS))
            cells.append(f"{statistics.mean(train):.1f} / {statistics.mean(held):.1f}")
        lines.append(" & ".join([DATASETS[dataset], *cells]) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular*}", r"\end{tableorg}", ""])


def main():
    sweep_dir = ROOT / "avcg_cvar_sweep"
    config_path = sweep_dir / "avcg_cvar_run_config.json"
    seed_path = sweep_dir / "all_datasets_avcg_cvar_per_seed.csv"
    for path in (config_path, seed_path):
        if not path.is_file():
            raise SystemExit(
                f"Missing {path.relative_to(ROOT)}. To generate the sweep inputs, run "
                "python run_avcg_cvar.py --alphas 0.2 --out avcg_cvar_sweep"
            )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["n_ensemble"] != 50 or config["base_epochs"] != 50:
        raise ValueError("The mask counts must come from the manuscript's 50-mask, 50-epoch protocol.")
    sweep = read_csv(seed_path)
    data = {dataset: load_seed_data(dataset, sweep) for dataset in DATASETS}
    OUT.mkdir(exist_ok=True)
    for pair, label, filename in [
        (("adult_income", "breast_cancer"), "tab:results_adult_breast1", "adult_breast.tex"),
        (("heart_disease", "spambase"), "tab:results_heart_spambase", "heart_spambase.tex"),
    ]:
        floats = [evaluation_float(ds, pair, label, i > 0, *data[ds]) for i, ds in enumerate(pair)]
        (OUT / filename).write_text("\n".join(floats), encoding="utf-8")
    (OUT / "mask_retention.tex").write_text(retention_table(data), encoding="utf-8")


if __name__ == "__main__":
    main()
