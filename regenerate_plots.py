import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np

# ---------------------------------------------------------------------------
# 8. Plotting (Copied and Enhanced from main.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 8,
})

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

def plot_epsilon_sweep(df, dataset_name, out_dir="."):
    epsilons    = sorted(df["Epsilon"].unique())
    all_methods = df["Method"].unique().tolist()
    rash_meths  = sorted([m for m in all_methods if "Rashomon CVAE" in m],
                         key=lambda m: float(m.split("eps=")[1].rstrip(")")))

    ncols = 3
    n_metrics = len(METRICS_TO_PLOT)
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig = plt.figure(figsize=(5 * ncols, 4 * nrows), facecolor="white")
    
    # Title - bold and centered
    fig.suptitle(f"Rashomon Sweep - {dataset_name.replace('_', ' ').title()}",
                 fontsize=20, fontweight="bold", color="black", y=1.02)

    # Best approach for centering incomplete last row in Matplotlib:
    # Use a large number of columns in GridSpec (e.g., 6, since 3 columns * 2).
    from matplotlib.gridspec import GridSpec
    lcm_cols = 6 
    gs = GridSpec(nrows, lcm_cols, figure=fig)
    axes = []
    
    for i in range(n_metrics):
        row = i // ncols
        col = i % ncols
        
        # Check how many plots are in this row
        n_in_this_row = ncols
        if row == nrows - 1:
            n_in_this_row = n_metrics % ncols
            if n_in_this_row == 0: n_in_this_row = ncols
        
        if n_in_this_row == ncols:
            # Normal row: each plot takes 2 columns in a 6-col grid
            axes.append(fig.add_subplot(gs[row, col*2 : (col+1)*2]))
        else:
            # Centered row: offset = (total_cols - n_in_row * cols_per_plot) / 2
            # For 2 plots in a 6-col grid: (6 - 2 * 2) // 2 = 1.
            # Plot 1 at [1:3], Plot 2 at [3:5].
            offset = (lcm_cols - n_in_this_row * 2) // 2
            axes.append(fig.add_subplot(gs[row, offset + col*2 : offset + (col+1)*2]))

    cmap = plt.get_cmap("viridis")
    eps_color = {e: cmap(i / max(len(epsilons) - 1, 1)) for i, e in enumerate(sorted(epsilons))}

    for idx, (metric, label, _) in enumerate(METRICS_TO_PLOT):
        ax = axes[idx]
        ax.set_facecolor("#fcfcfc")
        for s in ax.spines.values():
            s.set_edgecolor("#333333")
            s.set_linewidth(1.2)
        
        ax.set_title(metric, color="black", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel(r"$\epsilon$", color="black", fontsize=12, fontweight="bold")
        ax.set_ylabel(label.capitalize(), color="black", fontsize=11)

        rash_pts = []
        for meth in rash_meths:
            eps_val = float(meth.split("eps=")[1].rstrip(")"))
            sub = df[df["Method"] == meth]
            if sub.empty or metric not in sub.columns: continue
            yval, ystd = sub[metric].values[0], sub[f"{metric}_std"].values[0]
            rash_pts.append((eps_val, yval, ystd))
            ax.errorbar([eps_val], [yval], yerr=[[ystd]] if np.isfinite(ystd) and ystd > 0 else None,
                        fmt="o", color=eps_color[eps_val], capsize=4, markersize=8, elinewidth=1.5)
        
        if rash_pts:
            xs, ys, _ = zip(*sorted(rash_pts))
            ax.plot(xs, ys, color="#1f77b4", linewidth=2.5, zorder=1, alpha=0.8)

        ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Legend at the bottom
    handles = [Line2D([0], [0], color="#1f77b4", linewidth=3, marker="o", markersize=8)]
    labels  = ["AVCG-R"]

    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=True, framealpha=1.0,
               facecolor="white", edgecolor="black", labelcolor="black", fontsize=14, 
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    path = os.path.join(out_dir, f"epsilon_sweep_{dataset_name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Research-ready plot saved to {path}")

def main():
    results_dir = "rashomon_results"
    datasets = ["adult_income", "breast_cancer", "heart_disease", "spambase"]
    
    for ds in datasets:
        ds_path = os.path.join(results_dir, ds)
        csv_file = os.path.join(ds_path, f"results_{ds}.csv")
        if os.path.exists(csv_file):
            print(f"Processing {ds}...")
            df = pd.read_csv(csv_file)
            plot_epsilon_sweep(df, ds, out_dir=ds_path)
        else:
            print(f"CSV file not found: {csv_file}")

if __name__ == "__main__":
    main()
