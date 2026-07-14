import pandas as pd
import numpy as np
import os
import glob

def generate_summaries():
    results_dir = "rashomon_results"
    target_eps = [0.0, 0.5, 0.8]
    
    # Define metrics to include in the summary (same as in main.py)
    metrics = [
        "Validity", "IM1", "Implausibility", "Diversity", 
        "Cross-Model Validity", "Validity After Training", 
        "NE Robustness (s=0.01)", "NE Robustness (s=0.001)", 
        "R_IC (s=0.01)", "R_IC (s=0.001)", 
        "Rashomon Validity Ratio", "Inference Time (s)"
    ]

    # Find all results_*.csv files
    csv_files = glob.glob(os.path.join(results_dir, "**", "results_*.csv"), recursive=True)
    
    if not csv_files:
        print("No results_*.csv files found.")
        return

    for csv_path in csv_files:
        dataset_dir = os.path.dirname(csv_path)
        dataset_name = os.path.basename(dataset_dir)
        
        print(f"Processing {dataset_name}...")
        
        df = pd.read_csv(csv_path)
        
        # Identify methods
        all_methods = df["Method"].unique()
        rash_methods = [m for m in all_methods if "Rashomon CVAE" in m]
        base_methods = [m for m in all_methods if "Rashomon CVAE" not in m]
        
        comparison_rows = []
        
        # We will create entries for each target epsilon for BOTH Rashomon and Baselines
        for target in target_eps:
            # 1. Handle Rashomon CVAE (Dynamic: find nearest trained model)
            if rash_methods:
                rash_df = df[df["Method"].str.contains("Rashomon CVAE", na=False)]
                available_eps = rash_df["Epsilon"].unique()
                nearest_eps = available_eps[np.argmin(np.abs(available_eps - target))]
                
                row = df[(df["Method"] == f"Rashomon CVAE (eps={nearest_eps})") & (abs(df["Epsilon"] - nearest_eps) < 1e-6)]
                if not row.empty:
                    row_copy = row.iloc[0].copy()
                    row_copy["Method"] = f"Rashomon CVAE (eps={target})"
                    comparison_rows.append(row_copy)

            # 2. Handle Baselines (Static CF, but measured against this specific epsilon's Rashomon Set)
            for bm in base_methods:
                # Baselines in the main result file are stored in rows corresponding to the epsilon 
                # they were evaluated against during the sweep.
                # We pick the row where the baseline was evaluated against our target epsilon.
                available_eps_for_bm = df[df["Method"] == bm]["Epsilon"].unique()
                if len(available_eps_for_bm) > 0:
                    nearest_eps = available_eps_for_bm[np.argmin(np.abs(available_eps_for_bm - target))]
                    row = df[(df["Method"] == bm) & (abs(df["Epsilon"] - nearest_eps) < 1e-6)]
                    if not row.empty:
                        row_copy = row.iloc[0].copy()
                        row_copy["Method"] = f"{bm} (eval eps={target})"
                        comparison_rows.append(row_copy)
        
        if not comparison_rows:
            print(f"  No matching rows found for {dataset_name}.")
            continue
            
        comp_df = pd.DataFrame(comparison_rows)
        
        # Keep only desired columns
        cols_to_keep = ["Method"] + [m for m in metrics if m in comp_df.columns]
        comp_df = comp_df[cols_to_keep]
        
        # Save CSV
        out_csv = os.path.join(dataset_dir, f"summary_vs_baselines_{dataset_name}.csv")
        comp_df.to_csv(out_csv, index=False)
        
        # Save TXT (readable table)
        out_txt = os.path.join(dataset_dir, f"summary_vs_baselines_{dataset_name}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"RASHOMON CVAE (eps=0, 0.5, 0.8) VS BASELINES - {dataset_name.upper()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(comp_df.to_string(index=False))
        
        print(f"  Saved summary to {out_csv} and {out_txt}")

if __name__ == "__main__":
    generate_summaries()
