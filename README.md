# AVCG experiments

This repository evaluates counterfactual generators on Adult Income, Breast Cancer, Heart Disease, and Spambase from UCI. Tables 1–2 combine the main experiment with three companion runs. The mask-retention table uses the same evaluation sweep. Table 4 comes from a separate mixed-family experiment.

## Reproduce the results

### 1. Set up the manuscript branch

From a new PowerShell session, clone the branch containing the experiment scripts and reference results. The commands use the Windows Python launcher `py`. On other platforms, use the `python` executable from your active environment.

```powershell
git clone --branch master_up --single-branch https://github.com/jamie-duell/AVCG_Updated.git
cd AVCG_Updated
py -m pip install torch numpy pandas matplotlib scikit-learn scipy ucimlrepo
py test_loading.py
```

Run the remaining commands from the repository root. The loading check fetches all four UCI datasets and prints a success or error for each. Training uses CUDA when available and otherwise uses CPU. A fresh clone gives reruns their own working directory because the training scripts write to their default result folders.

### 2. Reproduce Tables 1–2 and mask retention

**Using saved results.** This branch includes the seed-level results for the main comparison, AVCG-L, and the robust baselines. The remaining input is the five-tolerance CVaR sweep. From a fresh clone, run the two commands below. If you already have `avcg_cvar_sweep/all_datasets_avcg_cvar_per_seed.csv` and its matching `avcg_cvar_run_config.json` from the manuscript sweep, run only the second command. Building the tables from saved inputs uses only the Python standard library.

```powershell
py run_avcg_cvar.py --alphas 0.2 --out avcg_cvar_sweep
py generate_rvr_tables.py
```

**Retraining from the data.** To reproduce the experiment measurements as well as their aggregation, run the following sequence. Start with the primary comparison. Each companion script then loads the data preparation, base model, and evaluation definitions from `main.py` and fits its own models. The first four commands can run independently. The last command needs all four sets of seed results.

```powershell
py main.py
py run_avcg_l.py
py run_robust_recourse_baselines.py --kind tabular
py run_avcg_cvar.py --alphas 0.2 --out avcg_cvar_sweep
py generate_rvr_tables.py
```

| Producer | Manuscript rows or quantities | Input the table generator reads |
| --- | --- | --- |
| `main.py` | AVCG-B (`MM-CF-CVAE` in the CSV), AVCG-R (`Rashomon CVAE (eps=…)`), Batten et al., QUCE, and Schut et al. The raw main run also includes CF-VAE. | `rashomon_results/<dataset>/seed_level_results_<dataset>.csv` |
| `run_avcg_l.py` | AVCG-L, trained using last-layer Laplace samples and scored on the common dropout-mask protocol | `avcg_l_results/<dataset>/<dataset>_avcg_l_per_seed.csv` |
| `run_robust_recourse_baselines.py` | BetaRCE (`BETARCE`), ROAR (`ROAR-local`), and ElliCE (`ElliCE-embedding`) | `robust_baseline_results/tabular/<dataset>/<dataset>_new_methods_per_seed.csv` |
| `run_avcg_cvar.py` | AVCG-CVaR at α = 0.2, plus the construction and held-out mask counts for each dataset, seed, and evaluation ε | `avcg_cvar_sweep/all_datasets_avcg_cvar_per_seed.csv` and `avcg_cvar_sweep/avcg_cvar_run_config.json` |

The main run uses stratified train, validation, and test splits with split seed 42. It reserves 20% for testing and 15% of the remainder for validation. The five experiment seeds are 11, 22, 33, 44, and 55. For each dataset and seed, the main run evaluates up to 100 correctly classified test inputs with the opposite class as the counterfactual target. AVCG-R trains a generator for each ε. The full-pool CVaR run trains one generator per dataset, seed, and α and scores each generated example at all five tolerances. The archived `avcg_cvar_results/` run uses a single evaluation ε of 0.8. [Main experiment](main.py) · [Laplace run settings](avcg_l_results/avcg_l_run_config.json) · [Archived CVaR settings](avcg_cvar_results/avcg_cvar_run_config.json)

**Check the outputs.** The generator writes `paper_tables/adult_breast.tex` (Table 1), `paper_tables/heart_spambase.tex` (Table 2), and `paper_tables/mask_retention.tex`. The two comparison tables contain 200 method rows (ten methods × four datasets × five tolerances). Filtering the sweep CSV to full-pool α = 0.2 gives 100 dataset–seed–tolerance records. The generator averages their `RVR Raw Mask Count` and `VAT Raw Mask Count` over seeds to produce the 20 construction/held-out pairs. [Table generator](generate_rvr_tables.py)

Using the same saved inputs reproduces the displayed numbers. For a new training run, record the Python and package versions and whether PyTorch used CPU or CUDA. Refitting can change scores and near-threshold mask counts, and inference time depends on the machine. Before combining a new sweep with archived scores, compare its `RVR Raw Mask Count` to the archived `Raw Rashomon Set Size` for each dataset, seed, and ε. The retention table below gives the reference averages for both banks. If the counts differ, use matching-run seed results and mask counts together. [Construction counts by seed](avcg_cross_family/rashomon_set_sizes.csv)

For a worked example, take **Batten et al., Adult Income, ε = 0**. In the main seed CSV, select that method and tolerance. Averaging the five `Rashomon Validity Ratio` values gives the construction RVR mean of **0.699**. For held-out RVR, use `Validity After Training` from the same rows and retain only seeds with a positive `VAT Raw Mask Count` in the matching sweep. The four remaining seed means give **0.689**. Taking the sample SD of each contributing set gives **0.075** and **0.071**, respectively. The same calculation applies to the other methods using their own seed files. [Adult Income seed results](rashomon_results/adult_income/seed_level_results_adult_income.csv)

### 3. Reproduce Table 4

```powershell
py run_avcg_cross_family.py
```

This experiment reuses the data split and generator architecture from `main.py`, then trains fresh generators and independent evaluation models. The three reported training distributions are `masks` (AVCG-R), `linear` (AVCG-Linear), and `masks + linear` (AVCG-Mixed). Within each seed, the script averages the evaluation models’ target-class votes over counterfactuals and models in each family. It then takes the mean and sample SD across seeds. For the L1 distance it counts each generated counterfactual once.

The numeric and formatted CSVs each have 60 rows (four datasets × five tolerances × three training distributions). Both are included in the checkout for readers who want the reported values. A rerun also writes the individual predictions to `avcg_cross_family/per_instance.csv` and the within-seed family averages to `avcg_cross_family/per_seed_family.csv`. [Numeric Table 4](avcg_cross_family/mixed_family_table.csv) · [Formatted Table 4](avcg_cross_family/mixed_family_table_formatted.csv) · [Evaluation model records](avcg_cross_family/heldout_models.csv)

### 4. Reproduce related analyses

| Question | Command | Output |
| --- | --- | --- |
| How many construction masks pass the threshold in each seed? | `py count_rashomon_set_sizes.py` | Rebuilds the base models and construction banks and writes `rashomon_results/rashomon_set_sizes_by_seed.csv` and `rashomon_results/rashomon_set_sizes_summary.csv`. The paired held-out counts come from step 2. |
| How do the primary AVCG-R metrics vary with ε? | `py regenerate_plots.py` | Reads the saved `rashomon_results/<dataset>/results_<dataset>.csv` summaries and redraws `epsilon_sweep_<dataset>.png` alongside them. |
| How does the CVaR objective change with α? | `py run_avcg_cvar.py --out cvar_alpha_sweep` | Trains the default sweep over α = 0.1, 0.2, 0.5, and 1.0 and writes combined CSVs and metric plots under `cvar_alpha_sweep/`. |
| How does ElliCE perform with a tolerance calibrated to its own surrogate loss? | `py run_ellice.py --kind tabular` | Writes the separate calibrated experiment to `ellice_results/tabular/<dataset>/`. The ElliCE rows in Tables 1–2 come from the robust-baseline runner in step 2. |

`main.py` also writes the primary comparison summaries for all five manuscript tolerances. The older `generate_summaries.py` rewrites the comparison summaries using historical targets 0, 0.5, and 0.8 and selects the nearest available evaluated ε. For the manuscript tables, use the seed files and generator in step 2.

For a quick installation check of the robust-baseline runner, this command uses one dataset, seed, tolerance, and test input and writes to a separate directory.

```powershell
py run_robust_recourse_baselines.py --kind tabular --tabular-datasets breast_cancer --seeds 11 --epsilons 0.0 --n-eval-tabular 1 --epochs-tabular 1 --n-ensemble-tabular 2 --skip-ric --out smoke_results
```

## Reading the results

### Construction and held-out masks

For each dataset and seed, the experiment fits one MC-dropout network and draws two banks of frozen dropout masks from its fitted weights. The construction bank has 50 candidates and the independently drawn held-out bank has 25. At each tolerance `ε`, a mask qualifies when its validation negative log-likelihood is at most the **undropped base network’s validation loss + ε**. AVCG-R uses the qualifying construction masks to train its generator.

| Manuscript column | Saved metric | What is averaged for each counterfactual |
| --- | --- | --- |
| RVR or RVR_tr | `Rashomon Validity Ratio` | Fraction of retained construction masks predicting the requested target |
| RVR_ho | `Validity After Training` (VAT) | Fraction of retained held-out masks predicting the requested target, for seeds with at least one qualifying held-out mask |
| CMV | `Cross-Model Validity` | Fraction of five separately trained MLPs predicting the requested target |

The retention table gives the **mean number of qualifying masks** across seeds 11, 22, 33, 44, and 55. Each cell shows construction / held-out counts. These counts describe the evaluation banks shared by the methods. For example, AVCG-CVaR with full-pool training uses all 50 construction candidates to train its generator, then applies the tolerance when scoring RVR.

| Dataset | ε = 0 | ε = 0.01 | ε = 0.05 | ε = 0.10 | ε = 0.50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adult Income | 2.6 / 2.0 | 24.2 / 12.8 | 50.0 / 25.0 | 50.0 / 25.0 | 50.0 / 25.0 |
| Breast Cancer | 15.2 / 7.6 | 37.4 / 20.4 | 50.0 / 24.8 | 50.0 / 25.0 | 50.0 / 25.0 |
| Heart Disease | 11.0 / 7.4 | 19.0 / 12.0 | 43.4 / 22.8 | 49.4 / 25.0 | 50.0 / 25.0 |
| Spambase | 5.0 / 2.2 | 15.6 / 6.4 | 46.0 / 22.0 | 49.2 / 24.4 | 50.0 / 25.0 |

**Adult Income at ε = 0** is the only case in this sweep with an empty bank. The five held-out counts are 0, 2, 2, 2, and 4, giving the reported mean of 2.0. When a bank is empty, the legacy scorer uses its lowest-loss mask as a fallback. For manuscript RVR_ho, seed 11 has an undefined ratio and is excluded. A method with finite evaluations in the other four seeds therefore contributes four seed means. In general, the tables average finite counterfactual scores within each seed and report the mean and sample SD across contributing seeds. A single contributing seed supplies a mean only.

### MLP evaluation in Tables 1–2 and Table 4

| | Tables 1–2 · CMV | Table 4 · MLP |
| --- | --- | --- |
| Evaluators per seed | Five PyTorch MLPs, each with hidden widths (64, 64) | Two scikit-learn MLPs, one with a single hidden layer of width 32 and one of width 64 |
| Training examples | Each MLP uses the full training partition | Each MLP uses its own full-size bootstrap sample of the training partition |
| Fitting | Separate initializations and shuffle seeds, 30 epochs | Separate initializations, early stopping |

Table 4’s **Deep** column uses a third bootstrapped MLP with hidden widths (64, 32). The mixed-family experiment trains its own generators and generates its own counterfactuals. Differences between its MLP scores and CMV can reflect both the evaluator and the generated examples. [Main evaluation](main.py) · [Mixed-family evaluation](run_avcg_cross_family.py) · [Mixed-family results](avcg_cross_family/mixed_family_table.csv)

## Other metrics

- **Validity** measures whether the counterfactual reaches the target class under the base model.
- **IM1** compares reconstruction errors from the target-class and original-class autoencoders.
- **Implausibility** measures mean distance to target-class training samples.
- **Diversity** measures average pairwise L2 distance between generated candidates.
- **NE robustness** measures how predictions change when noise is added to the counterfactual.
- **R_IC** measures how the generated counterfactual changes when the original input is perturbed.
- **Inference time** records counterfactual generation time in seconds.
