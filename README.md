# AVCG experiments

This repository evaluates counterfactual generators on Adult Income, Breast Cancer, Heart Disease, and Spambase from UCI. The main comparison measures how often a generated counterfactual reaches its requested target class under several predictors.

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

## Running the tabular experiments

The commands below use the Windows Python launcher `py`. Install the dependencies and check that the UCI datasets load.

```powershell
py -m pip install torch numpy pandas matplotlib scikit-learn scipy ucimlrepo
py test_loading.py
```

| Experiment | Command | Default results directory |
| --- | --- | --- |
| Main AVCG and baseline comparison | `py main.py` | [rashomon_results](rashomon_results/) |
| Last-layer Laplace (AVCG-L) | `py run_avcg_l.py` | [avcg_l_results](avcg_l_results/) |
| CVaR objective | `py run_avcg_cvar.py` | [avcg_cvar_results](avcg_cvar_results/) |
| BetaRCE, ROAR, and ElliCE | `py run_robust_recourse_baselines.py --kind tabular` | [robust_baseline_results/tabular](robust_baseline_results/tabular/) |
| Mixed-family transfer | `py run_avcg_cross_family.py` | [avcg_cross_family](avcg_cross_family/) |

The main run evaluates up to 100 correctly classified test inputs per dataset and seed at the five tolerances shown above. In each dataset directory, `seed_level_results_<dataset>.csv` contains the seed means used for aggregation and `results_<dataset>.csv` contains the summary. The archived CVaR results in this branch use evaluation ε = 0.8. The manuscript CVaR rows use α = 0.2 and score the full-pool generator at each of the five tabulated tolerances. [Archived CVaR run configuration](avcg_cvar_results/avcg_cvar_run_config.json)

To reproduce the construction-bank retention counts without training counterfactual generators, run `py count_rashomon_set_sizes.py`. The mixed-family run also saves [construction counts by dataset, seed, and tolerance](avcg_cross_family/rashomon_set_sizes.csv).

This command runs a small robust-baseline smoke test.

```powershell
py run_robust_recourse_baselines.py --kind tabular --tabular-datasets breast_cancer --seeds 11 --epsilons 0.0 --n-eval-tabular 1 --epochs-tabular 1 --n-ensemble-tabular 2 --skip-ric --out smoke_results
```

## Other metrics

- **Validity** measures whether the counterfactual reaches the target class under the base model.
- **IM1** compares reconstruction errors from the target-class and original-class autoencoders.
- **Implausibility** measures mean distance to target-class training samples.
- **Diversity** measures average pairwise L2 distance between generated candidates.
- **NE robustness** measures how predictions change when noise is added to the counterfactual.
- **R_IC** measures how the generated counterfactual changes when the original input is perturbed.
- **Inference time** records counterfactual generation time in seconds.
