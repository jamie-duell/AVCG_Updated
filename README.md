# AVCG Experiments

This repository contains experiments for evaluating AVCG-style counterfactual generation under MC Dropout and Rashomon-set model uncertainty.


- `main.py`: tabular epsilon-sweep experiments on UCI datasets.
- `main_image.py`: image epsilon-sweep experiments on MNIST and CIFAR-10.
- `run_robust_recourse_baselines.py`: runs BETARCE, ROAR-local, and ElliCE-embedding against the same experiment protocols.

## Setup

Use Python 3.8+ with PyTorch installed. On Windows, this machine resolves Python as `py`, not `python`, so the commands below use `py`.

Install dependencies:

```powershell
py -m pip install torch torchvision numpy pandas matplotlib scikit-learn scipy ucimlrepo
```

Optional dataset-loading check for the UCI tabular datasets:

```powershell
py test_loading.py
```

## Run Tabular AVCG Experiments

Run the full tabular epsilon sweep:

```powershell
py main.py
```

This runs:

- Datasets: `breast_cancer`, `heart_disease`, `spambase`, `adult_income`
- Epsilons: `0.0, 0.1, 0.2, 0.5, 0.8`
- Seeds: `11, 22, 33, 44, 55`
- Ensemble size: `50`
- Evaluation inputs per dataset/seed: `100`

Outputs are written to:

```text
rashomon_results/<dataset>/
```

The default configuration is defined in the `if __name__ == "__main__":` block of `main.py`.

## Run Image AVCG Experiments

Run the full image epsilon sweep:

```powershell
py main_image.py
```

This runs:

- Datasets: `mnist`, `cifar10`
- Epsilons: `0.0, 0.1, 0.2, 0.5, 0.8`
- Seeds: `11, 22, 33, 44, 55`
- Ensemble size: `20`
- Evaluation inputs per dataset/seed: `100`

Outputs are written to:

```text
rashomon_results_image/<dataset>/
```

The default configuration is defined in the `if __name__ == "__main__":` block of `main_image.py`.

## Generate One CIFAR AVCG-L Counterfactual

Run a compact CIFAR-10 AVCG-L example:

```powershell
py generate_cifar_avcg_l_example.py
```

This trains a small CIFAR base CNN, fits a last-layer Laplace posterior, trains a ConvCVAE with the AVCG-L posterior expectation objective, and saves one counterfactual comparison image.

Outputs are written to:

```text
cifar_avcg_l_example/
```

For a faster smoke test:

```powershell
py generate_cifar_avcg_l_example.py --train-subset 512 --val-subset 256 --base-epochs 1 --cvae-epochs 1 --n-heads 8 --refine-steps 25
```

The script tries all non-original CIFAR-10 target classes by default and stops when it finds a target-reaching counterfactual. Use `--target-class {0..9}` to force a specific target.

## Generate Summary Tables

After running the tabular experiments:

```powershell
py generate_summaries.py
```

After running the image experiments:

```powershell
py generate_summaries_image.py
```

These scripts post-process saved CSV outputs and generate comparison tables in the corresponding results directories.

## Run Robust Recourse Baselines

Run BETARCE, ROAR-local, and ElliCE-embedding using the local experiment scripts:

```powershell
py run_robust_recourse_baselines.py
```

By default, this uses:

- `main.py` as the tabular source
- `main_image.py` as the image source
- `robust_recourse_methods.py` for method implementations
- Output directory: `robust_baseline_results/`

To run only tabular experiments:

```powershell
py run_robust_recourse_baselines.py --kind tabular
```

To run only image experiments:

```powershell
py run_robust_recourse_baselines.py --kind image
```

Useful small smoke test:

```powershell
py run_robust_recourse_baselines.py --kind tabular --tabular-datasets breast_cancer --seeds 11 --epsilons 0.0 --n-eval-tabular 1 --epochs-tabular 1 --n-ensemble-tabular 2 --skip-ric
```

Useful options:

- `--tabular-datasets breast_cancer,heart_disease`
- `--image-datasets mnist,cifar10`
- `--epsilons 0.0,0.5,0.8`
- `--seeds 11,22,33`
- `--n-eval-tabular 10`
- `--n-eval-image 10`
- `--skip-ric` to skip R_IC regeneration and reduce runtime

Robust baseline outputs are written to:

```text
robust_baseline_results/<kind>/<dataset>/
```

Each dataset directory contains:

- `<dataset>_new_methods_per_instance.csv`
- `<dataset>_new_methods_summary.csv`

## Metrics

The experiment pipeline reports:

- `Validity`: whether the counterfactual reaches the target class under the base model.
- `IM1`: class-conditional autoencoder reconstruction ratio.
- `Implausibility`: distance to target-class training samples.
- `Diversity`: average pairwise L2 distance between generated candidates.
- `Cross-Model Validity`: target-class agreement across separately trained surrogate models.
- `Validity After Training`: validity under a fresh evaluation Rashomon ensemble.
- `NE Robustness`: output stability under input noise.
- `R_IC`: counterfactual stability under perturbations of the original input.
- `Rashomon Validity Ratio`: fraction of epsilon-restricted Rashomon models that accept the counterfactual.
- `Inference Time (s)`: counterfactual generation time.

## Notes

- Scripts automatically use CUDA when available and otherwise fall back to CPU.
- Full runs can take a long time, especially image experiments and robust baselines with R_IC enabled.
- The baseline runner intentionally imports definitions from `main.py` and `main_image.py` without executing their `__main__` blocks.
- If `python` is not recognized on Windows, use `py` as shown above.
