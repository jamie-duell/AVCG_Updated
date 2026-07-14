# Rashomon Epsilon Sweep Experiment (MC Dropout Version)

This repository contains the implementation and evaluation scripts for the Rashomon Epsilon Sweep experiment. The study focuses on evaluating Counterfactual Generative Models (CVAE) under the Rashomon Effect, specifically using MC Dropout to approximate the model posterior.

## Project Structure

- `main.py`: The primary script for running the experiments. It handles data loading, model training (Base, CVAE, Baselines), counterfactual generation, and metric evaluation.
- `generate_summaries.py`: A post-processing script that aggregates results from the `main.py` output and generates comparative summary tables (CSV and TXT) at specific epsilon thresholds (0.0, 0.5, 0.8).
- `test_loading.py`: A utility script to verify that all datasets can be correctly fetched and preprocessed from the UCI repository.
- `rashomon_results/`: Directory where all experimental results, including plots and tables, are stored, organized by dataset.

## Prerequisites

To run the experiments, you need Python 3.8+ and the following libraries:

```bash
pip install torch numpy pandas matplotlib scikit-learn ucimlrepo
```

## Reproducing Results

Follow these steps to reproduce the experimental results:

### 1. Verify Data Loading (Optional)
Before running the full experiment, you can verify that the datasets are accessible:
```bash
python test_loading.py
```
This script will attempt to fetch and preprocess the Breast Cancer, Heart Disease, Spambase, and Adult Income datasets from the UCI repository.

### 2. Run the Main Experiment
To run the full epsilon sweep across all datasets, execute:
```bash
python main.py
```
**What this script does:**
- Iterates through four UCI datasets: `breast_cancer`, `heart_disease`, `spambase`, and `adult_income`.
- Performs an epsilon sweep for $\epsilon \in \{0.0, 0.1, 0.2, 0.5, 0.8\}$.
- Trains a base MC Dropout model and extracts a diverse Rashomon set of models.
- Trains a **Rashomon CVAE** for each epsilon, specifically optimized against the restricted posterior.
- Evaluates our method against several baselines: **CF-VAE**, **MM-CF-CVAE**, **Batten et al.**, **QUCE**, and **Schut et al.**
- Saves raw results, plots, and a preliminary summary in `rashomon_results/[dataset_name]/`.

### 3. Generate Summary Tables
After the main experiment completes, run the summary generator to produce the final comparison tables used in the analysis:
```bash
python generate_summaries.py
```
This script processes the raw CSV files generated in Step 2 and creates `summary_vs_baselines_[dataset_name].txt` files. These tables compare our method against baselines specifically at $\epsilon \in \{0.0, 0.5, 0.8\}$, providing a clear "stress-test" view of how each method handles expanding model uncertainty.

## Metrics Explained

The evaluation pipeline computes a wide array of metrics to assess counterfactual quality:

- **Validity**: Fraction of CFs that successfully flip the base model's prediction to the target class.
- **IM1**: Interpretable Metric 1; compares reconstruction error between class-specific autoencoders (lower is better).
- **Implausibility**: Average distance to the nearest training samples of the target class (lower is better).
- **Diversity**: Average pairwise L2 distance between multiple generated candidates, measuring user agency.
- **Cross-Model Validity (CMV)**: Consensus validity across an ensemble of 5 independently trained MLPs.
- **Validity After Training (VAT)**: Consensus validity against a fresh, independent Rashomon ensemble of models.
- **Rashomon Validity Ratio**: Fraction of the current epsilon-restricted Rashomon set that agrees on the CF's validity.
- **NE Robustness**: Robustness of the CF's prediction to small input noise.
- **R_IC**: Input Change Robustness; measures how much the generated CF shifts when the original input is slightly perturbed.
- **Inference Time**: Time taken to generate the counterfactual.

## Notes on Configuration
- **Device**: The scripts automatically detect and use CUDA if available, otherwise defaulting to CPU.
- **Hyperparameters**: Key experiment parameters such as the number of ensemble models (`N_ENSEMBLE`), evaluation samples (`N_EVAL`), and seeds (`SEEDS`) are defined in the `if __name__ == "__main__":` block of `main.py`. By default, it runs with 3 seeds and 50 evaluation samples per dataset to balance thoroughness and execution time.
