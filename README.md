# On the Use of Large Language Models for Zero-Shot Risk Matrix Learning

Code, prompts, model outputs and evaluation scripts for the paper
*On the Use of Large Language Models for Zero-Shot Risk Matrix Learning*
(N. A. Rufino, I. P. Matsuno, M. G. de Oliveira, R. M. Marcacini),
accepted at the HBDA-LLM workshop, IEEE BigData 2026.

Given the free-text description of an incident report, an LLM estimates its
severity (1 to 4) and likelihood (1 to 5) and writes a short justification.
The per-report predictions are aggregated into a severity × likelihood risk
matrix. The paper evaluates two compact models (`mistral-nemo`,
`llama-4-scout`) on incident reports from SaferProducts.gov, using
DeepSeek-V3 as a reference for likelihood.

- Paper: *link to be added*
- Release used in the paper: `v1.0`

## Repository layout

| Folder | Content |
|---|---|
| [`LLM-BASED/`](./LLM-BASED/README.md) | Preprocessing, LLM scoring, evaluation, plots and significance tests, plus all model outputs under `results/` |
| [`baselines/`](./baselines/README.md) | Majority-class, keyword-rule and TF-IDF + logistic regression baselines |

## Data

The incident reports come from the public database of the U.S. Consumer
Product Safety Commission (CPSC), SaferProducts.gov
(<https://www.saferproducts.gov>). This repository is not affiliated with or
endorsed by the CPSC.

The raw, interim and processed data (`LLM-BASED/data/`) are **not** included.
To rebuild them, export the full database from the SaferProducts.gov
Public Search page, place `IncidentReports.csv` in
`LLM-BASED/data/raw/saferproducts_gov/` and run the preprocessing steps in
[`LLM-BASED/README.md`](./LLM-BASED/README.md#pipeline). Counts per category
and severity level are in
[`LLM-BASED/docs/saferproducts_dataset.md`](./LLM-BASED/docs/saferproducts_dataset.md).

## Model outputs

`LLM-BASED/results/saferproducts_gov/` has one folder per model run, named
after the model:

| Folder | Content |
|---|---|
| `mistral-nemo/`, `llama-4-scout/` | Evaluated models |
| `deepseek-chat-v3-0324/` | DeepSeek-V3, used as the reference (it receives the ground-truth hints described in the paper) |
| `deepseek-chat-v3-0324_sample/`, `deepseek-chat-v3-0324_sample_hints/` | DeepSeek-V3 on the stratified sample, without and with hints (the check described in the paper) |

Each folder has one `<category>_scored.csv` per product category with the
columns `report_id`, `incident_description`, `victim_severity` (ground truth),
`severity_level`, `occurrence_likelihood`, `justification` and
`scoring_failed`. The tables and figures can be reproduced from these files
without calling any API.

## License

The code is released under the [MIT License](./LICENSE). The incident reports
included in `results/` remain subject to the SaferProducts.gov terms (see
[Data](#data)).

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r LLM-BASED/requirements.txt
pip install -e LLM-BASED
```

## Reproducing the paper's results

Run these from `LLM-BASED/`. `evaluate.py` computes metrics and matrices per
category, and `visualize.py` draws the comparison plots (the figure numbers
are those of the paper).

```bash
cd LLM-BASED

# Evaluation against the ground truth (severity)
python src/evaluation/evaluate.py --source saferproducts_gov --model mistral-nemo
python src/evaluation/evaluate.py --source saferproducts_gov --model llama-4-scout

# Evaluation against the DeepSeek-V3 reference (likelihood and matrix level)
python src/evaluation/evaluate.py --source saferproducts_gov --model mistral-nemo --ref deepseek-chat-v3-0324
python src/evaluation/evaluate.py --source saferproducts_gov --model llama-4-scout --ref deepseek-chat-v3-0324

# Plots
python src/evaluation/visualize.py --source saferproducts_gov --models mistral-nemo llama-4-scout
python src/evaluation/visualize.py --source saferproducts_gov --models mistral-nemo llama-4-scout --ref deepseek-chat-v3-0324
```

| Figure | File (under `LLM-BASED/results/saferproducts_gov/comparisons/`) |
|---|---|
| Fig. 4, F1-macro for severity | `vs_ground_truth/f1_severity_macro.png` |
| Fig. 5, F1-macro for likelihood | `vs_deepseek-chat-v3-0324/f1_likelihood_macro.png` |
| Fig. 6, ordinal error for severity | `vs_ground_truth/ordinal_error_severity.png` |
| Fig. 7, ordinal error for likelihood | `vs_deepseek-chat-v3-0324/ordinal_error_likelihood.png` |
| Fig. 9, cell-wise MAE | `vs_deepseek-chat-v3-0324/mixed/matrix_level_mae.png` |
| Fig. 10, marginal divergence | `vs_deepseek-chat-v3-0324/mixed/divergence.png` |

Fig. 8 (confusion matrices for the Baby category) is built from the
`confusion_*.csv` files written by `evaluate.py`; the command is in the
docstring of `LLM-BASED/src/evaluation/combine_confusion.py`.

**Significance tests** (paired bootstrap):

```bash
python experiments/run_significance.py                  # llama-4-scout vs. mistral-nemo
python experiments/run_significance_baseline.py both    # each LLM vs. the TF-IDF baseline
```

**Baselines** (from the repository root; see [`baselines/README.md`](./baselines/README.md)):

```bash
python baselines/majority_class/run.py
python baselines/keyword_rules/run.py
python baselines/tfidf_logreg/run.py
```

## Re-running the LLMs

Not needed to reproduce the results above. To score reports again, put your
API key in `LLM-BASED/config/config.yaml` (the `api_key` field is a
placeholder; do not commit the file with a real key), choose the model and
the input/output paths, and run `python experiments/run_risk_assessor_eval.py`
from `LLM-BASED/`. The prompts are in
`LLM-BASED/src/risk_scorer/risk_scorer.py` and
`LLM-BASED/config/sources/saferproducts_gov.yaml`.

The check of the DeepSeek-V3 hints on a stratified sample is
`python experiments/run_hint_ablation_sample.py`; it calls the API, and the
outputs of the run reported in the paper are already in `results/`.
