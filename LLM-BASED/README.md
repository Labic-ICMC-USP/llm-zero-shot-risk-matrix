# Risk Matrix Scorer

Reads incident/complaint descriptions from a dataset, uses an LLM to assign:

- `severity_level` (1–4)
- `occurrence_likelihood` (1–5)
- `justification` (short rationale)

...and writes a new CSV with those columns, used to build a risk matrix.

Likelihood can optionally be left out of the LLM's job (`predict_likelihood: false`
in `config/config.yaml`, see [Severity-only mode](#severity-only-mode)).

The pipeline is organized around data sources (this repository includes
`saferproducts_gov`), each with its own preprocessing rules and LLM prompt,
defined in `config/sources/<source>.yaml`.

## Project structure

```
config/
  config.yaml              # per-run settings: model, paths, source, behavior
  sources/
    saferproducts_gov.yaml  # per-source: columns, severity scale, prompt
data/                       # not versioned (see data/ setup below)
  raw/<source>/
  interim/<source>/
  processed/<source>/       # one parquet file per category
docs/
  saferproducts_dataset.md  # dataset stats: counts per category/severity, text length
experiments/
  run_risk_assessor_eval.py # runs the scorer for one category
  run_significance.py       # paired bootstrap: llama-4-scout vs. mistral-nemo
  run_significance_baseline.py  # paired bootstrap: each LLM vs. TF-IDF + LogReg baseline
  run_hint_ablation_sample.py   # DeepSeek-V3 reference with vs. without hints, on a stratified sample
notebooks/                  # dataset exploration notebook
results/
  <source>/<model>/<category>_scored.csv
  <source>/<model>/evaluations/<category>/  # metrics, confusion matrices, etc.
  <source>/comparisons/                     # multi-model plots from evaluation/visualize.py
src/
  preprocessing/
    clean_incidents.py            # column selection/rename, optional filters
    map_severity.py               # maps ground-truth severity labels (passthrough if source has none)
    generate_categories_datasets.py  # splits into per-category parquet files
  risk_scorer/
    risk_scorer.py           # RiskAssessor: runs the LLM scoring
  evaluation/
    metrics.py                # comparison metrics (model vs. model, or model vs. ground truth)
    evaluate.py                # writes results/<source>/<model>/evaluations/<category>/*
    visualize.py               # plots from evaluate.py's outputs
    combine_confusion.py       # merges several confusion matrices into one figure (paper Fig. 8)
    significance.py            # paired bootstrap CIs/p-values (used by experiments/run_significance*.py)
  utils/
    source_config.py         # loads config/sources/<source>.yaml
    fix_error_rows.py        # re-scores rows with scoring_failed=True
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

The package uses a src-layout, so scripts under `src/` are run as modules
from the project root, e.g.:

```bash
python -m preprocessing.clean_incidents --source saferproducts_gov
```

Running a `.py` file directly (`python src/preprocessing/clean_incidents.py`)
will not resolve the `utils`/`preprocessing`/`risk_scorer` imports correctly.

### `data/`

Not committed to the repo (raw files are large; some are multiple GB).
Recreate the structure and place the raw files manually or via a source's
download step:

```bash
mkdir -p data/raw/saferproducts_gov
```

- `saferproducts_gov`: `IncidentReports.csv` from saferproducts.gov (manual export,
  see `docs/saferproducts_dataset.md`).

## Pipeline

Run in order, for a given `--source`:

```bash
python -m preprocessing.clean_incidents --source <source>
python -m preprocessing.map_severity --source <source>
python -m preprocessing.generate_categories_datasets --source <source>
```

- `clean_incidents.py`: selects/renames columns; if the source config sets
  `input_path`, reads from there instead of the raw file; optionally drops
  rows with an empty text column (`drop_empty_column`).
- `map_severity.py`: maps the source's raw severity labels to a 1–4 scale
  using `severity_map`. If the source has no ground-truth severity label
  (no `severity_map` in its config), this step is a passthrough
  (copies the file unchanged) so the next step can always rely on
  `incident_reports_mapped.csv` existing.
- `generate_categories_datasets.py`: splits into one Parquet file per
  `product_category` under `data/processed/<source>/`.

## Running the scorer

1. Edit `config/config.yaml`: `source`, `model_name`, `api_key`,
   `input_path` (a category's `.parquet` file), `output_path`.
2. Run:

```bash
python experiments/run_risk_assessor_eval.py
```

This is resumable: if `output_path` already has rows (e.g. the run was
interrupted), already-scored rows (matched by `report_id`) are skipped and
only the rest are processed, appending to the same file.

Rows that fail after all retries (`max_retries` in the config) are written
with `scoring_failed=True` and a conservative default score, instead of
being silently dropped or mixed in with genuine low-risk assessments.

### Severity-only mode

Set `predict_likelihood: false` in `config/config.yaml` to have the LLM assign
only `severity_level` and `justification`: likelihood is removed from the
prompt and from the JSON Schema, and no `occurrence_likelihood` column is
written. Use this when likelihood comes from a separate method. The default (`true`)
scores both, as described above.

### Recovering failed rows

```bash
python src/utils/fix_error_rows.py       # re-runs the LLM only for rows with scoring_failed=True
```

Scans every `*_scored.csv` in the folder of `output_path` from
`config/config.yaml` (e.g. `results/<source>/<model>/`).
Keep `predict_likelihood` matching the run being fixed.

## Evaluating results

```bash
python src/evaluation/evaluate.py --source <source> --model <model>                  # vs. this source's ground-truth labels, if any
python src/evaluation/evaluate.py --source <source> --model <model> --ref <ref_model> # vs. another model's own run

# Plots built from the above
python src/evaluation/visualize.py --source <source> --models <model1> <model2> [--ref <ref_model>]
```

`evaluate.py` always writes `results/<source>/<model>/evaluations/<category>/risk_matrix.csv`
(just this model's own predictions -- the same regardless of what it's
compared against). Everything that depends on the comparison target goes
one level deeper, in a `vs_<ref_model>/` or `vs_ground_truth/` subfolder,
so running both modes for the same model never overwrites the other's
results.

Ground-truth mode (no `--ref`) has one shape only, so its files
(`metrics_*.csv`, `confusion_*.csv`) sit directly in `vs_ground_truth/`.
Per-variable metrics/confusion are only written for whichever of
severity/likelihood this source has a real label for (see
`ground_truth_columns` below) -- for a source with no labels at all,
only the risk matrix is written.

`--ref` mode can produce two different comparisons, so `vs_<ref_model>/`
splits further:
- `normal/`: per-variable `metrics_*.csv`/`confusion_*.csv` against
  `--ref`'s own predictions, plus `risk_diff.csv` (this model's matrix
  vs. `--ref`'s own full matrix).
- `mixed/`: only written if this source's `ground_truth_columns` covers
  exactly one of severity/likelihood -- a "best available" reference
  matrix (`risk_matrix.csv`, `risk_diff.csv`) using ground truth for the
  covered axis and `--ref`'s prediction for the other. If it covers
  neither axis there's nothing to mix in; if it covers both, the full
  ground-truth matrix is already produced by the no-`--ref` run, so
  `mixed/` is skipped either way.

Ground truth, when available, is not a separate dataset -- it's a column
already present in the scored CSV (e.g. saferproducts_gov's
`victim_severity`, kept unchanged from the raw data through the whole
pipeline). `evaluate.py` reads it directly rather than merging in another
file.

`visualize.py` mirrors the same layout. Confusion-matrix PNGs are saved
next to the CSV they visualize (`evaluations/<category>/vs_<target>/[normal/]`),
since they're specific to one model/category. Everything else compares
several models on one chart, so it's saved under
`results/<source>/comparisons/vs_<target>/` instead: F1 and ordinal
error stay flat there (neither has a mixed-matrix counterpart), while
the three matrix-level plots (MAE, divergence, top-k overlap) split into
`normal/`/`mixed/` the same way `evaluate.py` does, skipped with a
message when `mixed/` doesn't exist for this source.

## Significance testing

Paired bootstrap (stratified by category, severity F1-macro, 2000 resamples)
for the comparisons reported in the paper. Run on demand from the project
root, not part of `evaluate.py`/`visualize.py`:

```bash
python experiments/run_significance.py                              # llama-4-scout vs. mistral-nemo
python experiments/run_significance_baseline.py [llama|mistral|both] # each LLM vs. TF-IDF + LogReg
```

`run_significance_baseline.py` reads the baseline's predictions from
`../baselines/tfidf_logreg/results/` (run `baselines/tfidf_logreg/run.py`
first) and only covers the 12 categories where that baseline was evaluated.

## `config/sources/<source>.yaml` reference

| Key | Used by | Purpose |
|---|---|---|
| `raw_file`, `encoding`, `skiprows` | `clean_incidents.py` | how to read the raw file |
| `columns_keep`, `columns_rename` | `clean_incidents.py` | column selection and renaming |
| `input_path` | `clean_incidents.py` | optional override; reads from here instead of the raw file |
| `drop_empty_column` | `clean_incidents.py` | optional; drops rows with an empty value in this (renamed) column |
| `injury_keyword`, `severity_map` | `map_severity.py` | maps raw severity labels to 1–4; omit both if the source has no ground truth |
| `prompt_intro`, `severity_levels` | `risk_scorer.py` | domain-specific LLM prompt content |
| `ground_truth_columns` | `evaluate.py` / `visualize.py` | optional map of `severity`/`likelihood` to the column already holding that variable's ground truth (e.g. `severity: victim_severity`); omit a key (or the whole map) if this source has no label for it |

## Adding a new source

1. Add `config/sources/<new_source>.yaml` (see the reference above).
2. Place the raw file under `data/raw/<new_source>/`.
3. Run the pipeline with `--source <new_source>`.

No changes to the pipeline scripts themselves should be needed.