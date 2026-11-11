# Towards Benchmarking Agentic Data Scientists

**Kartik Nagaraj Nayak\*, Nitishkumar Solpure\*, Niladri Mitra\*, Tom Zehle, Alexander Pfefferle, Omar Swelam, Frank Hutter**
University of Freiburg · ELLIS Institute Tübingen · Zuse School ELIZA · Prior Labs  
\* Equal contribution

*AutoAI-FM Workshop at IJCAI-ECAI 2026 (oral + poster)*

[📄 Paper (PDF)](paper/Towards_Benchmarking_Agentic_Data_Scientists.pdf) · [Official code repository](https://github.com/KARTIK-N-NAYAK/Benchmarking-Agentic-Data-Scientists)
<!-- TODO: add the online paper link (arXiv / OpenReview / workshop page) next to the PDF link -->

> This is my copy of the official repository cited in the paper. I'm an equal-contribution co-author; see [My contributions](#my-contributions).

---

## TL;DR

Agentic ML systems use an LLM to write, run, and fix their own machine-learning pipelines. We put four of them (**AgentK, AIDE, MLE-STAR, MLZero**) and the AutoML baseline **AutoGluon** under a single protocol on 10 tabular datasets. All five systems used the same splits, the same metrics, and the same backbone LLM, and we measured both predictive performance and whether each system produced a valid result at all.

- **AutoGluon ranked first overall and never failed.** It was also the cheapest system to run, because it uses no LLM tokens.
- **The agents can win individual datasets but are often unreliable.** Out of 30 runs each, MLZero failed 15, AgentK 11, AIDE 5 and MLE-STAR 0.
- **Taking reliability into account changes the ranking.** On mean performance alone, MLZero moves from 5th to 2nd. Several of its best scores come from datasets where only one of three folds finished.
- **Two of the 12 curated datasets leaked the target into the test files** (Bike Sharing Demand, Bioresponse). The agents that read the raw data exploited the leak, so we removed those datasets after a per-fold audit.

## Results

Average rank across the 10 datasets (lower is better). The robustness-adjusted score combines performance, cross-fold stability, and reliability (see [Scoring](#robustness-adjusted-score)).

| Method | Avg. rank, robustness-adjusted ↓ | Avg. rank, mean performance only ↓ | Failed runs (of 30) | Success rate |
|---|---|---|---|---|
| **AutoGluon** | **1.75 ± 1.36** | **2.10** | 0 | 100% |
| AIDE | 2.75 ± 1.23 | 3.05 | 5 | 83.3% |
| MLE-STAR† | 3.15 ± 1.06 | 3.70 | 0 | 100% |
| AgentK | 3.55 ± 1.54 | 3.65 | 11 | 63.3% |
| MLZero | 3.80 ± 1.14 | 2.50 | 15 | 50.0% |

† Open-source reimplementation ([MLE-STAR-Open](https://github.com/WalkingDevFlag/MLE-STAR-Open)), not the original Google release.

**Statistical significance.** The Friedman test finds a significant difference between methods (χ² = 10.46, p = 0.033). The Nemenyi critical difference is 1.93 and the full spread of average ranks is 2.05, so only the extremes separate. Pairwise Wilcoxon tests do not survive Holm or Benjamini–Hochberg correction, so we make no pairwise claims. With 10 datasets, statistical resolution is the main limitation. See Section 6 of the paper.

## Benchmark design

### Systems

| System | Approach | Reference |
|---|---|---|
| AgentK | LLM agent that builds pipelines through iterative reasoning and tool use | Grosnit et al., 2025 ([arXiv:2411.03562](https://arxiv.org/abs/2411.03562)) |
| AIDE | Iterative code refinement driven by execution feedback | Jiang et al., 2025 ([arXiv:2502.13138](https://arxiv.org/abs/2502.13138)) |
| MLE-STAR | Web-search initialisation followed by targeted refinement of code blocks | Nam et al., 2025 ([arXiv:2506.15692](https://arxiv.org/abs/2506.15692)) |
| MLZero | Multi-agent system with search over ML pipelines | Fang et al., 2025 ([arXiv:2505.13941](https://arxiv.org/abs/2505.13941)) |
| AutoGluon | Non-agentic AutoML baseline (ensembling / stacking) | Erickson et al., 2020 ([arXiv:2003.06505](https://arxiv.org/abs/2003.06505)) |

### Datasets

We started from 23 public tabular datasets (mostly Kaggle) and cut them to 12 after a quality review. The leakage audit then removed two more, leaving 10. Each dataset is split into **three fixed, non-overlapping folds**, and each fold has its own train/test split. Every system sees exactly the same splits.

| Dataset | Task | Metric | Kaggle competition |
|---|---|---|---|
| Santander Customer Satisfaction | Binary | ROC-AUC | `santander-customer-satisfaction` |
| Forest Cover Type Prediction | Multiclass | Multiclass ROC-AUC | `forest-cover-type-prediction` |
| Give Me Some Credit | Binary | ROC-AUC | `GiveMeSomeCredit` |
| Otto Group Product Classification | Multiclass | Multiclass ROC-AUC | `otto-group-product-classification-challenge` |
| Porto Seguro Safe Driver Prediction | Binary | ROC-AUC | `porto-seguro-safe-driver-prediction` |
| Predict Who Is More Influential | Binary | ROC-AUC | `predict-who-is-more-influential-in-a-social-network` |
| Home Data for ML Course | Regression | R², RMSE | `home-data-for-ml-course` |
| Mercedes-Benz Greener Manufacturing | Regression | R², RMSE | `mercedes-benz-greener-manufacturing` |
| Restaurant Revenue Prediction | Regression | R², RMSE | `restaurant-revenue-prediction` |
| Sberbank Russian Housing Market | Regression | R², RMSE | `sberbank-russian-housing-market` |

The task, metric, and dataset descriptions each agent received are in [`descriptions_for_agents/`](descriptions_for_agents/), exactly as they were passed in.

### Protocol

- **Execution engine.** For every run, the engine places the fold's `train.csv` / `test.csv` at a fixed location and starts the agent with the task description. It then collects the prediction file. A run is **valid** only if that file parses and contains a prediction for every test row. Anything else counts as a failure. There are no automatic retries.
- **Backbone LLM.** Every agent used `gpt-oss-120b`, served behind an OpenAI-compatible endpoint. Decoding settings were each system's own defaults.
- **Hardware.** Each run was one SLURM job with 8 CPU cores and one NVIDIA RTX 2080 Ti. Compute nodes could reach only the model endpoint, with no other internet access.
- **Time budgets.**

  | System | Budget per run |
  |---|---|
  | AutoGluon | 7,200 s (`best_quality` preset) |
  | AgentK | capped at 14,400 s (internal budget: 18,000 s total, 600 s per submission) |
  | AIDE, MLE-STAR, MLZero | 24 h |

- **Leakage audit.** We checked every fold of the 12 datasets for five things:
  - the target column present in the test file
  - the target reconstructible from a single feature or from an additive pair of features
  - schema mismatches between train and test
  - row overlap between train and test

### Robustness-adjusted score

For each dataset and system:

```
score = p · r / (1 + s)
```

where

- **p** is normalised performance in [0, 1]:
  - ROC-AUC: `clip((μ − 0.5) / 0.5, 0, 1)`
  - R²: `clip(μ, 0, 1)`
  - RMSE: `min_j(μ_j) / μ`, i.e. relative to the best system on that dataset
- **s** is the normalised spread across folds:
  - ROC-AUC: `σ / 0.5`
  - R²: `σ`
  - RMSE: `σ / μ`
- **r** is reliability, the fraction of the three folds that produced a valid result.

All three factors are dimensionless, so scores can be compared across datasets and metrics. A system that finishes only one fold gets s = 0 by construction, but r = 1/3 then dominates its score. Ranks are computed per dataset and averaged.

## Repository structure

```
├── descriptions_for_agents/   Task, metric and dataset descriptions given to every agent
├── paper/                     Workshop paper (PDF)
├── config/                    Kaggle competition lists used by the downloader
├── src/
│   ├── autogluon_runner_cla.py    AutoGluon baseline, classification (ROC-AUC)
│   ├── autogluon_runner_reg.py    AutoGluon baseline, regression
│   ├── agentk_runner.py           AgentK launcher as used on our SLURM cluster
│   ├── download_data.py           Kaggle competition downloader
│   └── download_description.py    Kaggle description fetcher
└── third_party/
    ├── agentk/        AgentK
    ├── aideml/        AIDE
    ├── mle_star/      MLE-STAR-Open (unofficial reimplementation)
    ├── mlzero/        AutoGluon Assistant / MLZero
    ├── autogluon/     Notes for a standalone AutoGluon environment
    └── evaluation/    Per-fold metric scripts (classification, regression)
```

## Getting started

The experiments ran on Linux (a SLURM GPU cluster). MLZero is Linux-only. On Windows, use WSL.

### 1. Install

Requirements: Python 3.11, [`uv`](https://docs.astral.sh/uv/), and an OpenAI-compatible LLM endpoint. A CUDA GPU is optional; `pyproject.toml` pulls PyTorch from the CUDA 12.8 wheel index.

```bash
git clone https://github.com/<your-username>/Benchmarking-Agentic-Data-Scientists.git
cd Benchmarking-Agentic-Data-Scientists
uv sync
uv pip install autogluon.tabular      # needed for the AutoGluon baseline
```

### 2. Configure the LLM endpoint

```bash
cp .env.example .env
# then set OPENAI_BASE_URL, OPENAI_API_KEY and ROOT_AGENT_MODEL in .env
set -a; source .env; set +a
```

### 3. Get the data

The datasets come from Kaggle and can't be redistributed. You need Kaggle API credentials (`~/.kaggle/kaggle.json`), and you have to accept each competition's rules on its Kaggle page. Then:

```bash
kaggle competitions download -c santander-customer-satisfaction -p data/raw
# repeat for each competition in the dataset table above
```

Split each dataset into three non-overlapping folds, each with its own `train.csv` and `test.csv`. The fold files themselves are not included in this repository.

### 4. Run a system on one fold

**AutoGluon**

```bash
uv run python src/autogluon_runner_cla.py \
  --train_path data/<dataset>/fold0/train.csv \
  --test_path  data/<dataset>/fold0/test.csv \
  --target <target_column> \
  --output_path outputs/autogluon/<dataset>/fold0/predictions.csv \
  --time_limit 7200 --num_gpus 1
# use src/autogluon_runner_reg.py for regression datasets
```

**AgentK.** AgentK has its own environment (Python 3.10–3.11, defined in `third_party/agentk/pyproject.toml`).

```bash
cd third_party/agentk
unzip -n third_party/ramp-hyperopt.zip -d third_party
unzip -n third_party/ramp-workflow.zip -d third_party
export PYTHONPATH="$PWD/src:$PWD/third_party/ds-agent/src"

python script/run_complete_pipeline_with_react.py \
  --task_id <task> --is_local_task --tabular_task \
  --alt_raw_data_root /path/to/raw_local_tasks \
  --llm openai/gpt-oss-120b-llmlb --code_llm openai/gpt-oss-120b-llmlb \
  --total_time 18000 --max_time_per_submission 600 --attempt 0 \
  --workspace_name /path/to/workspaces/<task> \
  --blend_after_n 2 --post_scaffold_top_n 2 --post_scaffold_timeout 600 \
  --post_scaffold_llm openai/gpt-oss-120b-llmlb
```

`--llm` names a model config under `third_party/agentk/configs/llm/`. To use your own endpoint, set `base_url` in that YAML, or add a new YAML next to it.

**AIDE**

```bash
uv run python -m third_party.aideml.aide.run \
  data_dir=/path/to/fold_dir desc_file=/path/to/task_description.txt
uv run python third_party/aideml/finalize_submission.py   # copies the best submission to output/
```

**MLE-STAR-Open.** Put the task under `third_party/mle_star/machine_learning_engineering/tasks/<task_name>/`, then run:

```bash
uv run python third_party/mle_star/scripts/run_pipeline.py \
  --task-name <task_name> --task-type "Tabular Classification"   # or "Tabular Regression"
```

**MLZero.** The input directory must contain `train.csv`, `test.csv`, and `sample-submission.csv`.

```bash
uv run python third_party/mlzero/run_and_cleanup.py -i /path/to/fold_dir -o outputs/mlzero/<task>
```

### 5. Score predictions

```bash
python third_party/evaluation/eval_classification.py --pred predictions.csv --true test_y.csv --proba --out-json results/metrics.json
python third_party/evaluation/eval_regression.py     --pred predictions.csv --true test_y.csv --out-json results/metrics.json
```

These scripts align rows by an `id` column when one exists and infer the target and prediction columns automatically. Use `--id-col`, `--pred-y-col`, or `--true-y-col` to override.

## Limitations

- **Small suite.** With 10 datasets and 3 folds each, variance estimates are rough.
- **Unequal settings.** Time budgets and decoding settings differ between systems on purpose (see the paper, Section 3.6). That weakens any claim of a strictly equal comparison.
- **MLE-STAR is the reimplementation.** Results are for the open-source version, which uses DuckDuckGo instead of Google Search. They are not results for MLE-STAR as published.
- **Backbone mismatch.** None of the agents was designed or tuned for `gpt-oss-120b`. Some of the failures we saw probably reflect that mismatch rather than the systems themselves.
- **Tabular only.** Several of these systems also handle image, text, or audio tasks. This benchmark does not test those.

## My contributions

I'm one of three equal-contribution first authors. My work on this project:

- **AgentK.** Integrated AgentK into the benchmark, served it with `gpt-oss-120b`, and ran it on the cluster.
- **AutoGluon baseline.** Wrote the AutoGluon runners for classification and regression.
- **Evaluation.** Wrote the per-fold metric scripts.
- **Paper.** Edited the paper.
- **Repository.** Did the final setup of the public repository, including the agent descriptions.

## Citation

```bibtex
@inproceedings{nayak2026towards,
  title     = {Towards Benchmarking Agentic Data Scientists},
  author    = {Nayak, Kartik Nagaraj and Solpure, Nitishkumar and Mitra, Niladri and
               Zehle, Tom and Pfefferle, Alexander and Swelam, Omar and Hutter, Frank},
  booktitle = {AutoAI-FM Workshop at IJCAI-ECAI 2026},
  year      = {2026}
}
```

## Acknowledgements and third-party code

The agent implementations under `third_party/` are the work of their original authors and are covered by their own licences. They are bundled here with modifications for running under a shared protocol.

| Code | Upstream | Licence |
|---|---|---|
| AIDE | [WecoAI/aideml](https://github.com/WecoAI/aideml) | MIT |
| AutoGluon Assistant / MLZero | [autogluon/autogluon-assistant](https://github.com/autogluon/autogluon-assistant) | Apache-2.0 |
| MLE-STAR-Open | [WalkingDevFlag/MLE-STAR-Open](https://github.com/WalkingDevFlag/MLE-STAR-Open) | see upstream |
| AgentK | Huawei Agent K team (Grosnit et al., 2025) | see upstream |

Funding acknowledgements are listed in the paper.
