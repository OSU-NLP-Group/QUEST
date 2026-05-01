# QUEST

<p align="center">
  <strong>QUEST</strong>
</p>

<div align="center" style="line-height: 1; margin-top: 16px;">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=for-the-badge&logo=arXiv&logoColor=white" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="#"><img src="https://img.shields.io/badge/Dataset-FFB7B2?style=for-the-badge&logo=huggingface&logoColor=ffffff" alt="Dataset"></a>
  <a href="#"><img src="https://img.shields.io/badge/Model-FFD966?style=for-the-badge&logo=huggingface&logoColor=ffffff" alt="Model"></a>
  <a href="#documentation-map"><img src="https://img.shields.io/badge/Docs-2563EB?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"></a>
</div>

<br>

<p align="center">
  <a href="#">Hugging Face</a> | <a href="#documentation-map">Documentation</a>
</p>

## Introduction

Quest is a general-purpose Deep Search Agent designed to handle a wide range of
search tasks, with strong capabilities in fact seeking, citation grounding, and
report synthesis.

## Table of Contents

- [Introduction](#introduction)
- [Environment Setup](#environment-setup)
- [Runtime Configuration](#runtime-configuration)
- [Benchmark Replication](#benchmark-replication)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Mid-training / SFT Training](#mid-training--sft-training)
- [Run Training](#run-training)
  - [RL Backend](#rl-backend)
- [Data Generation](#data-generation)
  - [Objective Tasks](#objective-tasks)
  - [Objective Verifier Scripts](#objective-verifier-scripts)
  - [Open-Ended Tasks](#open-ended-tasks)
  - [Open-Ended Evaluation](#open-ended-evaluation)
- [Documentation Map](#documentation-map)

## Environment Setup

Create an environment and install the shared runtime dependencies:

```bash
pip install -r requirements.txt
```

This environment is intended for inference, data generation, and evaluation
workflows. Training uses separate backend stacks: install SFT dependencies under
`training_scripts/sft/` according to LlamaFactory requirements, and install RL
dependencies under `training_scripts/rl/` according to VERL requirements.

Optional local databases and caches used by search, visit, and scholar tools
live under the repository-level `database/` directory:

```text
database/
```

These files are not included in the repository. If you do not download existing
databases, the search and visit caches are created automatically during runs.
Providing prebuilt databases is useful when you want to reuse cached results,
reduce external requests, or run workflows that require prepared search/scholar
indexes.

## Runtime Configuration

The exact environment variables depend on the workflow. Common groups include:

| Group | Examples | Used By |
| --- | --- | --- |
| Search | `SERPER_KEY_ID` | Search and scholar fallback |
| Visit | `JINA_API_KEYS` | Page reading and page summarization |
| Azure/OpenAI-compatible | `API_KEY`, `API_BASE`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT` | Shared legacy and fallback LLM paths |
| Inference summary and memory | `SUMMARY_MODEL_NAME`, `MEMORY_MODEL_NAME`, `MEMORY_API_KEY`, `MEMORY_API_BASE` | Visit summarization and memory condensation |
| Reward/eval LLMs | `EVAL_LLM_*`, `CITATION_EVAL_LLM_*`, `OPENENDED_EVAL_LLM_*` | Objective, citation, and open-ended reward evaluation |
| Services | `SEARCH_NODES_CONF`, `SCHOLAR_NODES_CONF`, `PYTHON_NODES_CONF`, `EVAL_LLM_NODES_CONF` | Tool and local eval-node routing |

For inference, see [`inference/api_config.yaml`](inference/api_config.yaml) for
the default configuration template. For the full RL backend environment list,
see the [DeepResearch recipe README](training_scripts/rl/recipe/deepresearch/README.md#secrets-and-environment).

## Benchmark Replication

### Inference

Use `inference/` when you have a model endpoint and want to run benchmark
predictions with the QUEST agent.

Before launching, configure:

```text
api_config.yaml
server_endpoints.conf
```

Then check the benchmark script and update:

```text
DATASET
OUTPUT_PATH
TASK_LOG_DIR
MODEL_PATH
MAX_WORKERS
MEMORY_THRESHOLD
LLM_MAX_TOKENS
API_CONFIG_FILE
SERVER_ENDPOINTS_FILE
```

Run the benchmark-specific launch script from `inference/` after configuration.
Endpoint routing is controlled by `server_endpoints.conf`, which the agent can
reload during a run. See [`inference/README.md`](inference/README.md) for the
available launch scripts and benchmark-specific defaults.

### Evaluation

Evaluation scripts consume prediction directories produced by `inference/`.

| Benchmark | Directory |
| --- | --- |
| BrowseComp | [`evaluation/browsecomp/`](evaluation/browsecomp/) |
| GAIA | [`evaluation/gaia/`](evaluation/gaia/) |
| HLE | [`evaluation/hle/`](evaluation/hle/) |
| DeepResearch Bench | [`evaluation/drbench/`](evaluation/drbench/) |
| Mind2Web2 | [`evaluation/Mind2Web2/`](evaluation/Mind2Web2/) |

For a new run, update the target result directory, dataset path, model or run
name, judge model, worker count, and judge credentials.

See [`evaluation/README.md`](evaluation/README.md) for benchmark-specific
commands and notes.

## Mid-training / SFT Training

Use `training_scripts/sft` for mid-training and supervised fine-tuning workflows.
Before training, prepare the mid-training/SFT datasets and convert them to the
format expected by LlamaFactory.

Dataset release:

```text
https://huggingface.co/datasets/<org>/<quest-midtraining-sft-data>
```

The SFT backend is based on LlamaFactory. Use its data configuration and training
entrypoints under `training_scripts/sft/LlamaFactory/` after the datasets are
prepared.

## Run Training

### RL Backend

Use `training_scripts/rl` as the working directory:

```bash
cd training_scripts/rl
```

The active recipe is:

```text
recipe/deepresearch/
```

Core files:

| Path | Purpose |
| --- | --- |
| `recipe/deepresearch/run_deepresearch_fully_async_megatron.sh` | Main fully async Megatron launcher |
| `recipe/deepresearch/agent_loop/` | Multi-turn research rollout logic |
| `recipe/deepresearch/reward.py` | Reward routing for objective, citation, and open-ended tasks |
| `recipe/deepresearch/tools/` | Search, scholar, visit, Python, memory-related tool implementations |
| `recipe/deepresearch/scripts/` | Search/scholar services and FAISS build scripts |
| `recipe/deepresearch/config/` | Tool, service-node, eval-node, and trainer configs |
| `recipe/deepresearch/data/` | Default train/validation parquet files |

Before building FAISS, confirm that the required databases are available:

```text
visit database
search database
scholar database
```

Also make sure the Python interpreter service is running if the training workers
will use the Python tool.

Then build the FAISS indexes:

```bash
bash recipe/deepresearch/scripts/init_faiss_search.sh --skip-merge
bash recipe/deepresearch/scripts/init_faiss_scholar.sh --skip-merge
```

Then start the services:

```bash
bash recipe/deepresearch/scripts/run_search_service.sh
bash recipe/deepresearch/scripts/run_scholar_service.sh
```

Launch training:

```bash
bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
```

The full runbook, including environment variables and FAISS setup, is in:

```text
training_scripts/rl/recipe/deepresearch/README.md
```

## Data Generation

### Objective Tasks

Objective tasks use a verifiable rubric-tree pipeline:

High-level flow:

```text
generate trajectories -> merge rubric predictions -> format verifier inputs
-> refine rubric trees -> verify rubric trees -> extract accepted questions
```

See [`task/obj_task/README.md`](task/obj_task/README.md) for the runnable
commands and expected input/output paths.

### Objective Verifier Scripts

Generate one Python verifier script per formatted objective task.

See [`task/obj_eval/README.md`](task/obj_eval/README.md) for the generation
command and expected formatted-task input structure.

### Open-Ended Tasks

Open-ended longform generation lives under `task/sub_task/`.

High-level flow:

```text
generate longform tasks -> extract proposed QAs -> generate criteria
-> polish criteria -> generate reference answers -> refine final answers
-> extract final answers
```

See [`task/sub_task/README.md`](task/sub_task/README.md) for the runnable
commands and expected input/output paths.

### Open-Ended Evaluation

Rubric-based document quality evaluation lives under `task/sub_eval/`:

```bash
cd task/sub_eval
bash run_eval.sh
```

It compares an answer against a reference answer across criteria such as
comprehensiveness, insight, instruction following, and readability.

## Documentation Map

We provide details of each component in the READMEs below.

| Area | Directory | Main README | What It Contains |
| --- | --- | --- | --- |
| Inference | [`inference/`](inference/) | [`inference/README.md`](inference/README.md) | QUEST inference pipeline |
| RL backend | [`training_scripts/rl/recipe/deepresearch/`](training_scripts/rl/recipe/deepresearch/) | [`training_scripts/rl/recipe/deepresearch/README.md`](training_scripts/rl/recipe/deepresearch/README.md) | QUEST RL training recipe |
| SFT backend | [`training_scripts/sft/`](training_scripts/sft/) | [`training_scripts/sft/README.md`](training_scripts/sft/README.md) | LlamaFactory-based SFT backend |
| Objective task generation | [`task/obj_task/`](task/obj_task/) | [`task/obj_task/README.md`](task/obj_task/README.md) | Objective task generation pipeline |
| Objective verifier scripts | [`task/obj_eval/`](task/obj_eval/) | [`task/obj_eval/README.md`](task/obj_eval/README.md) | Objective-task verifier generation |
| Open-ended task generation | [`task/sub_task/`](task/sub_task/) | [`task/sub_task/README.md`](task/sub_task/README.md) | Open-ended task generation pipeline |
| Open-ended evaluation | [`task/sub_eval/`](task/sub_eval/) | [`task/sub_eval/README.md`](task/sub_eval/README.md) | Open-ended task evaluation pipeline |
| Evaluation | [`evaluation/`](evaluation/) | [`evaluation/README.md`](evaluation/README.md) | Benchmark evaluation scripts |
