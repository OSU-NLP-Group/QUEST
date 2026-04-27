# QUEST

QUEST is a long-horizon research-agent codebase for inference, reinforcement
learning, task generation, and benchmark evaluation.

## Overview

| Area | Directory | What It Contains | Main README |
| --- | --- | --- | --- |
| Inference | [`inference/`](inference/) | ReAct-style research agent, model endpoint routing, search/scholar/visit/memory/python tools, benchmark launch scripts | [`inference/README.md`](inference/README.md) |
| DeepResearch RL | [`training_scripts/rl/recipe/deepresearch/`](training_scripts/rl/recipe/deepresearch/) | DeepResearch agent loop, reward logic, tools, FAISS services, data, fully async Megatron launcher | [`training_scripts/rl/recipe/deepresearch/README.md`](training_scripts/rl/recipe/deepresearch/README.md) |
| RL backend | [`training_scripts/rl/`](training_scripts/rl/) | Vendored VERL-based RL stack used by the DeepResearch recipe | [`training_scripts/rl/README.md`](training_scripts/rl/README.md) |
| SFT backend | [`training_scripts/sft/`](training_scripts/sft/) | Vendored LlamaFactory for SFT workflows | [`training_scripts/sft/README.md`](training_scripts/sft/README.md) |
| Objective task generation | [`task/obj_task/`](task/obj_task/) | Objective research-task generation, trajectory formatting, rubric-tree verification | [`task/obj_task/README.md`](task/obj_task/README.md) |
| Objective verifier scripts | [`task/obj_script/`](task/obj_script/) | Converts formatted objective tasks into executable verifier scripts | [`task/obj_script/README.md`](task/obj_script/README.md) |
| Open-ended task generation | [`task/sub_task/`](task/sub_task/) | Longform/open-ended task generation, criteria generation, reference-answer generation | [`task/sub_task/README.md`](task/sub_task/README.md) |
| Open-ended evaluation | [`task/sub_eval/`](task/sub_eval/) | Rubric-based document-quality evaluation against reference answers | [`task/sub_eval/README.md`](task/sub_eval/README.md) |
| Evaluation | [`evaluation/`](evaluation/) | BrowseComp, HLE, DeepResearch Bench, and Mind2Web2 evaluation scripts | [`evaluation/README.md`](evaluation/README.md) |

Note: Some historical task directories use the name `sub`. In the RL recipe and
newer documentation, the same task family is called `open-ended`.

## Secrets

Never commit real API keys. Local secrets should live under:

```text
.secrets/
```

This directory is gitignored. A typical local file is:

```text
.secrets/deepresearch_api_keys.env
```

The exact variables depend on the workflow. Common groups include:

| Group | Examples | Used By |
| --- | --- | --- |
| Search | `SERPER_KEY_ID` | Search and scholar fallback |
| Visit | `JINA_API_KEY`, `JINA_API_KEYS` | Page reading and page summarization |
| Azure/OpenAI-compatible | `API_KEY`, `API_BASE`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` | Shared legacy and fallback LLM paths |
| Reward/eval LLMs | `EVAL_LLM_*`, `CITATION_EVAL_LLM_*`, `OPENENDED_EVAL_LLM_*` | Objective, citation, and open-ended reward evaluation |
| Memory and visit summary | `MEMORY_*`, `VISIT_SUMMARY_*` | Memory condensation and page-summary generation |
| Services | `SEARCH_NODES_CONF`, `SCHOLAR_NODES_CONF`, `PYTHON_NODES_CONF`, `EVAL_LLM_NODES_CONF` | Tool and local eval-node routing |

For the full DeepResearch RL environment list, see the
[DeepResearch recipe README](training_scripts/rl/recipe/deepresearch/README.md#secrets-and-environment).

## Inference

Use `inference/` when you already have a model endpoint and want benchmark
predictions.

```bash
cd inference
pip install -r requirements.txt
```

Before launching, check the benchmark script and update:

```text
DATASET
OUTPUT_PATH
TASK_LOG_DIR
MODEL_PATH
MAX_WORKERS
MEMORY_THRESHOLD
LLM_MAX_TOKENS
SERVER_ENDPOINTS_FILE
```

Common entrypoints:

```bash
bash run_react_infer_bc.sh
bash run_react_infer_hle.sh
bash run_react_infer_drb.sh
bash run_react_infer_m2w2.sh
```

Endpoint routing is controlled by:

```text
inference/server_endpoints.conf
```

The agent can reload endpoint configuration during a run.

## DeepResearch RL

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

Start tool services when workers should use shared HTTP services:

```bash
bash recipe/deepresearch/scripts/run_search_service.sh
bash recipe/deepresearch/scripts/run_scholar_service.sh
```

Launch training:

```bash
bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
```

Common data filters:

```bash
DATA_KIND=both bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
DATA_KIND=obj bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
DATA_KIND=openended bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
```

The full runbook, including environment variables and FAISS setup, is in:

```text
training_scripts/rl/recipe/deepresearch/README.md
```

## Task Generation

### Objective Tasks

Objective tasks use a verifiable rubric-tree pipeline:

```bash
cd task/obj_task
bash run_generate_tasks.sh
bash run_verify_rubric_trees.sh
```

High-level flow:

```text
generate trajectories -> merge rubric predictions -> format verifier inputs
-> verify rubric trees -> extract accepted questions
```

### Objective Verifier Scripts

Generate one Python verifier script per formatted objective task:

```bash
python task/obj_script/obj_script_generation.py \
  --input /path/to/formatted_tasks \
  --template task/obj_script/generation_prompt.md \
  --output /path/to/output_obj_scripts
```

### Open-Ended Tasks

Open-ended longform generation lives under `task/sub_task/`:

```bash
cd task/sub_task
bash run_generate_tasks_longform.sh
```

High-level flow:

```text
generate longform tasks -> extract proposed QAs -> generate criteria
-> polish criteria -> generate reference answers -> refine final answers
```

### Open-Ended Evaluation

Rubric-based document quality evaluation lives under `task/sub_eval/`:

```bash
cd task/sub_eval
bash run_eval.sh
```

It compares an answer against a reference answer across criteria such as
comprehensiveness, insight, instruction following, and readability.

## Evaluation

Evaluation scripts consume prediction directories produced by `inference/`.

BrowseComp:

```bash
cd evaluation/browsecomp
bash run_judge.sh
```

HLE:

```bash
cd evaluation/hle
bash run_judge.sh
```

DeepResearch Bench:

```bash
cd evaluation/drbench/eval/deep_research_bench
python convert_to_eval_format.py
bash run_benchmark.sh
```

For a new run, update the target result directory, dataset path, model or run
name, judge model, worker count, and judge credentials.

## Documentation Map

| README | Scope |
| --- | --- |
| [`inference/README.md`](inference/README.md) | Inference parameters, benchmark scripts, endpoint hot reload, resume behavior |
| [`training_scripts/rl/recipe/deepresearch/README.md`](training_scripts/rl/recipe/deepresearch/README.md) | DeepResearch RL recipe, services, data, secrets, launch commands |
| [`training_scripts/rl/README.md`](training_scripts/rl/README.md) | Vendored VERL documentation |
| [`training_scripts/sft/README.md`](training_scripts/sft/README.md) | Vendored LlamaFactory notes |
| [`task/obj_task/README.md`](task/obj_task/README.md) | Objective task generation and rubric verification |
| [`task/obj_script/README.md`](task/obj_script/README.md) | Verifier-script generation from formatted objective tasks |
| [`task/sub_task/README.md`](task/sub_task/README.md) | Open-ended longform task generation workflow |
| [`task/sub_eval/README.md`](task/sub_eval/README.md) | Open-ended rubric-based evaluation workflow |
| [`evaluation/README.md`](evaluation/README.md) | Benchmark-specific scoring workflows |
