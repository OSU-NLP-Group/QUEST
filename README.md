# QUEST

QUEST is a research-agent codebase for long-horizon web research, task
generation, inference, benchmark evaluation, and DeepResearch reinforcement
learning. The repository is organized around four workflows:

- task generation for objective and open-ended research tasks
- inference with a multi-turn ReAct-style research agent
- benchmark evaluation for generated answers
- RL training for the DeepResearch agent recipe

## Repository Layout

`inference/`

Inference-time research agent code. This directory contains the ReAct agent,
search/scholar/visit/memory/python tools, endpoint configuration, and benchmark
launch scripts. Start here when running a trained or served model on BrowseComp,
HLE, DeepResearch Bench, or Mind2Web2.

See [`inference/README.md`](inference/README.md).

`evaluation/`

Benchmark-specific evaluation scripts and assets. Current benchmark folders
include BrowseComp, HLE, DeepResearch Bench, and Mind2Web2. Inference should be
run first, then the corresponding evaluation script should be pointed at the new
prediction directory.

See [`evaluation/README.md`](evaluation/README.md).

`task/`

Task construction and task-evaluation utilities.

- `task/obj_task/`: objective task generation and rubric-tree verification
- `task/obj_script/`: conversion from formatted objective tasks to executable
  verifier scripts
- `task/sub_task/`: open-ended longform task generation
- `task/sub_eval/`: rubric-based document-quality evaluation for open-ended
  answers

The historical directory names still use `sub` in some places. In the RL recipe,
the same task family is referred to as `open-ended`.

`training_scripts/`

Training code. The active DeepResearch RL recipe lives under:

```text
training_scripts/rl/recipe/deepresearch/
```

See [`training_scripts/rl/recipe/deepresearch/README.md`](training_scripts/rl/recipe/deepresearch/README.md).

## Secrets

Do not commit real API keys. Local secrets should live under:

```text
.secrets/
```

The repository `.gitignore` excludes this directory. A typical local file is:

```text
.secrets/deepresearch_api_keys.env
```

The exact variables depend on the workflow, but commonly include search,
visit-summary, judge, Azure/OpenAI-compatible, and reward/eval LLM credentials.
Use placeholders in committed scripts and load real values from the local secrets
file or the shell environment.

## Inference Quick Start

Install inference dependencies in your runtime environment:

```bash
cd inference
pip install -r requirements.txt
```

Before running, check the script for the target benchmark and update the usual
run-specific fields:

- `DATASET`
- `OUTPUT_PATH`
- `TASK_LOG_DIR`
- `MODEL_PATH`
- `MAX_WORKERS`
- `MEMORY_THRESHOLD`
- `LLM_MAX_TOKENS`
- `SERVER_ENDPOINTS_FILE`

Example launches:

```bash
cd inference
bash run_react_infer_bc.sh
bash run_react_infer_hle.sh
bash run_react_infer_drb.sh
bash run_react_infer_m2w2.sh
```

Endpoint routing is controlled by `inference/server_endpoints.conf`. The agent
can reload endpoint configuration during a run, so serving nodes can be adjusted
without rewriting the inference script.

## Evaluation Quick Start

Run inference first, then evaluate the generated predictions.

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

For a new run, update the target prediction directory, dataset path, model/run
name, judge model, and required judge credentials in the benchmark-specific
script.

## DeepResearch RL Quick Start

Use the RL root as the working directory:

```bash
cd training_scripts/rl
```

Prepare local secrets:

```bash
mkdir -p ../../.secrets
$EDITOR ../../.secrets/deepresearch_api_keys.env
```

Start tool services when training workers should call shared search/scholar
HTTP endpoints instead of loading local indexes directly:

```bash
bash recipe/deepresearch/scripts/run_search_service.sh
bash recipe/deepresearch/scripts/run_scholar_service.sh
```

Service endpoints are configured in:

```text
recipe/deepresearch/config/search_nodes.conf
recipe/deepresearch/config/scholar_nodes.conf
recipe/deepresearch/config/python_nodes.conf
recipe/deepresearch/config/eval_llm_nodes.conf
```

The default training data is:

```text
recipe/deepresearch/data/train_v4.parquet
recipe/deepresearch/data/val_v4.parquet
```

Launch the fully async Megatron training recipe:

```bash
bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
```

Common run-time controls are passed through environment variables. Examples:

```bash
DATA_KIND=both bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
DATA_KIND=obj bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
DATA_KIND=openended bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
```

For FAISS cache setup and full training configuration details, read the recipe
README.

## Task Generation Quick Start

Objective tasks:

```bash
cd task/obj_task
bash run_generate_tasks.sh
bash run_verify_rubric_trees.sh
```

Objective verifier script generation:

```bash
python task/obj_script/obj_script_generation.py \
  --input /path/to/formatted_tasks \
  --template task/obj_script/generation_prompt.md \
  --output /path/to/output_obj_scripts
```

Open-ended longform tasks:

```bash
cd task/sub_task
bash run_generate_tasks_longform.sh
```

Open-ended rubric evaluation:

```bash
cd task/sub_eval
bash run_eval.sh
```

Several task-generation scripts contain workflow-specific default paths. Check
the local README in each task directory before running a new generation job.

## Recommended Workflow

1. Generate or select tasks under `task/`.
2. Train or fine-tune the agent under `training_scripts/rl/recipe/deepresearch/`.
3. Serve the model and tool endpoints.
4. Run benchmark inference from `inference/`.
5. Run benchmark scoring from `evaluation/`.

For most day-to-day work, the detailed README files in the subdirectories are
the authoritative runbooks.
