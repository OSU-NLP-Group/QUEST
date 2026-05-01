# Objective Task Generation

This directory contains the objective task generation pipeline used by QUEST.
It generates research trajectories, formats rubric-tree verifier inputs, verifies
rubric trees, and extracts accepted objective questions.

## Workflow

High-level flow:

```text
generate trajectories -> merge rubric predictions -> format verifier inputs
-> verify rubric trees -> extract accepted questions
```

Recommended execution order:

```bash
cd task/obj_task
bash run_generate_tasks.sh
python merge_rubric_predictions.py
python format_trajectories.py
bash run_verify_rubric_trees.sh
python extract_proposed_questions.py
```

## Generate Trajectories

`run_generate_tasks.sh` configures the generation model, search/visit tools,
optional memory settings, and cache paths, then runs `generate_tasks.py`.

Default outputs:

```text
./outputs/objective_trajectories/
./outputs/objective_task_logs/
```

Before running, set the generation and tool credentials required by your
provider. Common variables include:

```text
SERPER_KEY_ID
JINA_API_KEYS
SUMMARY_AZURE_API_KEY
SUMMARY_AZURE_API_BASE
SUMMARY_AZURE_API_VERSION
SUMMARY_MODEL_NAME
DEEPRESEARCH_AWS_CREDENTIALS
DEEPRESEARCH_MODEL_NAME
DEEPRESEARCH_OPENAI_API_KEY
```

The main path variables can be overridden:

```bash
TRAJ_DIR=/path/to/objective_trajectories \
TASK_LOG_DIR=/path/to/objective_task_logs \
bash run_generate_tasks.sh
```

## Post-process Trajectories

Merge refined rubric predictions back into raw trajectories:

```bash
python merge_rubric_predictions.py \
  --input-dir /path/to/objective_trajectories
```

Format trajectories into verifier inputs:

```bash
python format_trajectories.py \
  --input-dir /path/to/objective_trajectories
```

Formatted verifier inputs are written under:

```text
/path/to/objective_trajectories/formatted/
```

## Verify Rubric Trees

`run_verify_rubric_trees.sh` runs `verify_rubric_trees.py` over formatted
trajectory files and writes verification results, logs, and accepted
trajectories.

Before running, set the rubric-verifier model credentials:

```text
RUBRIC_VERIFIER_AWS_ACCESS_KEY_ID
RUBRIC_VERIFIER_AWS_SECRET_ACCESS_KEY
RUBRIC_VERIFIER_AWS_REGION_NAME
RUBRIC_VERIFIER_MODEL_NAME
```

Typical command:

```bash
FORMATTED_TRAJ_DIR=/path/to/objective_trajectories/formatted \
bash run_verify_rubric_trees.sh
```

Default verification outputs:

```text
/path/to/objective_trajectories/formatted/verifier/rubric-tree-verifier-logs/
/path/to/objective_trajectories/formatted/verifier/rubrc-tree-verification-results.json
/path/to/objective_trajectories/formatted/verifier/accepted_trajectories/
```

## Extract Accepted Questions

After verification, extract accepted proposed questions into a clean JSONL file:

```bash
python extract_proposed_questions.py \
  --input-dir /path/to/objective_trajectories/formatted/verifier/accepted_trajectories \
  --output-file /path/to/objective_trajectories/extracted_questions.jsonl
```

Each output row contains:

```text
question
answer
filename
```

## Main Files

| File | Purpose |
| --- | --- |
| `run_generate_tasks.sh` | Generation launcher and default environment configuration |
| `generate_tasks.py` | Main task generation entrypoint |
| `generation_agent.py` | Multi-turn generation agent |
| `generation_prompts.py` | Objective task and rubric prompts |
| `tool_search.py`, `tool_visit.py` | Search and page-reading tools |
| `merge_rubric_predictions.py` | Merge refined rubric predictions back into trajectories |
| `format_trajectories.py` | Convert raw trajectories into verifier inputs |
| `run_verify_rubric_trees.sh` | Rubric verification launcher |
| `verify_rubric_trees.py` | Rubric-tree verifier |
| `extract_proposed_questions.py` | Extract accepted questions into JSONL |
