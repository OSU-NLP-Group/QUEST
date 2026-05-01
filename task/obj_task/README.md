# Objective Task Generation

This directory runs the QUEST objective task generation pipeline. It produces
research trajectories, formats them for rubric-tree verification, filters
accepted tasks, and exports accepted objective questions.

## Overview

High-level flow:

```text
generate trajectories -> merge rubric predictions -> format verifier inputs
-> verify rubric trees -> extract accepted questions
```

Main entrypoints:

| Stage | Command | Output |
| --- | --- | --- |
| Generate trajectories | `bash run_generate_tasks.sh` | Raw trajectory JSON files |
| Merge rubric refinements | `python merge_rubric_predictions.py` | Updated trajectory JSON files |
| Format verifier inputs | `python format_trajectories.py` | `formatted/` verifier inputs |
| Verify rubric trees | `bash run_verify_rubric_trees.sh` | Verification logs and accepted trajectories |
| Extract accepted questions | `python extract_proposed_questions.py` | JSONL question set |

## Run Pipeline

From this directory:

```bash
cd task/obj_task
bash run_generate_tasks.sh
python merge_rubric_predictions.py
python format_trajectories.py
bash run_verify_rubric_trees.sh
python extract_proposed_questions.py
```

The default output root is:

```text
./outputs/objective_trajectories/
```

## Configuration

Before generation, configure the model and tool credentials required by your
provider. Common variables include:

| Variable | Purpose |
| --- | --- |
| `SERPER_KEY_ID` | Search and scholar fallback |
| `JINA_API_KEYS` | Page reading through the visit tool |
| `SUMMARY_MODEL_NAME` | Visit-summary model |
| `SUMMARY_AZURE_API_KEY`, `SUMMARY_AZURE_API_BASE`, `SUMMARY_AZURE_API_VERSION` | Azure/OpenAI-compatible summary endpoint |
| `DEEPRESEARCH_MODEL_NAME` | Main generation model |
| `DEEPRESEARCH_AWS_CREDENTIALS` | Bedrock credential list for generation |
| `DEEPRESEARCH_OPENAI_API_KEY` | OpenAI-compatible generation key, if used |

Path variables can be overridden from the shell:

```bash
TRAJ_DIR=/path/to/objective_trajectories \
TASK_LOG_DIR=/path/to/objective_task_logs \
bash run_generate_tasks.sh
```

Search and visit caches are enabled by default and use local files under
`./database/`. Override these only when you want to reuse or relocate caches:

| Variable | Default |
| --- | --- |
| `VISIT_CACHE_FILE` | `./database/visit_cache.db` |
| `SEARCH_CACHE_FILE` | `./database/search_cache.db` |
| `VISIT_CACHE_ENABLED`, `SEARCH_CACHE_ENABLED` | `true` |
| `VISIT_CACHE_RESUME`, `SEARCH_CACHE_RESUME` | `true` |

Rubric verification uses its own model configuration:

| Variable | Purpose |
| --- | --- |
| `RUBRIC_VERIFIER_MODEL_NAME` | Rubric verifier model |
| `RUBRIC_VERIFIER_AWS_ACCESS_KEY_ID` | Bedrock access key |
| `RUBRIC_VERIFIER_AWS_SECRET_ACCESS_KEY` | Bedrock secret key |
| `RUBRIC_VERIFIER_AWS_REGION_NAME` | Bedrock region |

To verify a specific formatted directory:

```bash
FORMATTED_TRAJ_DIR=/path/to/objective_trajectories/formatted \
bash run_verify_rubric_trees.sh
```

## Outputs

Default generated files are organized as:

```text
outputs/objective_trajectories/
outputs/objective_trajectories/formatted/
outputs/objective_trajectories/formatted/verifier/
outputs/objective_trajectories/formatted/verifier/accepted_trajectories/
outputs/objective_trajectories/extracted_questions.jsonl
```

The extracted JSONL rows contain:

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
