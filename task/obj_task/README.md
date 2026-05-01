# Objective Task Generation

This directory runs the QUEST objective task generation pipeline. It produces
research trajectories, formats them for rubric-tree verification, filters
accepted tasks, and exports accepted objective questions.

## Overview

High-level flow:

```text
generate trajectories -> merge rubric predictions -> format verifier inputs
-> refine rubric trees -> verify rubric trees -> extract accepted questions
```

Main entrypoints:

| Stage | Command | Output |
| --- | --- | --- |
| Generate trajectories | `bash run_generate_tasks.sh` | Raw trajectory JSON files |
| Merge rubric refinements | `python merge_rubric_predictions.py` | Updated trajectory JSON files |
| Format verifier inputs | `python format_trajectories.py` | `formatted/` verifier inputs |
| Refine rubric trees | `bash run_refine_rubric_trees.sh` | `formatted/refined/` verifier inputs |
| Verify rubric trees | `bash run_verify_rubric_trees.sh` | Verification logs and accepted trajectories |
| Extract accepted questions | `python extract_proposed_questions.py` | JSONL question set |

## Run Pipeline

From this directory:

```bash
cd task/obj_task
bash run_generate_tasks.sh
python merge_rubric_predictions.py
python format_trajectories.py
bash run_refine_rubric_trees.sh
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

Search and visit caches are enabled by default and use local files under the
repository-level `database/` directory. Override these only when you want to
reuse or relocate caches:

| Variable | Default |
| --- | --- |
| `VISIT_CACHE_FILE` | `<repo>/database/visit_cache.db` |
| `SEARCH_CACHE_FILE` | `<repo>/database/search_cache.db` |
| `VISIT_CACHE_ENABLED`, `SEARCH_CACHE_ENABLED` | `true` |
| `VISIT_CACHE_RESUME`, `SEARCH_CACHE_RESUME` | `true` |

Rubric refinement uses its own model configuration:

| Variable | Purpose |
| --- | --- |
| `REFINE_MODEL_NAME` | Rubric refine model, default `openai/gpt-5.2` |
| `REFINE_OPENAI_API_KEY` | OpenAI-compatible key for refine model |
| `REFINE_AZURE_API_KEY`, `REFINE_AZURE_API_BASE`, `REFINE_AZURE_API_VERSION` | Azure/OpenAI-compatible refine endpoint |
| `REFINE_AWS_ACCESS_KEY_ID`, `REFINE_AWS_SECRET_ACCESS_KEY`, `REFINE_AWS_REGION_NAME` | Bedrock refine credentials |
| `REFINE_WORKERS` | Refine concurrency |

The refine stage writes accepted or repaired formatted files to:

```text
outputs/objective_trajectories/formatted/refined/
```

If `formatted/refined/` exists, `run_verify_rubric_trees.sh` uses it by default.
Otherwise, it verifies the original `formatted/` directory. The question
extraction script follows the same convention for accepted trajectories.

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
outputs/objective_trajectories/formatted/refined/
outputs/objective_trajectories/formatted/refined/verifier/
outputs/objective_trajectories/formatted/refined/verifier/accepted_trajectories/
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
| `run_refine_rubric_trees.sh` | Rubric-tree refine launcher |
| `refine_rubric_trees.py` | Refine formatted rubric trees before verification |
| `refine_rubric_prompt.py` | Rubric-tree refine prompt |
| `run_verify_rubric_trees.sh` | Rubric verification launcher |
| `verify_rubric_trees.py` | Rubric-tree verifier |
| `extract_proposed_questions.py` | Extract accepted questions into JSONL |
