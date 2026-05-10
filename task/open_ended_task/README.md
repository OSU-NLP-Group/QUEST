# Open-Ended Task Generation

This directory runs the QUEST open-ended task generation pipeline. It produces
longform research tasks, builds evaluation criteria, generates reference
answers, and exports final open-ended QA data.

## Overview

High-level flow:

```text
generate longform tasks -> extract proposed QAs -> generate criteria -> generate reference answers -> refine final answers
-> extract final answers
```

Main entrypoints:

| Stage | Command | Output |
| --- | --- | --- |
| Generate longform tasks | `bash run_generate_tasks_longform.sh` | Raw trajectory JSON files |
| Extract proposed QAs | `python extract_proposed_qa.py` | JSONL task set |
| Generate criteria | `python longform_rubric/generate_criteria.py` | Draft criteria JSONL |
| Generate reference answers | `bash ref_gen/run.sh` | Reference-answer JSONL files |
| Refine final answers | `python polish_answer.py` | Refined answer files |
| Extract final answers | `python extract_polished_answer.py` | Final open-ended QA JSONL |

## Run Pipeline

From the repository root:

```bash
cd task/open_ended_task
bash run_generate_tasks_longform.sh
python extract_proposed_qa.py \
  --input_dir /path/to/openended_trajectories \
  --output_file /path/to/openended_outputs/proposed_qa.jsonl
python longform_rubric/generate_criteria.py \
  --input_file /path/to/openended_outputs/proposed_qa.jsonl \
  --output_file /path/to/openended_outputs/criteria.jsonl
DATASET=/path/to/openended_outputs/proposed_qa.jsonl \
OUTPUT_PATH=/path/to/openended_outputs/reference_answers \
bash ref_gen/run.sh
python polish_answer.py \
  --files_to_polish /path/to/teacher_model_logs \
  --output_dir /path/to/openended_outputs/refined_answers
python extract_polished_answer.py \
  --output_dir /path/to/openended_outputs/refined_answers \
  --output_file /path/to/openended_outputs/final_answers.jsonl
```

The default trajectory output root is:

```text
./outputs/openended_trajectories/
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
| `DEEPRESEARCH_AWS_ACCESS_KEY_ID`, `DEEPRESEARCH_AWS_SECRET_ACCESS_KEY`, `DEEPRESEARCH_AWS_REGION_NAME` | Bedrock generation credentials |

Path variables can be overridden from the shell:

```bash
TRAJ_DIR=/path/to/openended_trajectories \
bash run_generate_tasks_longform.sh
```

Search and visit caches are enabled by default and use local files under the
repository-level `database/` directory. If prebuilt databases are not present,
the cache files are created automatically during runs. Override these only when
you want to reuse or relocate caches:

| Variable | Default |
| --- | --- |
| `VISIT_CACHE_FILE` | `<repo>/database/visit_cache.db` |
| `SEARCH_CACHE_FILE` | `<repo>/database/search_cache.db` |
| `VISIT_CACHE_ENABLED`, `SEARCH_CACHE_ENABLED` | `true` |
| `VISIT_CACHE_RESUME`, `SEARCH_CACHE_RESUME` | `true` |

Reference-answer generation uses the same search and visit cache configuration.
Common reference-generation variables are:

| Variable | Purpose |
| --- | --- |
| `DATASET` | Input JSONL file, usually the proposed-QA output |
| `OUTPUT_PATH` | Reference-answer output prefix or directory |
| `MODEL_NAME` | Reference-generation model |
| `ROLLOUT_COUNT` | Number of answers per task |
| `MAX_WORKERS` | Reference-generation concurrency |

## Outputs

Typical generated files are organized as:

```text
outputs/openended_trajectories/
/path/to/openended_outputs/proposed_qa.jsonl
/path/to/openended_outputs/criteria.jsonl
/path/to/openended_outputs/reference_answers
/path/to/openended_outputs/refined_answers/
/path/to/openended_outputs/final_answers.jsonl
```

## Main Files

| File | Purpose |
| --- | --- |
| `run_generate_tasks_longform.sh` | Longform generation launcher and default environment configuration |
| `generate_longform_tasks.py` | Main longform task generation entrypoint |
| `generation_agent_longform.py` | Multi-turn generation agent |
| `generation_prompt_longform.py` | Open-ended task generation prompt |
| `tool_search.py`, `tool_visit.py` | Search and page-reading tools |
| `extract_proposed_qa.py` | Extract generated QAs into JSONL |
| `longform_rubric/generate_criteria.py` | Generate open-ended evaluation criteria |
| `ref_gen/run.sh` | Reference-answer generation launcher |
| `ref_gen/run_multi_react_ref_gen.py` | Multi-worker reference generation entrypoint |
| `polish_answer.py` | Refine answers from teacher-model trajectories |
| `extract_polished_answer.py` | Export refined answers into final JSONL |
