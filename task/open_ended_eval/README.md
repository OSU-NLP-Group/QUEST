# Open-Ended Evaluation

This directory runs rubric-based evaluation for QUEST open-ended tasks. It
compares generated answers against reference answers using task-specific
criteria and writes document-level scores plus detailed criterion judgments.

## Overview

High-level flow:

```text
load criteria -> load answers -> load reference answers
-> score each criterion -> aggregate weighted scores -> write results
```

Main entrypoints:

| File | Purpose |
| --- | --- |
| `run_eval.sh` | Evaluation launcher with endpoint checks |
| `evaluate_criteria_args_parallel_open_ended.py` | Parallel rubric-evaluation script |
| `eva_open_ended.py` | Evaluation prompt templates |

## Run Evaluation

Start one or more OpenAI-compatible judge endpoints, then configure
`endpoints.conf` in this directory:

```bash
HOSTNAME_LIST=localhost
PORTS=8000,8001
```

From the repository root:

```bash
cd task/open_ended_eval
PROMPT_TO_EVAL=/path/to/polished_criteria.jsonl \
ANSWER_TO_EVAL=/path/to/final_answers.jsonl \
REF_TO_EVAL=/path/to/reference_answers.jsonl \
OUTPUT_FILE=/path/to/eval_results.jsonl \
MODEL_NAME=eval_model \
bash run_eval.sh
```

You can also run the Python script directly:

```bash
python evaluate_criteria_args_parallel_open_ended.py \
  --model eval_model \
  --prompt_to_eval /path/to/polished_criteria.jsonl \
  --answer_to_eval /path/to/final_answers.jsonl \
  --ref_to_eval /path/to/reference_answers.jsonl \
  --output_file /path/to/eval_results.jsonl \
  --hostname_list localhost \
  --ports 8000,8001 \
  --max_workers 300
```

## Configuration

Common variables for `run_eval.sh`:

| Variable | Purpose |
| --- | --- |
| `PROMPT_TO_EVAL` | Criteria JSONL from open-ended task generation |
| `ANSWER_TO_EVAL` | Candidate answers to evaluate |
| `REF_TO_EVAL` | Reference answer JSONL, or comma-separated reference files |
| `OUTPUT_FILE` | Evaluation result JSONL |
| `MODEL_NAME` | Judge model name exposed by the endpoint |
| `MAX_WORKERS` | Document-level evaluation concurrency |
| `SERVER_ENDPOINTS_FILE` | Endpoint config file, default `endpoints.conf` |

`run_eval.sh` checks endpoint health before launching evaluation. Endpoint
routing can be updated through `endpoints.conf`; changes are picked up during
runtime by the Python evaluator.

## Outputs

Each output row includes the evaluated query, final score, document and
reference scores, dimension-level ratios, dimension weights, and detailed
criterion judgments.
