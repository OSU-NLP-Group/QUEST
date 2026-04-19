# `sub_eval` - Document Quality Evaluation

This directory contains a document quality evaluation pipeline that compares answers against reference answers using a rubric-based scoring system.

## Overview

The evaluation system uses a model-based scorer that evaluates documents across multiple dimensions (comprehensiveness, insight, instruction following, readability) based on predefined criteria. It compares Document A (to evaluate) against Document B (reference) and produces relative scores.

## Files

- `eva_drb.py` - Contains the system prompt and user prompt templates for the evaluation model
- `evaluate_criteria_args_parallel_drb.py` - Main evaluation script with parallel processing support
- `run_eval.sh` - Bash script to run the evaluation with server health checks

## How It Works

1. Loads evaluation criteria (rubrics) from a JSONL file
2. Loads answers to evaluate from a JSONL file
3. Loads reference answers from a JSONL file
4. For each criterion, sends a request to the evaluation model to score both documents
5. Calculates weighted scores across all criteria and dimensions
6. Outputs final scores and detailed evaluation results

## Usage

### Configuration

Before running, configure your vLLM server endpoints in `endpoints.conf`:

```bash
HOSTNAME_LIST=localhost
PORTS=8000,8001,8002
```

### Running Evaluation

```bash
bash run_eval.sh
```

Or run the Python script directly:

```bash
python -u evaluate_criteria_args_parallel_drb.py \
    --model "eval_model" \
    --prompt_to_eval "path/to/criteria.jsonl" \
    --answer_to_eval "path/to/answers.jsonl" \
    --ref_to_eval "path/to/reference.jsonl" \
    --output_file "path/to/output.jsonl" \
    --hostname_list "localhost" \
    --ports "8000,8001" \
    --max_workers 300
```


## Hot-Swap Support

The script supports hot-swapping of vLLM endpoints. You can modify `endpoints.conf` at runtime without restarting the script. The changes take effect on the next evaluation batch.
