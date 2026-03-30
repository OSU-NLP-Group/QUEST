# `obj_task` Workflow

This directory contains a stripped-down task-generation and rubric-verification pipeline copied from the original DeepResearch inference codebase.

The overall workflow is:

1. Generate raw trajectories
2. Merge the refined rubric tree back into each trajectory
3. Convert trajectories into formatted verifier inputs
4. Run rubric-tree verification
5. Extract accepted proposed questions

## Files

Generation:
- `run_generate_tasks.sh`
- `generate_tasks.py`
- `generation_agent.py`
- `generation_prompts.py`
- `tool_search.py`
- `tool_visit.py`

Post-processing:
- `merge_rubric_predictions.py`
- `format_trajectories.py`
- `extract_proposed_questions.py`

Rubric verification:
- `run_verify_rubric_trees.sh`
- `verify_rubric_trees.py`

## Step-by-step

### Step 1: Generate raw trajectories

Entry point:
- `run_generate_tasks.sh`

What it does:
- Activates the `deepresearch` conda environment
- Sets the model / cache / web access environment variables
- Runs `generate_tasks.py`
- `generate_tasks.py` uses `generation_agent.py` plus `tool_search.py` and `tool_visit.py`
- Output trajectories are written under `TRAJ_DIR`

Important environment variables in `run_generate_tasks.sh`:
- `SERPER_KEY_ID`
- `SUMMARY_AZURE_API_KEY`
- `SUMMARY_AZURE_API_BASE`
- `SUMMARY_AZURE_API_VERSION`
- `SUMMARY_MODEL_NAME`
- `DEEPRESEARCH_AWS_CREDENTIALS`
- `DEEPRESEARCH_MODEL_NAME`
- `DEEPRESEARCH_OPENAI_API_KEY`
- `SAVE_TRAJ`
- `TRAJ_DIR`
- `MEMORY_ENABLED`
- `MEMORY_CONTEXT_THRESHOLD`
- `TASK_LOG_DIR`
- `MEMORY_TOKENIZER_PATH`
- `VISIT_SERVICE`
- `JINA_API_KEYS`
- `VISIT_CACHE_ENABLED`
- `VISIT_CACHE_FILE`
- `VISIT_CACHE_RESUME`
- `SEARCH_CACHE_ENABLED`
- `SEARCH_CACHE_FILE`
- `SEARCH_CACHE_RESUME`

Typical command:

```bash
bash run_generate_tasks.sh
```

### Step 2: Merge refined rubric predictions back into trajectories

Script:
- `merge_rubric_predictions.py`

What it does:
- Reads raw trajectory JSON files
- Replaces `prediction` with the second-last assistant message
- If the last assistant message contains a revised rubric tree, it injects that rubric tree back into the prediction
- Rebuilds the `answer` field from `<answer>...</answer>`

Typical use:
- Run this after generation if your trajectories include an extra rubric-refinement assistant turn

### Step 3: Build formatted verifier inputs

Script:
- `format_trajectories.py`

What it does:
- Reads trajectory JSON files
- Extracts the final question, constraints, solution, and rubric tree
- Computes rubric statistics such as depth and width
- Writes `_formatted.json` files for later verification

Expected result:
- A directory of formatted files ending in `_formatted.json`

### Step 4: Verify rubric trees

Entry point:
- `run_verify_rubric_trees.sh`

What it does:
- Runs `verify_rubric_trees.py` over a folder of `_formatted.json` files
- Calls the rubric-verifier model
- Writes:
  - verification result JSON
  - per-file prompt/response logs
  - `accepted_trajectories/`
  - `rubric_filter_report.json`

Important environment variables in `run_verify_rubric_trees.sh`:
- `RUBRIC_VERIFIER_AWS_ACCESS_KEY_ID`
- `RUBRIC_VERIFIER_AWS_SECRET_ACCESS_KEY`
- `RUBRIC_VERIFIER_AWS_REGION_NAME`
- `RUBRIC_VERIFIER_MODEL_NAME`
- `TRAJ_DIR`
- `LOG_DIR`
- `OUTPUT_FILE`

Typical command:

```bash
bash run_verify_rubric_trees.sh
```

### Step 5: Extract proposed questions from accepted trajectories

Script:
- `extract_proposed_questions.py`

What it does:
- Reads accepted trajectory JSON files
- Parses the JSON stored in `answer`
- Extracts `proposed_question`
- Writes a JSONL file with:
  - `question`
  - `answer`
  - `filename`

Use this after Step 4 if you want a clean dataset of accepted generated questions.

## Recommended execution order

If you want the full pipeline, the recommended order is:

1. `bash run_generate_tasks.sh`
2. `python merge_rubric_predictions.py`
3. `python format_trajectories.py`
4. `bash run_verify_rubric_trees.sh`
5. `python extract_proposed_questions.py`

## Notes

- Many scripts still contain hard-coded default paths. You will likely want to edit those before running on a new dataset.
- `run_verify_rubric_trees.sh` is independent from the generation-time summary model settings.
- External Python dependencies are not vendored in this directory. You still need the runtime environment to provide packages such as:
  - `qwen_agent`
  - `litellm`
  - `openai`
  - `transformers`
  - `requests`
  - `json5`
