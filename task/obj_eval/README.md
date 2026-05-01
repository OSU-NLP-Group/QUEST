# Generate Obj Eval Scripts from Formatted Obj Tasks

This directory is used to generate one Python script per formatted obj task JSON file.

Core flow:
1. Prepare formatted obj task files (`*.json`).
2. Run `task/obj_eval/obj_eval_generation.py`.
3. Get one generated `tree2py_*.py` script for each task.

## Step 1: Prepare Environment

```bash
python -m pip install litellm tqdm
```

Configure the model provider credentials required by LiteLLM before running the
generator.

## Step 2: Prepare Input (Formatted Obj Tasks)

Input should be:
- a single formatted task JSON file, or
- a directory containing many formatted task JSON files.

Each JSON should include at least:
- `proposed_question`
- `rubric_tree_analysis_refined.formatted_tree`

## Step 3: Run Generation

### Option A: Run on a directory

```bash
python task/obj_eval/obj_eval_generation.py \
  --input /path/to/objective_trajectories/formatted \
  --template task/obj_eval/generation_prompt.md \
  --output /path/to/objective_verifiers \
  --concurrency 20
```

### Option B: Run on one JSON file

```bash
python task/obj_eval/obj_eval_generation.py \
  --input /path/to/formatted_task.json \
  --template task/obj_eval/generation_prompt.md \
  --output /path/to/objective_verifiers \
  --concurrency 1
```

## Step 4: Check Outputs

In `--output` directory:
- success file format: `tree2py_<input_json_stem>.py`
- if model returns empty code block, raw response is saved as:
  `tree2script_formatted_<input_json_stem>_raw.md`

The script also prints:
- total/success/failed counts
- prompt/completion/reasoning/total token usage
