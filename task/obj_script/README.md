# Generate Obj Scripts from Formatted Obj Tasks

This directory is used to generate one Python script per formatted obj task JSON file.

Core flow:
1. Prepare formatted obj task files (`*.json`).
2. Run `task/obj_script/obj_script_generation.py`.
3. Get one generated `tree2py_*.py` script for each task.

## Step 1: Prepare Environment

From repo root:

```bash
cd /fs/ess/PAA0201/zilu/QUEST
python -m pip install litellm tqdm
```

Set your API key (recommended):

```bash
export OPENAI_API_KEY="YOUR_REAL_KEY"
```

Note:
- `obj_script_generation.py` currently contains:
  `os.environ["OPENAI_API_KEY"] = "your_api_key"`
- Replace `"your_api_key"` in the script with your real key, or remove that line and rely on the exported environment variable.

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
python task/obj_script/obj_script_generation.py \
  --input /path/to/formatted_tasks \
  --template /fs/ess/PAA0201/zilu/QUEST/task/obj_script/generation_prompt.md \
  --output /path/to/output_obj_scripts \
  --concurrency 20
```

### Option B: Run on one JSON file

```bash
python task/obj_script/obj_script_generation.py \
  --input /path/to/formatted_task.json \
  --template /fs/ess/PAA0201/zilu/QUEST/task/obj_script/generation_prompt.md \
  --output /path/to/output_obj_scripts \
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

