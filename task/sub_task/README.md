# `sub_task` Workflow

This directory contains a stripped-down task-generation pipeline for longform question generation.

The overall workflow is:

1. Generate raw trajectories
2. Extract proposed questions from JSON trajectory files
3. Generate evaluation criteria
4. Polish evaluation criteria

## Files

Generation:
- `run_generate_tasks_longform.sh`
- `generate_longform_tasks.py`
- `generation_agent_longform.py`
- `generation_prompt_longform.py`
- `tool_search.py`
- `tool_visit.py`

Extraction:
- `extract_proposed_qa.py`

Criteria Generation:
- `longform_rubric/generate_criteria.py`

Criteria Polishing:
- `longform_rubric/polish_criteria.py`

## Step-by-step

### Step 1: Generate raw trajectories

Entry point:
- `run_generate_tasks_longform.sh`

What it does:
- Activates the `deepresearch` conda environment
- Sets the model / cache / web access environment variables
- Runs `generate_longform_tasks.py`
- `generate_longform_tasks.py` uses `generation_agent_longform.py` plus `tool_search.py` and `tool_visit.py`
- Output trajectories are written under `TRAJ_DIR`

Important environment variables in `run_generate_tasks_longform.sh`:
- `SERPER_KEY_ID`
- `SUMMARY_AZURE_API_KEY`
- `SUMMARY_AZURE_API_BASE`
- `SUMMARY_AZURE_API_VERSION`
- `SUMMARY_MODEL_NAME`
- `DEEPRESEARCH_AWS_ACCESS_KEY_ID`
- `DEEPRESEARCH_AWS_SECRET_ACCESS_KEY`
- `DEEPRESEARCH_AWS_REGION_NAME`
- `DEEPRESEARCH_MODEL_NAME`
- `SAVE_TRAJ`
- `TRAJ_DIR`
- `JINA_API_KEYS`
- `VISIT_CACHE_ENABLED`
- `VISIT_CACHE_FILE`
- `VISIT_CACHE_RESUME`
- `SEARCH_CACHE_ENABLED`
- `SEARCH_CACHE_FILE`
- `SEARCH_CACHE_RESUME`

Important parameters in `generate_longform_tasks.py`:
- `num_iterations`: Total number of tasks to generate (default: 10)
- `workers`: Concurrency control - number of workers running simultaneously (default: 1)

Typical command:

```bash
bash run_generate_tasks_longform.sh
```

### Step 2: Extract proposed questions from JSON trajectory files

Entry point:
- `extract_proposed_qa.py`

What it does:
- Reads JSON trajectory files from an input directory
- Extracts prediction fields
- Outputs a JSONL file with extracted data

Required arguments:
- `--input_dir`: Input directory containing JSON trajectory files
- `--output_file`: Output JSONL file path (default: `extracted_questions.jsonl`)

Typical command:

```bash
python extract_proposed_qa.py --input_dir /path/to/json/files --output_file extracted_questions.jsonl
```

### Step 3: Generate evaluation criteria

Entry point:
- `longform_rubric/generate_criteria.py`

What it does:
- Reads JSONL file containing extracted questions
- Generates dimension weights and evaluation criteria for each question
- Outputs a JSONL file with criteria for comprehensiveness, insight, instruction_following, and readability

Required arguments:
- `--input_file`: Input JSONL file path
- `--output_file`: Output JSONL file path

Typical command:

```bash
python longform_rubric/generate_criteria.py --input_file extracted_questions.jsonl --output_file criteria.jsonl
```

### Step 4: Polish evaluation criteria

Entry point:
- `longform_rubric/polish_criteria.py`

What it does:
- Reads JSONL file containing evaluation criteria
- Polishes and refines the criteria
- Outputs a JSONL file with polished criteria

Required arguments:
- `--input_file`: Input JSONL file path
- `--output_file`: Output JSONL file path

Optional arguments:
- `--num_threads`: Number of concurrent threads (default: 50)

Typical command:

```bash
python longform_rubric/polish_criteria.py --input_file criteria.jsonl --output_file polished_criteria.jsonl
```

## Notes

- Many scripts still contain hard-coded default paths. You will likely want to edit those before running on a new dataset.
- External Python dependencies are not vendored in this directory. You still need the runtime environment to provide packages such as:
  - `qwen_agent`
  - `litellm`
  - `openai`
  - `transformers`
  - `requests`
  - `json5`
