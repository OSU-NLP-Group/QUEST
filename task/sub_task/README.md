# `sub_task` Workflow

This directory contains a stripped-down task-generation pipeline for longform question generation.

The overall workflow is:

1. Generate raw trajectories

## Files

Generation:
- `run_generate_tasks_longform.sh`
- `generate_longform_tasks.py`
- `generation_agent_longform.py`
- `generation_prompt_longform.py`
- `tool_search.py`
- `tool_visit.py`

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

Typical command:

```bash
bash run_generate_tasks_longform.sh
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
