# Open-ended Task Generation Workflow

This directory contains a stripped-down task-generation pipeline for longform question generation.

The overall workflow is:

1. Generate raw trajectories
2. Extract proposed questions from JSON trajectory files
3. Generate evaluation criteria
4. Polish evaluation criteria
5. Generate reference answers
6. Collect trajectory from a teacher model
7. Refine final answers
8. Extract polished answers


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

Reference Generation:
- `ref_gen/run.sh`
- `ref_gen/run_multi_react_ref_gen.py`
- `ref_gen/react_agent_ref_gen.py`
- `ref_gen/prompt_ref_gen.py`

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
export TRAJ_DIR=/path/to/openended_trajectories
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
python extract_proposed_qa.py \
  --input_dir /path/to/openended_trajectories \
  --output_file /path/to/openended_outputs/proposed_qa.jsonl
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
python longform_rubric/generate_criteria.py \
  --input_file /path/to/openended_outputs/proposed_qa.jsonl \
  --output_file /path/to/openended_outputs/criteria.jsonl
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
python longform_rubric/polish_criteria.py \
  --input_file /path/to/openended_outputs/criteria.jsonl \
  --output_file /path/to/openended_outputs/polished_criteria.jsonl
```

### Step 5: Generate reference answers

Entry point:
- `ref_gen/run.sh`

What it does:
- Sets environment variables for API keys and model configuration
- Runs `run_multi_react_ref_gen.py` to generate reference answers using a powerful research agent
- The agent uses search and visit tools to conduct multi-turn research
- Outputs JSONL files with reference answers for each question

Important environment variables in `ref_gen/run.sh`:
- `DATASET`: Input dataset file path (default: `extracted_questions.jsonl`)
- `OUTPUT_PATH`: Output file path (default: `inference_results/ref.jsonl`)
- `ROLLOUT_COUNT`: Number of answer for each question (default: 1)
- `MAX_WORKERS`: Number of concurrent workers (default: 10)

Typical command:

```bash
export DATASET=/path/to/openended_outputs/proposed_qa.jsonl
export OUTPUT_PATH=/path/to/openended_outputs/reference_answers
bash ref_gen/run.sh
```

### Step 6: Collect trajectory from a teacher model

Same as objective task.

### Step 7: Refine final answers

Entry point:
- `polish_answer.py`

What it does:
- Reads trajectory files from Step6
- Regenerates and polishes the final answers using a more powerful model
- Outputs refined answers with inline URLs for nontrivial claims

Required arguments:
- `--files_to_polish`: Input directory containing trajectory files (example: `/memory_logs`)
- `--output_dir`: Output directory for refined answers (example: `/results`)

Optional arguments:
- `--iter`: Iteration number filter (default: None, process all iterations)
- `--max-workers`: Maximum number of worker threads (default: 100)

Typical command:

```bash
python polish_answer.py \
  --files_to_polish /path/to/teacher_model_logs \
  --output_dir /path/to/openended_outputs/refined_answers
```

### Step 8: Extract polished answers

Entry point:
- `extract_polished_answer.py`

What it does:
- Reads `iter*_replace.jsonl` files from Step 7 output directory
- Extracts polished answers (prefers `replace_answer` when `replace_status == "Success"`, otherwise falls back to original answer)
- Outputs a JSONL file with `{"question", "answer", "iter"}` format

Required arguments:
- `--output_dir`: Directory containing `iter*_replace.jsonl` files

Optional arguments:
- `--input_file`: Single input file (overrides `--output_dir` mode)
- `--output_file`: Custom output file path (default: `<output_dir>/extracted_answers_all_iters.jsonl`)

Typical command:

```bash
python extract_polished_answer.py \
  --output_dir /path/to/openended_outputs/refined_answers \
  --output_file /path/to/openended_outputs/final_answers.jsonl
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
