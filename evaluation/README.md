# Evaluation Notes

This directory contains evaluation code and assets for the main benchmarks used in this project:

- `browsecomp`
- `hle`
- `drbench`
- `Mind2Web2`

In general, the evaluation workflow is:

1. Run inference first and make sure predictions are written to the expected result directory.
2. Update the evaluation script so `TARGET_DIR` or model name points to the new run.
3. Set the required judge or API credentials.
4. Run the benchmark-specific evaluation script.

## BrowseComp

Directory: [`browsecomp`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/browsecomp)

Main evaluation script:

- [`run_judge.sh`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/browsecomp/run_judge.sh)

Steps:

1. Make sure inference results exist under the BrowseComp result directory.
2. Edit `TARGET_DIRS` in `run_judge.sh` to point to the new result folder.
3. Check `DATASET_PATH`. By default it uses the official remote BrowseComp CSV, but it can also be changed to a local file.
4. Set `JUDGE_MODEL_NAME` and `JUDGE_OPENAI_API_KEY` or switch to another supported provider in the script.
5. Run:

```bash
bash run_judge.sh
```

Output:

- Judging is executed through [`eval.py`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/browsecomp/eval.py)
- Results are written under the target result directory you provide

## HLE

Directory: [`hle`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/hle)

Main evaluation script:

- [`run_judge.sh`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/hle/run_judge.sh)

Steps:

1. Make sure inference results exist under the HLE result directory.
2. Edit `TARGET_DIRS` in `run_judge.sh`.
3. Check `DATASET_PATH` and `WORKERS`.
4. Set `JUDGE_MODEL_NAME` and `JUDGE_OPENAI_API_KEY`.
5. Run:

```bash
cd /fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/hle
bash run_judge.sh
```

This runs [`judge.py`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/hle/judge.py).

## DeepResearch Bench

Directory: [`drbench`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/drbench)

Main evaluation package:

- [`eval/deep_research_bench`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/drbench/eval/deep_research_bench)

Typical evaluation script:

- [`convert_to_eval_format.py`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/drbench/eval/deep_research_bench/convert_to_eval_format.py)
- [`run_benchmark.sh`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/drbench/eval/deep_research_bench/run_benchmark.sh)

Steps:

1. First convert your inference outputs into the DeepResearch Bench evaluation format with [`convert_to_eval_format.py`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/drbench/eval/deep_research_bench/convert_to_eval_format.py).
2. Make sure the converted file is placed under `data/test_data/raw_data/<model_name>.jsonl`.
3. Edit [`run_benchmark.sh`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/drbench/eval/deep_research_bench/run_benchmark.sh) so the target model name matches your converted file.
4. Set the required evaluation credentials and model configuration.
5. Run:

```bash
cd /fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/drbench/eval/deep_research_bench
python convert_to_eval_format.py
bash run_benchmark.sh
```

Notes:

- The current default setup does not enable fact checking.
- The default evaluation flow is the benchmark scoring flow from `run_benchmark.sh`.

Output:

- Benchmark results are written under the benchmark results directory, typically `results/`

If needed, read the benchmark's own upstream documentation in [`eval/deep_research_bench/README.md`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/drbench/eval/deep_research_bench/README.md).

## Mind2Web2

Directory: [`Mind2Web2/Mind2Web2_with_local_model`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/Mind2Web2/Mind2Web2_with_local_model)

Current status:

- Mind2Web2 evaluation is not currently used in the active workflow.
- The local code is kept here for reference only.

## What Usually Needs To Be Updated

Before evaluating a new run, the fields that usually need to be changed are:

- Target result directory such as `TARGET_DIRS`
- Dataset path such as `DATASET_PATH`
- Model or run name such as `TARGET_MODEL`
- Judge model and API keys
- Worker count
- Endpoint or node configuration for local serving benchmarks
