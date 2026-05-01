# Evaluation

This directory contains benchmark-specific evaluation workflows for predictions
produced by `inference/`.

| Benchmark | Directory | Main Entry |
| --- | --- | --- |
| BrowseComp | [`browsecomp/`](browsecomp/) | `bash run_judge.sh` |
| GAIA | [`gaia/`](gaia/) | `bash run_judge.sh` |
| HLE | [`hle/`](hle/) | `bash run_judge.sh` |
| DeepResearch Bench | [`drbench/`](drbench/) | `python convert_to_eval_format.py`, then `bash run_benchmark.sh` |
| Mind2Web2 | [`Mind2Web2/`](Mind2Web2/) | See the benchmark-specific README and scripts |

For a new run, update the target result directory, dataset path, model or run
name, judge model, worker count, and judge credentials in the corresponding
benchmark script.

## GAIA

GAIA evaluation lives under [`gaia/`](gaia/). The text-only subset is bundled as
[`gaia-text-only-103.jsonl`](gaia/gaia-text-only-103.jsonl). Before judging,
unzip `gaia-103-org.zip` in the same directory, then update `TARGET_DIRS`,
`DATASET_PATH`, `WORKERS`, and judge credentials in `run_judge.sh`.

```bash
cd evaluation/gaia
bash run_judge.sh
```
