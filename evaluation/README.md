# Evaluation

This directory contains benchmark-specific evaluation workflows for predictions
produced by `inference/`.

| Benchmark | Directory | Main Entry |
| --- | --- | --- |
| BrowseComp | [`browsecomp/`](browsecomp/) | `bash run_judge.sh` |
| BrowseComp-Plus | [`browsecomp_plus/`](browsecomp_plus/) | `bash run_eval_quest.sh` |
| GAIA | [`gaia/`](gaia/) | `bash run_judge.sh` |
| HLE | [`hle/`](hle/) | `bash run_judge.sh` |
| DeepResearch Bench | [`drbench/`](drbench/) | `python convert_to_eval_format.py`, then `bash run_benchmark.sh` |
| LiveResearchBench | [`liveresearchbench/`](liveresearchbench/) | See the benchmark-specific README and scripts |
| Mind2Web2 | [`Mind2Web2/`](Mind2Web2/) | See the benchmark-specific README and scripts |
| WideSearch | [`widesearch/`](widesearch/) | `bash run_eval_quest.sh` |

For a new run, update the target result directory, dataset path, model or run
name, judge model, worker count, and judge credentials in the corresponding
benchmark script.

## Judge Prompt Source

BrowseComp and BrowseComp-Plus use the same A/B/C-style answer-matching judge
prompt adapted from Anthropic's Claude Opus 4.5 system card:
[`Claude Opus 4.5 System Card`](https://www-cdn.anthropic.com/bf10f64990cfda0ba858290be7b8cc6317685f47.pdf).
The prompt asks the judge to choose whether the sample answer matches the ground
truth answer, does not match it, or is effectively an "I don't know" response.
For scoring, choice `(A)` is treated as correct and choices `(B)` or `(C)` are
treated as incorrect.

## GAIA

GAIA evaluation lives under [`gaia/`](gaia/). The text-only subset is bundled as
[`gaia-text-only-103.jsonl`](gaia/gaia-text-only-103.jsonl). Before judging,
unzip `gaia-103-org.zip` in the same directory, then update `TARGET_DIRS`,
`DATASET_PATH`, `WORKERS`, and judge credentials in `run_judge.sh`.

```bash
cd evaluation/gaia
bash run_judge.sh
```

## BrowseComp-Plus

The 130-query QUEST input file is bundled as
[`browsecomp_plus_quest_130.jsonl`](browsecomp_plus/browsecomp_plus_quest_130.jsonl).
The judge requires the decrypted ground-truth file, defaulting to
`data/browsecomp_plus/browsecomp_plus_decrypted.jsonl`; override it with
`GROUND_TRUTH` when needed.

```bash
cd evaluation/browsecomp_plus
bash run_eval_quest.sh
```

## WideSearch

The English 100-query QUEST input file is bundled as
[`widesearch_en_input.jsonl`](widesearch/widesearch_en_input.jsonl). The eval
script defaults to `WIDESEARCH_LANGS=en`, auto-detects the latest compatible
run, and outputs under `inference/outputs/widesearch/`.

```bash
cd evaluation/widesearch
bash run_eval_quest.sh
```
