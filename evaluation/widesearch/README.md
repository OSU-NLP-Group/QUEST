# WideSearch Evaluation

This directory contains the WideSearch evaluation harness and QUEST-formatted
input files.

## Files

| File | Purpose |
| --- | --- |
| `widesearch_en_input.jsonl` | 100 English QUEST inference queries |
| `run_eval_quest_gpt5mini.sh` | Convert QUEST outputs and run WideSearch scoring |
| `run_widesearch_eval.py` | WideSearch evaluator entry point |
| `src/` | WideSearch data loading, metrics, judge, and utility code |

## Inference

Run from `inference/`:

```bash
bash scripts/run_react_infer_ws.sh
```

The launcher defaults to `WIDESEARCH_LANGS=en`, `widesearch_en_input.jsonl`, and
`gpt-5-mini` for summary, memory, and judging. Set `WIDESEARCH_LANGS=zh` or
`WIDESEARCH_LANGS=both` only when the matching input files are available.

## Evaluation

Run from this directory:

```bash
bash run_eval_quest_gpt5mini.sh
```

The script converts QUEST `iter*.jsonl` outputs into WideSearch response files,
then runs the scorer with `JUDGE_CONFIG=gpt-5-mini-eval` by default. Outputs are
written under `inference/outputs/widesearch/`.
