# BrowseComp-Plus Evaluation

This directory contains the BrowseComp-Plus inference input, local search agent
utilities, and judge conversion/evaluation scripts.

## Files

| File | Purpose |
| --- | --- |
| `browsecomp_plus_quest_130.jsonl` | 130-query QUEST inference input |
| `run_eval_quest.sh` | Convert QUEST iter files and run the BrowseComp-Plus judge |
| `scripts_evaluation/evaluate_with_openai.py` | OpenAI-compatible judge runner |
| `search_agent/` | BrowseComp-Plus prompts, FAISS search, and local tool helpers |

## Inference

Run from `inference/`:

```bash
bash scripts/run_react_infer_bcp.sh
```

The launcher defaults to `browsecomp_plus_quest_130.jsonl`, disables live visit
search, and uses Qwen3-Embedding-8B FAISS retrieval. The FAISS index is not
committed; set `FAISS_INDEX_PATH` if it is not available at the default
`data/browsecomp_plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl`.

## Evaluation

Run from this directory:

```bash
bash run_eval_quest.sh
```

The script auto-detects the latest compatible QUEST output directory by default.
Set `RUN_ROOT` when evaluating a specific run.
The decrypted ground truth is not committed; set `GROUND_TRUTH` if it is not
available at `data/browsecomp_plus/browsecomp_plus_decrypted.jsonl`.

The BrowseComp-Plus judge prompt is the same A/B/C-style answer-matching prompt
used by BrowseComp in this repo. It is adapted from Anthropic's
[`Claude Opus 4.5 System Card`](https://www-cdn.anthropic.com/bf10f64990cfda0ba858290be7b8cc6317685f47.pdf):
`(A)` means the response matches the ground truth, while `(B)` and `(C)` are
scored as incorrect.
