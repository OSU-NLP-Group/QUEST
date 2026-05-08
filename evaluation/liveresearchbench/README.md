# LiveResearchBench Evaluation

End-to-end DeepEval evaluation for predictions produced by
[`inference/scripts/run_react_infer_lrb.sh`](../../inference/scripts/run_react_infer_lrb.sh).

LiveResearchBench is a 100-task expert-curated benchmark for live, citation-grounded
deep-research reports; DeepEval is its multi-criteria LLM-as-judge framework
(presentation, consistency, citation, coverage, depth). See the upstream
[paper](https://arxiv.org/abs/2510.14240) and the vendored
[`LiveResearchBench/README.md`](LiveResearchBench/README.md) for the full method.

## Layout

```text
liveresearchbench/
├── prepare_lrb_questions.py       # Download LRB question jsonl from HuggingFace
├── prepare_lrb_full_questions.py  # Build a 100-qid LRB-Full jsonl (resume helper)
├── liveresearchbench_questions.jsonl   # Auto-generated; gated, not committed
└── LiveResearchBench/             # Vendored DeepEval framework
    ├── run_lrb_eval.sh            # Convert + preprocess + dual-judge + average
    ├── convert_to_lrb_format.py   # iter*.jsonl  ->  qid_<qid>_report.md
    ├── preprocess.py              # Build reports JSON index
    ├── main.py                    # DeepEval grader entrypoint
    ├── average_results.py         # Average Gemini + OpenAI judges
    ├── liveresearchbench/         # Grader implementations
    └── data/reference_reports/    # Reference reports for the depth (pairwise) grader
```

## Prerequisites

LiveResearchBench is a gated HuggingFace dataset
([`Salesforce/LiveResearchBench`](https://huggingface.co/datasets/Salesforce/LiveResearchBench)).
You need an HF account with access granted, plus an `HF_TOKEN`.

Add the token (and judge keys) to `inference/api_config.yaml` under `common:`:

```yaml
common:
  HF_TOKEN: hf_...
  OPENAI_API_KEY: sk-...
  GEMINI_API_KEY: AIza...
  # Optional, for vertexai-based Gemini calls (run `gcloud auth application-default login` first):
  VERTEXAI_PROJECT: <your-gcp-project>
```

`run_lrb_eval.sh` reuses the same `api_config.yaml` as inference.

Install the DeepEval dependencies (separate from QUEST runtime):

```bash
cd evaluation/liveresearchbench/LiveResearchBench
uv venv && source .venv/bin/activate && uv sync
# or: pip install -e .
```

## Inference (predictions)

Produces `iter1.jsonl` / `iter2.jsonl` / `iter3.jsonl` rollouts:

```bash
cd inference
bash scripts/run_react_infer_lrb.sh
```

The script auto-downloads `liveresearchbench_questions.jsonl` from HuggingFace
on first run via `prepare_lrb_questions.py` (uses `HF_TOKEN`).

Default LRB inference settings (override via env):

| Variable | Default | Notes |
| --- | --- | --- |
| `MEMORY_THRESHOLD` | `80000` | LRB outputs are long; budget high |
| `LLM_MAX_TOKENS` | `10000` | Per-step generation cap |
| `MAX_TURN` | `400` | Long horizon |
| `ROLLOUT_COUNT` | `3` | iter1/2/3 outputs |
| `OUTPUT_PATH` | `inference/outputs/lrb/results` | Inference puts iter*.jsonl under `${OUTPUT_PATH}/<MODEL_NAME>/<dataset_basename>/` |

## Evaluation

After inference, from this directory:

```bash
cd evaluation/liveresearchbench/LiveResearchBench
INFER_BASE_DIR=/path/to/outputs/lrb/results/deepresearch/liveresearchbench_questions \
IDENTIFIER=quest-30b-a3b \
bash run_lrb_eval.sh
```

`run_lrb_eval.sh` does:

1. **Convert** `iter*.jsonl` → `model_outputs/<IDENTIFIER>-iter{1,2,3}/qid_*_report.md`
   (`convert_to_lrb_format.py`).
2. **Preprocess** the report tree into a single `extracted_reports/reports_<IDENTIFIER>.json`
   index (`preprocess.py`, with `--use-realtime` to substitute today's date into LRB queries).
3. **Grade** with OpenAI/Azure OpenAI (`gpt-5-mini`) on
   `presentation,consistency,citation,coverage,depth` (`main.py`).
4. Optionally enable Gemini with `RUN_GEMINI=true` and average the two judges into `results/reports_<IDENTIFIER>_averaged/summary_multi_judge.json`
   (`average_results.py`).

Re-running the same command resumes from incremental jsonl saves; pass `--fresh`
to force re-convert + re-preprocess.

| Override | Default | Notes |
| --- | --- | --- |
| `INFER_BASE_DIR` | `inference/outputs/lrb/results/deepresearch/liveresearchbench_questions` | Dir containing `iter*.jsonl` |
| `IDENTIFIER` | `quest-run` | Becomes `<IDENTIFIER>-iter{1,2,3}` model name |
| `ITERS` | `1 2 3` | Which rollouts to grade |
| `CRITERIA` | `presentation,consistency,citation,coverage,depth` | Subset for ablations |
| `GEMINI_JUDGE` | `gemini-2.5-pro` | |
| `OPENAI_JUDGE` | `gpt-5-mini` | |
| `QUESTIONS_FILE` | `liveresearchbench_questions.jsonl` (auto-generated) | |
| `MODEL_OUTPUTS_DIR` | `LiveResearchBench/model_outputs` | |
| `EXTRACTED_DIR` | `LiveResearchBench/extracted_reports` | |

## Output

```text
LiveResearchBench/results/
└── reports_<IDENTIFIER>_graded_<provider>_<model>/
    ├── incremental/                       # Per-criterion jsonl streams (resume source)
    ├── summary_<timestamp>.json           # Aggregate stats per model + overall
    └── detailed_results_<timestamp>.json  # Per-report grader outputs
LiveResearchBench/results/
└── reports_<IDENTIFIER>_averaged/
    └── summary_multi_judge.json           # Final dual-judge averaged scores
```

## Notes

- LRB is **dynamic**: `{{current_year}}`/`{{date}}` placeholders are substituted with
  today's date at both inference and preprocess time, so re-running on a different
  day re-evaluates against fresh queries. The `--questions-file` override on
  `preprocess.py` keeps inference and eval question text aligned when they run
  on different days.
- The depth (pairwise) grader compares against the bundled
  [`LiveResearchBench/data/reference_reports/`](LiveResearchBench/data/reference_reports).
- The dataset is released under CC-BY-NC 4.0; the eval code is Apache 2.0.
