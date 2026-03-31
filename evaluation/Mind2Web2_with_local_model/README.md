# Mind2Web2 with Local Model

This guide follows the exact order you requested:
1) start models with `deploy.sh`,
2) configure nodes in `config/vllm_nodes.txt`,
3) run evaluation with `run.sh`.

## Prerequisites

- Python `3.10+`
- `vllm` installed
- Model path exists:
  - `./Qwen3-4B-Instruct-2507`
- Answers already prepared in:
  - `./answers/<agent_name>/<task_id>/answer_*.md`

## Step 1: Start model servers on worker nodes

Run this on each worker node that should host vLLM:

```bash
cd /QUEST/evaluation/Mind2Web2_with_local_model
bash deploy.sh
```

`deploy.sh` starts one vLLM server per GPU (`0..3`) on ports `6000..6003`.

## Step 2: Configure endpoint nodes

On the controller node, edit:

`/QUEST/evaluation/Mind2Web2_with_local_model/config/vllm_nodes.txt`

Recommended format: one host per line (ports auto-expand from `VLLM_PORTS`):

```txt
a0012
a0015
a0016
```

Also supported:
- `host:port`
- `http://host:port/v1`
- comma-separated entries

## Step 3: Run evaluation from controller

```bash
cd /QUEST/evaluation/Mind2Web2_with_local_model

export GOOGLE_MAPS_API_KEY="<your_google_maps_key>"
export JINA_API_KEYS="<your_jina_key>"

bash run.sh
```

## Outputs

- Run log: `log/<run_id>/run.log`
- Evaluation results: `eval_results/<agent_name>/...`
- Summary: `eval_results/<agent_name>/evaluation_summary.json`
