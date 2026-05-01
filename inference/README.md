# Inference

This directory runs QUEST inference against benchmark datasets using an existing
model endpoint.

## Setup

```bash
pip install -r requirements.txt
```

Configure external services and model endpoints before launching:

| File | Purpose |
| --- | --- |
| `api_config.yaml` | Search, visit, summary, and memory model credentials/configuration |
| `server_endpoints.conf` | Model serving hosts and ports used by the agent |

`react_agent.py` reloads `server_endpoints.conf` during execution, so endpoint
routing can be updated without restarting a run.

## Launch Scripts

Common entrypoints:

```bash
bash run_react_infer_bc.sh
bash run_react_infer_gaia.sh
bash run_react_infer_hle.sh
bash run_react_infer_drb.sh
bash run_react_infer_m2w2.sh
```

The staggered variants are useful when launching high-concurrency runs in
batches:

```bash
bash run_react_infer_bc_staggered.sh
bash run_react_infer_hle_staggered.sh
```

## Before A Run

Set or edit the following fields in the corresponding `run_react_infer_*.sh`
script:

| Variable | Purpose |
| --- | --- |
| `DATASET` | Input benchmark dataset file |
| `OUTPUT_PATH` | Directory for final prediction outputs |
| `TASK_LOG_DIR` | Directory for task logs and memory traces |
| `MODEL_PATH` | Model checkpoint or served model name |
| `MAX_TURN` | Maximum interaction turns per sample |
| `MAX_LLM_CALL_PER_RUN` | Maximum model calls per sample |
| `MAX_WORKERS` | Inference concurrency |
| `MEMORY_THRESHOLD` | Context or memory threshold before truncation/filtering |
| `LLM_MAX_TOKENS` | Generation token budget |
| `API_CONFIG_FILE` | API configuration file path |
| `SERVER_ENDPOINTS_FILE` | Endpoint configuration file path |
| `HOSTNAME_LIST`, `PORTS` | Endpoint hosts and ports when not using `server_endpoints.conf` |

## Script Defaults

| Script | Benchmark | Notable Defaults |
| --- | --- | --- |
| `run_react_infer_bc.sh` | BrowseComp | `MAX_TURN=400`, `MEMORY_THRESHOLD=80000`, `LLM_MAX_TOKENS=20000` |
| `run_react_infer_gaia.sh` | GAIA | `MAX_TURN=400`, `MEMORY_THRESHOLD=40000`, `LLM_MAX_TOKENS=16000` |
| `run_react_infer_hle.sh` | HLE | `MAX_TURN=200`, `MEMORY_THRESHOLD=80000`, `LLM_MAX_TOKENS=16000` |
| `run_react_infer_drb.sh` | DeepResearch Bench | `MAX_TURN=200`, `MEMORY_THRESHOLD=40000`, `LLM_MAX_TOKENS=10000` |
| `run_react_infer_m2w2.sh` | Mind2Web2 | `MAX_TURN=200`, `MEMORY_THRESHOLD=80000`, `LLM_MAX_TOKENS=16000` |

## Resume

The inference pipeline supports resume behavior. If a run is interrupted, rerun
the same script with the same output and log paths.

## Adding A Benchmark

Add a dataset file that matches the input format expected by `run_multi_react.py`,
then create or adapt a `run_react_infer_*.sh` launch script. Evaluation logic is
kept under [`../evaluation/`](../evaluation/).
