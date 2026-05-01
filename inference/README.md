# Inference

This directory runs QUEST inference against benchmark datasets using an existing
model endpoint.

## Overview

The inference pipeline provides a ReAct-style research agent with search,
scholar, visit, memory, and Python tools. Benchmark launch scripts live in this
directory, while benchmark-specific judging code lives under
[`../evaluation/`](../evaluation/).

Supported benchmark launchers:

| Benchmark | Script | Default Dataset |
| --- | --- | --- |
| BrowseComp | `run_react_infer_bc.sh` | `../evaluation/browsecomp/browsecomp.jsonl` |
| GAIA | `run_react_infer_gaia.sh` | `../evaluation/gaia/gaia-text-only-103.jsonl` |
| HLE | `run_react_infer_hle.sh` | `../evaluation/hle/hle_text_only_130.jsonl` |
| DeepResearch Bench | `run_react_infer_drb.sh` | `../evaluation/drbench/deepresearch_bench_questions.jsonl` |
| Mind2Web2 | `run_react_infer_m2w2.sh` | Set `DATASET` to your Mind2Web2 test file |

## Setup

Complete the root-level environment setup first. Then configure external
services and model endpoints before launching:

| File | Purpose |
| --- | --- |
| `api_config.yaml` | Search, visit, summary, and memory model credentials/configuration |
| `server_endpoints.conf` | Model serving hosts and ports used by the agent |

`react_agent.py` reloads `server_endpoints.conf` during execution, so endpoint
routing can be updated without restarting a run.

## Run Inference

From this directory, run the benchmark-specific script after configuration:

```bash
cd inference
bash run_react_infer_<benchmark>.sh
```

For a new run, update the corresponding script or override the variables from
the shell:

| Variable | Purpose |
| --- | --- |
| `DATASET` | Input benchmark dataset file |
| `OUTPUT_PATH` | Directory for final prediction outputs |
| `TASK_LOG_DIR` | Directory for task logs and memory traces |
| `MODEL_PATH` | Model checkpoint or served model name |
| `MAX_TURN` | Maximum interaction turns per sample |
| `MAX_WORKERS` | Inference concurrency |
| `MEMORY_THRESHOLD` | Context or memory threshold before truncation/filtering |
| `LLM_MAX_TOKENS` | Generation token budget |
| `API_CONFIG_FILE` | API configuration file path |
| `SERVER_ENDPOINTS_FILE` | Endpoint configuration file path |
| `HOSTNAME_LIST`, `PORTS` | Endpoint hosts and ports when not using `server_endpoints.conf` |

Example:

```bash
DATASET=/path/to/benchmark.jsonl \
OUTPUT_PATH=/path/to/outputs/results \
TASK_LOG_DIR=/path/to/outputs/logs \
bash run_react_infer_bc.sh
```

## Defaults

| Benchmark | `MAX_TURN` | `MEMORY_THRESHOLD` | `LLM_MAX_TOKENS` |
| --- | --- | --- | --- |
| BrowseComp | `400` | `80000` | `20000` |
| GAIA | `400` | `40000` | `16000` |
| HLE | `200` | `96000` | `32000` |
| DeepResearch Bench | `200` | `16000` | `10000` |
| Mind2Web2 | `200` | `80000` | `16000` |

## Resume

The inference pipeline supports resume behavior. If a run is interrupted, rerun
the same script with the same output and log paths.

## Adding A Benchmark

Add a dataset file that matches the input format expected by `run_multi_react.py`,
then create or adapt a `run_react_infer_*.sh` launch script. Keep scoring and
judge logic under [`../evaluation/`](../evaluation/).
