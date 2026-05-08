# Inference

This directory runs QUEST inference against benchmark datasets using an existing
model endpoint.

## Overview

The inference pipeline provides a ReAct-style research agent with search,
scholar, visit, memory, and Python tools. Benchmark launch scripts live under
[`scripts/`](scripts/), while benchmark-specific judging code lives under
[`../evaluation/`](../evaluation/).

Supported benchmark launchers:

| Benchmark | Script | Default Dataset |
| --- | --- | --- |
| BrowseComp | `scripts/run_react_infer_bc.sh` | `../evaluation/browsecomp/browsecomp.jsonl` |
| BrowseComp-Plus | `scripts/run_react_infer_bcp.sh` | `../evaluation/browsecomp_plus/browsecomp_plus_quest_130.jsonl` |
| GAIA | `scripts/run_react_infer_gaia.sh` | `../evaluation/gaia/gaia-text-only-103.jsonl` |
| HLE | `scripts/run_react_infer_hle.sh` | `../evaluation/hle/hle_text_only_130.jsonl` |
| DeepResearch Bench | `scripts/run_react_infer_drb.sh` | `../evaluation/drbench/deepresearch_bench_questions.jsonl` |
| LiveResearchBench | `scripts/run_react_infer_lrb.sh` | `../evaluation/liveresearchbench/liveresearchbench_questions.jsonl` |
| Mind2Web2 | `scripts/run_react_infer_m2w2.sh` | Set `DATASET` to your Mind2Web2 test file |
| WideSearch | `scripts/run_react_infer_ws.sh` | `../evaluation/widesearch/widesearch_en_input.jsonl` |

## Setup

Complete the root-level environment setup first. Then configure external
services and model endpoints before launching:

| File | Purpose |
| --- | --- |
| `api_config.yaml` | Search, visit, summary, and memory model credentials/configuration |
| `server_endpoints.conf` | Model serving hosts and ports used by the agent |

`react_agent.py` reloads `server_endpoints.conf` during execution, so endpoint
routing can be updated without restarting a run.

`server_endpoints.conf` is the default model-server routing file. It is a simple
`KEY=value` file with comma-separated hosts and ports:

```text
HOSTNAME_LIST=localhost
PORTS=6000
```

For a multi-node or multi-port deployment, replace the placeholders with your
serving hosts and ports:

```text
HOSTNAME_LIST=node1,node2
PORTS=6000,6001,6002,6003
```

The agent treats the configured hosts and ports as candidate model endpoints and
hot-reloads the file before service calls. The script variable
`SERVER_ENDPOINTS_FILE` points to this file by default; set it to another config
path when you want to keep machine-specific endpoints outside the repo. If the
file is missing, scripts fall back to `HOSTNAME_LIST` and `PORTS` environment
variables.

Some benchmarks require external assets that are intentionally not committed:

| Benchmark | External Asset |
| --- | --- |
| BrowseComp-Plus | FAISS shards matching `FAISS_INDEX_PATH`, default `../data/browsecomp_plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl` |
| LiveResearchBench | Gated HuggingFace data; set `HF_TOKEN` if `liveresearchbench_questions.jsonl` is missing |
| Mind2Web2 | Set `DATASET` to the local Mind2Web2 test file |

## Run Inference

From this directory, run the benchmark-specific script after configuration:

```bash
cd inference
bash scripts/run_react_infer_<benchmark>.sh
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
| `API_CONFIG_FILE` | API configuration file path |
| `SERVER_ENDPOINTS_FILE` | Endpoint configuration file path |
| `HOSTNAME_LIST`, `PORTS` | Endpoint hosts and ports when not using `server_endpoints.conf` |

Example:

```bash
DATASET=/path/to/benchmark.jsonl \
OUTPUT_PATH=/path/to/outputs/results \
TASK_LOG_DIR=/path/to/outputs/logs \
bash scripts/run_react_infer_bc.sh
```

## Resume

The inference pipeline supports resume behavior. If a run is interrupted, rerun
the same script with the same output and log paths.

## Adding A Benchmark

Add a dataset file that matches the input format expected by `run_multi_react.py`,
then create or adapt a `scripts/run_react_infer_*.sh` launch script. Keep scoring and
judge logic under [`../evaluation/`](../evaluation/).
