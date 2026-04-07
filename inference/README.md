# Inference Notes

## Ours

### Default Settings

- Memory threshold: `80K`
  Controls how much accumulated context or memory can be kept before the system starts truncating, filtering, or otherwise limiting retained information.
- Reasoning token: `16K`
  Controls the token budget reserved for the model's reasoning process during inference.

### Search Ranges

- Memory threshold range: `40K - 160K`
  Lower values reduce memory usage and context retention; higher values preserve more context but increase compute and latency.
- Reasoning token range: `10K - 32K`
  Lower values make inference cheaper and faster; higher values allow longer internal reasoning for harder tasks.

### Tunable Parameters

- `server_point` is adjustable.
  This controls which inference or service endpoint the run is routed to, so it can be changed based on available servers or deployment setup.

### Python Sandbox Endpoints

- The Python tool endpoint configuration is hot-reloadable.
- It reads from [`python_nodes.conf`](/fs/scratch/PAS1576/jianxie/DeepResearch/verl/recipe/deepresearch/config/python_nodes.conf).
- In practice, this usually does not need to be modified here.
- Internal hint: for now, keep it synchronized with the RL codebase and simply copy the corresponding config when needed.

## Benchmark

- BrowseComp + HLE + GAIA: Tianhe
- Mind2Web2: Zilu
- LiveResearch Bench + DeepResearch Bench: Yuting
- WideSearch + BrowseComp Plus: Zhehao

## Adding A New Benchmark

When adding a new benchmark, you need to provide an input dataset file in a format similar to [`browsecomp.jsonl`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/browsecomp/browsecomp.jsonl).

- The new benchmark should have a dataset file that matches the input style expected by the inference pipeline.
- The evaluation script for a new benchmark needs to be written separately by you.
- For evaluation-side expectations and organization, refer to [`evaluation/README.md`](/fs/scratch/PAS1576/jianxie/QUEST-github/QUEST/evaluation/README.md).

## Resume

- The current codebase supports `resume`.
- If a run is interrupted or only partially completed, you usually only need to rerun the corresponding `run_*.sh` script to continue.

## Before A New Test

When starting a new test run, check and update the following fields in the corresponding `run_react_infer_*.sh` script.

- `DATASET`
  Points to the evaluation dataset file for the benchmark you want to run.
- `OUTPUT_PATH`
  Controls where final prediction results are written. This should usually be changed for each new experiment to avoid overwriting old outputs.
- `TASK_LOG_DIR`
  Controls where intermediate memory or task logs are stored. This should also be separated across runs.
- `MODEL_PATH`
  Selects which checkpoint or model snapshot is used for inference.
- `MAX_TURN`
  Sets the maximum number of interaction turns allowed for one sample.
- `MAX_LLM_CALL_PER_RUN`
  Usually tied to `MAX_TURN`; limits the number of model calls in a single run.
- `MAX_WORKERS`
  Controls inference concurrency. This may need to be reduced or increased depending on benchmark load and server capacity.
- `MEMORY_THRESHOLD`
  Sets how much memory or retained context the agent can keep before truncation or filtering becomes necessary.
- `LLM_MAX_TOKENS`
  Sets the reasoning or generation token budget for the model.
- `HOSTNAME_LIST` and `PORTS`
  Control which inference servers are queried. These can be changed either directly in the script or through `server_endpoints.conf`.
- `API_CONFIG_FILE`
  Points to the API config used for shared external service settings.
- `SERVER_ENDPOINTS_FILE`
  Points to the endpoint config file. `react_agent.py` reloads this file during runtime, so server routing can be updated without rewriting the script.

## Script Differences

- `run_react_infer_bc.sh`
  Uses BrowseComp dataset, `MAX_TURN=400`, `MEMORY_THRESHOLD=80000`, and `LLM_MAX_TOKENS=20000`.
- `run_react_infer_hle.sh`
  Uses HLE dataset, `MAX_TURN=200`, `MEMORY_THRESHOLD=96000`, and `LLM_MAX_TOKENS=32000`.
- `run_react_infer_drb.sh`
  Uses DeepResearch Bench dataset, `MAX_TURN=200`, `MEMORY_THRESHOLD=16000`, and `LLM_MAX_TOKENS=10000`.
- `run_react_infer_m2w2.sh`
  Uses Mind2Web2 dataset, `MAX_TURN=200`, `MEMORY_THRESHOLD=80000`, and `LLM_MAX_TOKENS=16000`.

## Practical Rule

- If you are running a new benchmark, first update `DATASET`, `OUTPUT_PATH`, and `TASK_LOG_DIR`.
- If you are changing the model setup, update `MODEL_PATH`, `MEMORY_THRESHOLD`, `LLM_MAX_TOKENS`, `MAX_TURN`, and `MAX_WORKERS`.
- If you are switching machines or serving nodes, update `server_endpoints.conf`.
