# DeepResearch RL Recipe

This directory contains the DeepResearch reinforcement learning recipe used by
the fully async Megatron launcher. It includes the agent loop, tools, reward
logic, task evaluators, data files, service launchers, and configuration used by
training.

Run commands from the RL root unless noted otherwise:

```bash
cd training_scripts/rl
```

## Directory Layout

`agent_loop/`

DeepResearch rollout logic. This is where the agent executes multi-turn research
trajectories, calls tools, manages partial rollout state, and emits trajectories
for training.

`citation_task_eval/`

Inline citation evaluation. This checks whether final answers cite sources and
whether cited pages support the cited claims.

`config/`

Runtime configuration files:

- `tools.yaml`: tool registry and tool-specific settings.
- `agent_loop_config.yaml`: agent-loop behavior and tool configuration.
- `deepresearch_trainer.yaml`: base trainer config.
- `search_nodes.conf`: search service endpoints.
- `scholar_nodes.conf`: scholar service endpoints.
- `python_nodes.conf`: Python sandbox endpoints.
- `eval_llm_nodes.conf`: eval LLM endpoints, with sections such as `[obj]`,
  `[openended]`, and `[citation]`.

`data/`

Training and validation parquet files. The default launcher uses:

- `data/train_v4.parquet`
- `data/val_v4.parquet`

`eval_scripts/`

Objective task evaluation scripts. These scripts are loaded by `reward.py` for
task-specific checks.

`obj_task_eval/`

Objective-task evaluation utilities, including generated-verifier execution,
tooling, prompts, and LLM-client helpers.

`openended_task_eval/`

Open-ended rubric evaluation. This handles criteria-based scoring for tasks
whose ground truth is a rubric rather than a deterministic verifier.

`scripts/`

Operational entrypoints:

- `run_search_service.sh`: start the search HTTP service.
- `run_scholar_service.sh`: start the scholar HTTP service.
- `init_faiss_search.sh`: build the search FAISS index.
- `init_faiss_scholar.sh`: build the scholar FAISS index.
- `build_search_faiss.py`: Python entrypoint for search FAISS build.
- `build_scholar_faiss.py`: Python entrypoint for scholar FAISS build.

`tools/`

Tool implementations used by the agent:

- `search_tool.py` / `search_service.py`: web search cache, FAISS retrieval,
  and Serper fallback.
- `scholar_tool.py` / `scholar_service.py`: scholar search cache, FAISS
  retrieval, and Serper fallback.
- `visit_tool.py`: webpage fetch, cache, and summarization.
- `python_tool.py`: remote Python sandbox calls.
- `_faiss_build_worker.py`: helper worker for multi-GPU FAISS embedding builds.

Top-level Python files:

- `reward.py`: main reward function and eval routing.
- `reward_manager.py`: standard reward manager integration.
- `reward_loop_manager.py`: reward-loop integration.
- `deepresearch_ray_trainer.py`: DeepResearch trainer extensions and metrics.
- `deepresearch_main_ppo.py`: PPO entrypoint.
- `memory.py`: memory/condenser logic.
- `curriculum_sampler.py`: optional curriculum sampler.
- `session_algos.py`: session-level algorithm helpers.
- `run_deepresearch_fully_async_megatron.sh`: main fully async Megatron
  training launcher.

## Secrets And Environment

Do not commit real API keys. The launcher and service scripts load secrets from:

```bash
QUEST_ROOT/.secrets/deepresearch_api_keys.env
```

The file should define only local secrets and should be gitignored. Typical
variables include:

```bash
export SERPER_KEY_ID="..."
export JINA_API_KEY="..."
export API_KEY="..."
export API_BASE="..."
export AZURE_OPENAI_ENDPOINT="..."
export AZURE_OPENAI_API_VERSION="..."
export AZURE_OPENAI_DEPLOYMENT="..."
```

DeepResearch-specific LLM chains are configured independently:

```bash
export EVAL_LLM_PROVIDER="..."
export EVAL_LLM_API_KEY="..."
export EVAL_LLM_API_BASE="..."
export EVAL_LLM_MODEL_NAME="..."

export CITATION_EVAL_LLM_PROVIDER="..."
export CITATION_EVAL_LLM_API_KEY="..."
export CITATION_EVAL_LLM_MODEL_NAME="..."

export OPENENDED_EVAL_LLM_PROVIDER="..."
export OPENENDED_EVAL_LLM_API_KEY="..."
export OPENENDED_EVAL_LLM_MODEL_NAME="..."

export VISIT_SUMMARY_MODEL_NAME="..."
export VISIT_SUMMARY_API_KEY="..."
export VISIT_SUMMARY_API_BASE="..."

export MEMORY_MODEL_NAME="..."
export MEMORY_API_KEY="..."
export MEMORY_API_BASE="..."
```

The launcher forwards these variables into Ray runtime environments.

## Data

The default train/validation files are:

```bash
recipe/deepresearch/data/train_v4.parquet
recipe/deepresearch/data/val_v4.parquet
```

The task type is stored in the parquet `reward_model` / `extra_info` metadata.
Open-ended tasks use:

```text
type = open-ended
```

The launcher accepts:

```bash
DATA_KIND=both
DATA_KIND=obj
DATA_KIND=openended
```

When `DATA_KIND` is not `both`, the launcher builds filtered parquet files under
`recipe/deepresearch/data/cache/`.

## Tool Services

Search and scholar can run as separate HTTP services. This is recommended when
training workers should not load local FAISS indexes or perform Serper calls
directly.

Start search service:

```bash
cd training_scripts/rl
bash recipe/deepresearch/scripts/run_search_service.sh
```

Default port: `8000`

Override common settings:

```bash
export SEARCH_SERVICE_PORT=8000
export SEARCH_SERVICE_CONFIG=recipe/deepresearch/config/tools.yaml
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SEARCH_FAISS_READ_GPUS=0,1,2
export SEARCH_FAISS_WRITE_GPUS=3
bash recipe/deepresearch/scripts/run_search_service.sh
```

Start scholar service:

```bash
cd training_scripts/rl
bash recipe/deepresearch/scripts/run_scholar_service.sh
```

Default port: `8001`

Override common settings:

```bash
export SCHOLAR_SERVICE_PORT=8001
export SCHOLAR_SERVICE_CONFIG=recipe/deepresearch/config/tools.yaml
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SCHOLAR_FAISS_READ_GPUS=0,1,2
export SCHOLAR_FAISS_WRITE_GPUS=3
bash recipe/deepresearch/scripts/run_scholar_service.sh
```

Training workers discover these services from:

```text
recipe/deepresearch/config/search_nodes.conf
recipe/deepresearch/config/scholar_nodes.conf
```

The Python sandbox is configured through:

```text
recipe/deepresearch/config/python_nodes.conf
```

## Build FAISS Indexes

Install the required packages in the environment used for FAISS building:

```bash
pip install faiss-cpu sentence-transformers pyyaml
```

Set the embedding model:

```bash
export DEEPRESEARCH_EMBEDDING_MODEL=/path/to/embedding/model
```

Build search FAISS:

```bash
cd training_scripts/rl
bash recipe/deepresearch/scripts/init_faiss_search.sh --skip-merge
```

Build scholar FAISS:

```bash
cd training_scripts/rl
bash recipe/deepresearch/scripts/init_faiss_scholar.sh --skip-merge
```

Use `--skip-merge` when the merged SQLite cache already exists. Omit it only
when cache shards must first be merged.

The cache and FAISS paths are controlled by `tools.yaml` or these environment
variables:

```bash
export SEARCH_CACHE_DIR=recipe/deepresearch/database
export SEARCH_CACHE_FILE=recipe/deepresearch/database/search.db
export SCHOLAR_CACHE_DIR=recipe/deepresearch/database
export SCHOLAR_CACHE_FILE=recipe/deepresearch/database/scholar.db
```

## Start Training

Prepare the external services first:

1. Search service if `search_nodes.conf` points to HTTP endpoints.
2. Scholar service if `scholar_nodes.conf` points to HTTP endpoints.
3. Python sandbox service if `python_nodes.conf` is used.
4. Eval LLM endpoints listed in `eval_llm_nodes.conf`.

For a local Ray run:

```bash
cd training_scripts/rl
bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
```

For an existing Ray cluster:

```bash
cd training_scripts/rl
export RAY_ADDRESS=auto
bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
```

Useful launcher overrides:

```bash
export PROJECT_NAME=DeepResearch
export EXP_NAME=my-run
export MODEL_PATH=/path/to/model
export TRAIN_FILE=recipe/deepresearch/data/train_v4.parquet
export VAL_FILE=recipe/deepresearch/data/val_v4.parquet
export DATA_KIND=both
export TOTAL_ROLLOUT_STEPS=12800
export TARGET_TRAIN_STEPS=200
export N_RESP_PER_PROMPT=8
export TRAIN_PROMPT_MINI_BSZ=16
export MAX_PROMPT_LENGTH=24000
export MAX_RESPONSE_LENGTH=12288
export MAX_TURN_RESPONSE_LENGTH=10240
```

Then run:

```bash
bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
```

## Typical Startup Order

1. Fill `QUEST_ROOT/.secrets/deepresearch_api_keys.env` locally.
2. Configure `tools.yaml` and node conf files under `config/`.
3. Build FAISS indexes if using local cache + FAISS.
4. Start search and scholar services if using HTTP service mode.
5. Start Python sandbox nodes.
6. Start or connect to a Ray cluster.
7. Launch `run_deepresearch_fully_async_megatron.sh`.

## Notes

- `run_search_service.sh` and `run_scholar_service.sh` are entrypoints; the
  actual implementations live in `tools/search_service.py` and
  `tools/scholar_service.py`.
- `scripts/init_faiss_*.sh` are entrypoints; the FAISS build logic is in
  `scripts/build_*_faiss.py` and `tools/_faiss_build_worker.py`.
- `visit` does not have a FAISS index. It uses the visit SQLite cache and
  configured summarizer LLM.
- The launcher uses `set -euo pipefail`; missing required environment variables
  should fail early.
