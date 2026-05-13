# DeepResearch RL Recipe

This directory contains the DeepResearch reinforcement learning recipe used by
the fully async Megatron launcher. It includes the agent loop, tools, reward
logic, task evaluators, data files, service launchers, and configuration used by
training.

> **Status note:** The RL training code is still under active testing and may
> contain bugs. We are working to complete testing within the next two weeks.

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

Local directory for training and validation parquet files. Download the released
`train_parquet` data from the QUEST Hugging Face collection, place the parquet
files here or in another local directory, and set `TRAIN_FILE` / `VAL_FILE`
accordingly before launching training.

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

Do not commit real API keys. The launcher and service scripts load local secrets
from:

```bash
QUEST_ROOT/.secrets/deepresearch_api_keys.env
```

The file should be gitignored and should contain the real values for your
cluster or API providers. Committed scripts should keep empty defaults or
placeholders only.

### Minimum Variables For A Normal Run

Fill these first:

```bash
# Search fallback used by search/scholar tools and services.
export SERPER_KEY_ID="[PLACEHOLDER]"

# Visit-page reader key. `JINA_API_KEYS` can be a comma-separated pool; if it is
# unset, the launcher falls back to `JINA_API_KEY`.
export JINA_API_KEY="[PLACEHOLDER]"
export JINA_API_KEYS="${JINA_API_KEYS:-${JINA_API_KEY}}"

# Shared Azure/OpenAI-compatible endpoint used by legacy code paths and optional
# fallback chains. For an Azure-only setup, API_KEY should be the Azure key.
export API_KEY="[PLACEHOLDER]"
export API_BASE="[PLACEHOLDER]"  # OpenAI-compatible base URL, if used
export AZURE_OPENAI_ENDPOINT="[PLACEHOLDER]"
export AZURE_OPENAI_API_VERSION="[PLACEHOLDER]"
export AZURE_OPENAI_DEPLOYMENT="[PLACEHOLDER]"
```

`OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL_NAME` are compatibility
aliases in the launcher. If you are not using official OpenAI, keep them derived
from the Azure/shared values instead of filling a separate official OpenAI key.

### Eval Node Config Files

The following files provide node/service addresses, not API secrets:

```text
config/search_nodes.conf
config/scholar_nodes.conf
config/python_nodes.conf
config/eval_llm_nodes.conf
```

`config/eval_llm_nodes.conf` is used by local OpenAI-compatible eval-node
routing. It does not replace Azure/API credentials for the non-local fallback
chains below. If a chain uses `PROVIDER=local_openai`, the model request goes to
the configured local eval nodes first. If that local path is unavailable or the
provider is `azure` / `openai` / `api`, the corresponding API key/base/model
variables are used.

### LLM Chains To Fill Independently

These chains are intentionally separate. Do not rely on one chain silently
borrowing another unless you explicitly want that behavior.

Objective reward/eval LLM:

```bash
export EVAL_LLM_PROVIDER="local_openai"  # local_openai | azure | openai | api | auto
export EVAL_LLM_API_KEY="[PLACEHOLDER]"
export EVAL_LLM_API_BASE="[PLACEHOLDER]"
export EVAL_LLM_MODEL_NAME="[PLACEHOLDER]"
export EVAL_LLM_AZURE_ENDPOINT="[PLACEHOLDER]"
export EVAL_LLM_AZURE_API_VERSION="[PLACEHOLDER]"
export EVAL_LLM_AZURE_DEPLOYMENT="[PLACEHOLDER]"

export EVAL_LLM_FALLBACK_PROVIDER="azure"
export EVAL_LLM_FALLBACK_API_KEY="[PLACEHOLDER]"
export EVAL_LLM_FALLBACK_API_BASE="[PLACEHOLDER]"
export EVAL_LLM_FALLBACK_MODEL_NAME="[PLACEHOLDER]"
export EVAL_LLM_FALLBACK_AZURE_ENDPOINT="[PLACEHOLDER]"
export EVAL_LLM_FALLBACK_AZURE_API_VERSION="[PLACEHOLDER]"
export EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT="[PLACEHOLDER]"
export EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME="[PLACEHOLDER]"
```

Inline citation evaluator:

```bash
export CITATION_EVAL_LLM_PROVIDER="azure"
export CITATION_EVAL_LLM_API_KEY="[PLACEHOLDER]"
export CITATION_EVAL_LLM_API_BASE="[PLACEHOLDER]"
export CITATION_EVAL_LLM_MODEL_NAME="[PLACEHOLDER]"
export CITATION_EVAL_LLM_AZURE_ENDPOINT="[PLACEHOLDER]"
export CITATION_EVAL_LLM_AZURE_API_VERSION="[PLACEHOLDER]"
export CITATION_EVAL_LLM_AZURE_DEPLOYMENT="[PLACEHOLDER]"

export CITATION_EVAL_LLM_FALLBACK_PROVIDER="api"
export CITATION_EVAL_LLM_FALLBACK_API_KEY="[PLACEHOLDER]"
export CITATION_EVAL_LLM_FALLBACK_API_BASE="[PLACEHOLDER]"
export CITATION_EVAL_LLM_FALLBACK_MODEL_NAME="[PLACEHOLDER]"
export CITATION_EVAL_LLM_FALLBACK_AZURE_ENDPOINT="[PLACEHOLDER]"
export CITATION_EVAL_LLM_FALLBACK_AZURE_API_VERSION="[PLACEHOLDER]"
export CITATION_EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT="[PLACEHOLDER]"
export CITATION_EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME="[PLACEHOLDER]"
```

Open-ended rubric evaluator:

```bash
export OPENENDED_EVAL_LLM_PROVIDER="local_openai"
export OPENENDED_EVAL_LLM_API_KEY="[PLACEHOLDER]"
export OPENENDED_EVAL_LLM_API_BASE="[PLACEHOLDER]"
export OPENENDED_EVAL_LLM_MODEL_NAME="[PLACEHOLDER]"
export OPENENDED_EVAL_LLM_AZURE_ENDPOINT="[PLACEHOLDER]"
export OPENENDED_EVAL_LLM_AZURE_API_VERSION="[PLACEHOLDER]"
export OPENENDED_EVAL_LLM_AZURE_DEPLOYMENT="[PLACEHOLDER]"

export OPENENDED_EVAL_LLM_FALLBACK_PROVIDER="azure"
export OPENENDED_EVAL_LLM_FALLBACK_API_KEY="[PLACEHOLDER]"
export OPENENDED_EVAL_LLM_FALLBACK_API_BASE="[PLACEHOLDER]"
export OPENENDED_EVAL_LLM_FALLBACK_MODEL_NAME="[PLACEHOLDER]"
export OPENENDED_EVAL_LLM_FALLBACK_AZURE_ENDPOINT="[PLACEHOLDER]"
export OPENENDED_EVAL_LLM_FALLBACK_AZURE_API_VERSION="[PLACEHOLDER]"
export OPENENDED_EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT="[PLACEHOLDER]"
export OPENENDED_EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME="[PLACEHOLDER]"
```

Visit-page summarizer:

```bash
export VISIT_SUMMARY_MODEL_NAME="[PLACEHOLDER]"
export VISIT_SUMMARY_API_KEY="[PLACEHOLDER]"
export VISIT_SUMMARY_API_BASE="[PLACEHOLDER]"
export VISIT_SUMMARY_AZURE_ENDPOINT="[PLACEHOLDER]"
export VISIT_SUMMARY_AZURE_API_VERSION="[PLACEHOLDER]"

export VISIT_SUMMARY_FALLBACK_MODEL_NAME="[PLACEHOLDER]"
export VISIT_SUMMARY_FALLBACK_API_KEY="[PLACEHOLDER]"
export VISIT_SUMMARY_FALLBACK_API_BASE="[PLACEHOLDER]"
export VISIT_SUMMARY_FALLBACK_AZURE_ENDPOINT="[PLACEHOLDER]"
export VISIT_SUMMARY_FALLBACK_AZURE_API_VERSION="[PLACEHOLDER]"
```

Memory condenser:

```bash
export MEMORY_MODEL_NAME="[PLACEHOLDER]"
export MEMORY_API_KEY="[PLACEHOLDER]"
export MEMORY_API_BASE="[PLACEHOLDER]"
export MEMORY_AZURE_ENDPOINT="[PLACEHOLDER]"
export MEMORY_AZURE_API_VERSION="[PLACEHOLDER]"
export MEMORY_AZURE_DEPLOYMENT="[PLACEHOLDER]"

export MEMORY_FALLBACK_MODEL_NAME="[PLACEHOLDER]"
export MEMORY_FALLBACK_API_KEY="[PLACEHOLDER]"
export MEMORY_FALLBACK_API_BASE="[PLACEHOLDER]"
export MEMORY_FALLBACK_AZURE_ENDPOINT="[PLACEHOLDER]"
export MEMORY_FALLBACK_AZURE_API_VERSION="[PLACEHOLDER]"
export MEMORY_FALLBACK_AZURE_DEPLOYMENT="[PLACEHOLDER]"

export MEMORY_LOCAL_FALLBACK_MODEL_NAME="[PLACEHOLDER]"
export MEMORY_LOCAL_FALLBACK_API_KEY="[PLACEHOLDER]"
```

Local eval-node fallback:

```bash
export LOCAL_OPENAI_BASE_URLS="[PLACEHOLDER]"  # optional comma-separated URLs
export LOCAL_OPENAI_FALLBACK_API_KEY="[PLACEHOLDER]"
export LOCAL_OPENAI_FALLBACK_API_BASE="[PLACEHOLDER]"
export LOCAL_OPENAI_FALLBACK_MODEL_NAME="[PLACEHOLDER]"
export LOCAL_OPENAI_FALLBACK_AZURE_ENDPOINT="[PLACEHOLDER]"
export LOCAL_OPENAI_FALLBACK_AZURE_API_VERSION="[PLACEHOLDER]"
export LOCAL_OPENAI_FALLBACK_AZURE_DEPLOYMENT="[PLACEHOLDER]"

export LOCAL_OPENAI_SECONDARY_FALLBACK_API_KEY="[PLACEHOLDER]"
export LOCAL_OPENAI_SECONDARY_FALLBACK_API_BASE="[PLACEHOLDER]"
export LOCAL_OPENAI_SECONDARY_FALLBACK_MODEL_NAME="[PLACEHOLDER]"
export LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_ENDPOINT="[PLACEHOLDER]"
export LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_API_VERSION="[PLACEHOLDER]"
export LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_DEPLOYMENT="[PLACEHOLDER]"
```

### Optional Service And Tool Variables

These are not always required because the config files normally provide the
addresses:

```bash
export SANDBOX_FUSION_ENDPOINT="[PLACEHOLDER]"   # optional single Python sandbox endpoint
export SANDBOX_FUSION_ENDPOINTS="[PLACEHOLDER]"  # optional comma-separated endpoints
export PYTHON_SERVICE_URL="[PLACEHOLDER]"
export PYTHON_SERVICE_URLS="[PLACEHOLDER]"

export SEARCH_SERVICE_URL="[PLACEHOLDER]"
export SEARCH_NODES_CONF="recipe/deepresearch/config/search_nodes.conf"
export SCHOLAR_SERVICE_URL="[PLACEHOLDER]"
export SCHOLAR_NODES_CONF="recipe/deepresearch/config/scholar_nodes.conf"
export PYTHON_NODES_CONF="recipe/deepresearch/config/python_nodes.conf"
export EVAL_LLM_NODES_CONF="recipe/deepresearch/config/eval_llm_nodes.conf"
```

Optional keys for specific tools or benchmarks:

```bash
export GOOGLE_MAPS_API_KEY="[PLACEHOLDER]"
export HLE_JUDGE_MODEL_NAME="[PLACEHOLDER]"
export AWS_ACCESS_KEY="[PLACEHOLDER]"
export AWS_SECRET_KEY="[PLACEHOLDER]"
export AWS_REGION="[PLACEHOLDER]"
```

The launcher forwards the relevant variables into Ray runtime environments.

## Data

Download the released RL training parquet files from the QUEST Hugging Face
collection:

https://huggingface.co/collections/osunlp/quest

The released training data is provided under `train_parquet`. After downloading
the parquet files, point the launcher to the local paths:

```bash
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/val.parquet
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
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/val.parquet
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
