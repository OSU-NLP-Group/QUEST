#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# Use local mode by default; set RAY_ADDRESS=auto or <head_ip>:6379 for multi-node.
RAY_ADDRESS=${RAY_ADDRESS:-local}

ulimit -u 32768
ulimit -n 32768

# -----------------------------
# Experiment and algorithm
# -----------------------------
project_name=${PROJECT_NAME:-DeepResearch}
exp_name=${EXP_NAME:-Quest-35B-A3-fully-asyn-20260424-v1}

actor_strategy=${ACTOR_STRATEGY:-megatron}
critic_strategy=${CRITIC_STRATEGY:-${actor_strategy}}
if [[ "${actor_strategy}" != "megatron" || "${critic_strategy}" != "megatron" ]]; then
    echo "[ERROR] This Megatron launcher requires ACTOR_STRATEGY=megatron and CRITIC_STRATEGY=megatron." >&2
    exit 1
fi
fully_async_config_name=${FULLY_ASYNC_CONFIG_NAME:-fully_async_ppo_megatron_trainer}

adv_estimator=${ADV_ESTIMATOR:-grpo}
use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
loss_mode=${LOSS_MODE:-vanilla}
loss_agg_mode=${LOSS_AGG_MODE:-seq-mean-token-mean}
# Session weight correction: make split sessions collectively behave like one unsplit traj under token-mean.
session_weight_correction=${SESSION_WEIGHT_CORRECTION:-False}

# Optional filter_groups knobs are kept for compatibility.
enable_filter_groups=${ENABLE_FILTER_GROUPS:-False}
filter_groups_metric=${FILTER_GROUPS_METRIC:-acc}
max_num_gen_batches=${MAX_NUM_GEN_BATCHES:-10}

# -----------------------------
# Dynamic Curriculum Learning (C1-C9)
# -----------------------------
# Enable by default; set CURRICULUM_ENABLED=False to disable.
curriculum_enabled=${CURRICULUM_ENABLED:-False}
curriculum_objective=${CURRICULUM_OBJECTIVE:-adv}   # 'adv' (larger |advantage| has higher priority) or 'progress' (faster reward improvement has higher priority)
curriculum_lr=${CURRICULUM_LR:-0.1}
curriculum_temperature=${CURRICULUM_TEMPERATURE:-1.0}
curriculum_min_weight=${CURRICULUM_MIN_WEIGHT:-0.02}
curriculum_replacement=${CURRICULUM_REPLACEMENT:-False}   # True=sampling with replacement, False=sampling without replacement (default)

# -----------------------------
# Sequence lengths and sampling
# -----------------------------
max_prompt_length=${MAX_PROMPT_LENGTH:-24000}
# MAX_RESPONSE_LENGTH is the per-session / training response budget.
max_response_length=${MAX_RESPONSE_LENGTH:-12288}
# MAX_TURN_RESPONSE_LENGTH only caps a single assistant generation turn in vLLM.
max_turn_response_length=${MAX_TURN_RESPONSE_LENGTH:-10240}
# Budget for actor/ref training and logprob computation over the full sequence.
full_sequence_token_budget=$((max_prompt_length + max_response_length))
# Budget for rollout-side generation scheduling in vLLM.
rollout_generation_token_budget=$((max_turn_response_length + max_response_length))
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-16}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}

temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1} # 0 for HF rollout, -1 for vLLM rollout

# -----------------------------
# Fully-async policy knobs
# -----------------------------
# In fully async, train_batch_size is not used by trainer logic; keep 0 by default.
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-0}
# Streaming generation batch size should be 1 for best pipeline behavior.
gen_prompt_bsz=${GEN_PROMPT_BSZ:-1}
# Qwen3.5/Qwen3-Next GDN prefill on SM90 can fail FlashInfer JIT when the
# local CUDA/CCCL headers do not match FlashInfer's requirements. Default to
# Triton for stability; override with GDN_PREFILL_BACKEND=flashinfer if the
# environment is known-good.
gdn_prefill_backend=${GDN_PREFILL_BACKEND:-triton}

# Equivalent legacy scale target for rollout count alignment.
legacy_train_prompt_bsz=${LEGACY_TRAIN_PROMPT_BSZ:-64}
target_train_steps=${TARGET_TRAIN_STEPS:-200}
total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-$((legacy_train_prompt_bsz * target_train_steps))}
megatron_total_training_steps=${MEGATRON_TOTAL_TRAINING_STEPS:-${target_train_steps}}
megatron_lr_decay_steps=${MEGATRON_LR_DECAY_STEPS:-${megatron_total_training_steps}} 

if (( megatron_total_training_steps <= 0 )); then
    echo "[ERROR] MEGATRON_TOTAL_TRAINING_STEPS must be > 0, got ${megatron_total_training_steps}" >&2
    exit 1
fi

if (( megatron_lr_decay_steps <= 0 )); then
    echo "[ERROR] MEGATRON_LR_DECAY_STEPS must be > 0, got ${megatron_lr_decay_steps}" >&2
    exit 1
fi

rollout_total_epochs=${ROLLOUT_TOTAL_EPOCHS:-400}
rollout_test_freq=${ROLLOUT_TEST_FREQ:-100}
trainer_val_before_train=${TRAINER_VAL_BEFORE_TRAIN:-False}

async_require_batches=${ASYNC_REQUIRE_BATCHES:-1}

if (( async_require_batches <= 0 )); then
    echo "[ERROR] ASYNC_REQUIRE_BATCHES must be > 0, got ${async_require_batches}" >&2
    exit 1
fi
if (( train_prompt_mini_bsz <= 0 )); then
    echo "[ERROR] TRAIN_PROMPT_MINI_BSZ must be > 0, got ${train_prompt_mini_bsz}" >&2
    exit 1
fi

# Default to colocate-equivalent sync step:
# trigger_parameter_sync_step ~= legacy_train_prompt_bsz / (require_batches * ppo_mini_batch_size)
sync_step_denominator=$((async_require_batches * train_prompt_mini_bsz))
if [[ -n "${ASYNC_TRIGGER_PARAMETER_SYNC_STEP:-}" ]]; then
    async_trigger_parameter_sync_step=${ASYNC_TRIGGER_PARAMETER_SYNC_STEP}
else
    async_trigger_parameter_sync_step=$((legacy_train_prompt_bsz / sync_step_denominator))
    if (( async_trigger_parameter_sync_step < 1 )); then
        async_trigger_parameter_sync_step=1
    fi
    if (( legacy_train_prompt_bsz % sync_step_denominator != 0 )); then
        echo "[WARN] LEGACY_TRAIN_PROMPT_BSZ=${legacy_train_prompt_bsz} is not divisible by "\
"ASYNC_REQUIRE_BATCHES*TRAIN_PROMPT_MINI_BSZ=${sync_step_denominator}; "\
"default ASYNC_TRIGGER_PARAMETER_SYNC_STEP is floored to ${async_trigger_parameter_sync_step}."
    fi
fi

async_staleness_threshold=${ASYNC_STALENESS_THRESHOLD:-0.5}
# DeepResearch async-partial agent supports interruption/resume for partial rollout.
async_partial_rollout=${ASYNC_PARTIAL_ROLLOUT:-True}
rollout_correction_bypass_mode=${ROLLOUT_CORRECTION_BYPASS_MODE:-True}
# Reward-cancel behavior when parameter sync interrupts ongoing reward computation:
# - save_state: keep in-flight reward future and resume by awaiting the same future (default).
# - exit: drop in-flight reward and resume by re-running reward later.
async_reward_cancel_mode=${ASYNC_REWARD_CANCEL_MODE:-save_state}
# True will let Trainer launch extra async rollout servers for validation, which can double vLLM footprint.
async_use_trainer_do_validate=${ASYNC_USE_TRAINER_DO_VALIDATE:-False}

if [[ "${async_reward_cancel_mode}" != "exit" && "${async_reward_cancel_mode}" != "save_state" ]]; then
    echo "[ERROR] ASYNC_REWARD_CANCEL_MODE must be one of: exit, save_state. Got ${async_reward_cancel_mode}" >&2
    exit 1
fi

async_ckpt_enable=${ASYNC_CKPT_ENABLE:-True}
async_ckpt_overlap=${ASYNC_CKPT_OVERLAP_BROADCAST_AND_CONSUME:-False}
async_ckpt_device_buffer_size_m=${ASYNC_CKPT_DEVICE_BUFFER_SIZE_M:-1024}
# Save a rolling checkpoint after every logical sync step.
# save_freq checkpoints are kept as formal checkpoints named savefreq_step_<n>.
save_every_step_ckpt=${SAVE_EVERY_STEP_CKPT:-True}
export DEEPRESEARCH_SAVE_EVERY_STEP_CKPT="${save_every_step_ckpt}"

# -----------------------------
# Debug load/save-only mode
# -----------------------------
debug_load_save_only=${DEBUG_LOAD_SAVE_ONLY:-False}
debug_global_step=${DEBUG_GLOBAL_STEP:-0}
debug_load_checkpoint_path=${DEBUG_LOAD_CHECKPOINT_PATH:-null}
debug_save_base_dir=${DEBUG_SAVE_BASE_DIR:-}
debug_save_checkpoint_path=${DEBUG_SAVE_CHECKPOINT_PATH:-}

# -----------------------------
# Checkpoint resume
# -----------------------------
# resume_mode: disable=start from scratch, auto=find the latest checkpoint under default_local_dir, resume_path=use resume_from_path.
resume_mode=${RESUME_MODE:-auto}
# Only effective when resume_mode=resume_path. Must be a full path, for example:
#   ${CKPTS_DIR}/step_10
#   ${CKPTS_DIR}/savefreq_step_10
# Backward-compatible legacy format:
#   ${CKPTS_DIR}/global_step_10
resume_from_path=${RESUME_FROM_PATH:-}

# Restore optimizer param scheduler state from checkpoint by default when resuming.
# This keeps LR continuous across resume instead of rebuilding a fresh scheduler.
if [[ -n "${USE_CHECKPOINT_OPT_PARAM_SCHEDULER:-}" ]]; then
    use_checkpoint_opt_param_scheduler=${USE_CHECKPOINT_OPT_PARAM_SCHEDULER}
elif [[ "${resume_mode}" == "disable" ]]; then
    use_checkpoint_opt_param_scheduler=False
else
    use_checkpoint_opt_param_scheduler=True
fi

RESUME_PATH_OVERRIDES=()
if [[ "${resume_mode}" == "resume_path" ]] && [[ -n "${resume_from_path}" ]]; then
    RESUME_PATH_OVERRIDES=("trainer.resume_from_path='${resume_from_path}'")
elif [[ "${resume_mode}" == "resume_path" ]]; then
    echo "[ERROR] RESUME_FROM_PATH must be set when resume_mode=resume_path" >&2
    exit 1
fi

# -----------------------------
# Resource layout
# -----------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR=${WORKING_DIR:-"$(cd -- "${SCRIPT_DIR}/../.." && pwd)"}
QUEST_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
DEEPRESEARCH_SECRETS_ENV=${DEEPRESEARCH_SECRETS_ENV:-"${QUEST_ROOT}/.secrets/deepresearch_api_keys.env"}
if [[ -f "${DEEPRESEARCH_SECRETS_ENV}" ]]; then
    _deepresearch_had_xtrace=0
    _deepresearch_had_nounset=0
    case "$-" in
        *x*) _deepresearch_had_xtrace=1; set +x ;;
    esac
    case "$-" in
        *u*) _deepresearch_had_nounset=1; set +u ;;
    esac
    set -a
    source "${DEEPRESEARCH_SECRETS_ENV}"
    set +a
    if [[ "${_deepresearch_had_nounset}" == "1" ]]; then
        set -u
    fi
    if [[ "${_deepresearch_had_xtrace}" == "1" ]]; then
        set -x
    fi
    unset _deepresearch_had_xtrace _deepresearch_had_nounset
fi
NNODES=${NNODES:-8}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
ROLLOUT_NGPUS_PER_NODE=${ROLLOUT_NGPUS_PER_NODE:-$((NGPUS_PER_NODE / 2))}
TRAINER_NGPUS_PER_NODE=${TRAINER_NGPUS_PER_NODE:-$((NGPUS_PER_NODE - ROLLOUT_NGPUS_PER_NODE))}

if (( ROLLOUT_NGPUS_PER_NODE <= 0 )); then
    echo "[ERROR] ROLLOUT_NGPUS_PER_NODE must be > 0, got ${ROLLOUT_NGPUS_PER_NODE}" >&2
    exit 1
fi
if (( TRAINER_NGPUS_PER_NODE <= 0 )); then
    echo "[ERROR] TRAINER_NGPUS_PER_NODE must be > 0, got ${TRAINER_NGPUS_PER_NODE}" >&2
    exit 1
fi

# -----------------------------
# Paths
# -----------------------------
RAY_DATA_HOME=${RAY_DATA_HOME:-"${WORKING_DIR}/saves"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/qwen3_5-moe-mid-training-plus-sft_8500"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${RAY_DATA_HOME}/rollouts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${WORKING_DIR}/recipe/deepresearch/data/train_v4.parquet"}
VAL_FILE=${VAL_FILE:-}
DATA_KIND=${DATA_KIND:-both}

data_kind="$(echo "${DATA_KIND}" | tr '[:upper:]' '[:lower:]')"
case "${data_kind}" in
    both|obj|openended|open-ended)
        ;;
    *)
        echo "[ERROR] DATA_KIND must be one of: both, obj, openended. Got ${DATA_KIND}" >&2
        exit 1
        ;;
esac

if [[ "${data_kind}" == "open-ended" ]]; then
    data_kind="openended"
fi

filter_parquet_list_by_type() {
    local input_list=$1
    local split_name=$2
    local target_type=$3
    local cache_dir="${SCRIPT_DIR}/data/cache"
    local IFS=','
    local input_files=()
    local output_files=()

    mkdir -p "${cache_dir}"
    read -r -a input_files <<< "${input_list}"

    for input_path in "${input_files[@]}"; do
        input_path="$(echo "${input_path}" | xargs)"
        if [[ -z "${input_path}" ]]; then
            continue
        fi
        if [[ ! -f "${input_path}" ]]; then
            echo "[ERROR] ${split_name} parquet not found: ${input_path}" >&2
            exit 1
        fi

        local input_base
        local input_stem
        local output_path
        input_base="$(basename "${input_path}")"
        input_stem="${input_base%.parquet}"
        output_path="${cache_dir}/${input_stem}_${target_type}.parquet"

        if [[ ! -f "${output_path}" || "${input_path}" -nt "${output_path}" ]]; then
            echo "[INFO] Building ${split_name} parquet for DATA_KIND=${target_type}: ${output_path}" >&2
            python3 - "${input_path}" "${output_path}" "${target_type}" <<'PY'
import sys
import pandas as pd

input_path, output_path, target_type = sys.argv[1:4]
df = pd.read_parquet(input_path)

def _extract_type(record):
    if not isinstance(record, dict):
        return None
    ground_truth = record.get("ground_truth")
    if isinstance(ground_truth, dict):
        record_type = ground_truth.get("type")
        if record_type is not None:
            return record_type
    return record.get("type")

def _normalize_type(value):
    text = str(value or "").strip().lower()
    aliases = {
        "open-ended": "openended",
        "openended": "openended",
        "obj": "obj",
        "objective": "obj",
    }
    return aliases.get(text, text)

reward_types = df["reward_model"].map(_extract_type) if "reward_model" in df.columns else pd.Series([None] * len(df))
extra_types = df["extra_info"].map(_extract_type) if "extra_info" in df.columns else pd.Series([None] * len(df))
target_type = _normalize_type(target_type)
mask = reward_types.map(_normalize_type).eq(target_type) | extra_types.map(_normalize_type).eq(target_type)
filtered = df.loc[mask].reset_index(drop=True)

if filtered.empty:
    raise SystemExit(f"[ERROR] No rows matched type={target_type!r} in {input_path}")

filtered.to_parquet(output_path, index=False)
print(f"[INFO] Wrote {len(filtered)} / {len(df)} rows to {output_path}", file=sys.stderr)
PY
        fi
        output_files+=("${output_path}")
    done

    local joined_output=""
    for output_path in "${output_files[@]}"; do
        if [[ -n "${joined_output}" ]]; then
            joined_output+=","
        fi
        joined_output+="${output_path}"
    done

    echo "${joined_output}"
}

if [[ "${data_kind}" != "both" ]]; then
    TRAIN_FILE="$(filter_parquet_list_by_type "${TRAIN_FILE}" "train" "${data_kind}")"
    if [[ -z "${TRAIN_FILE}" ]]; then
        echo "[ERROR] DATA_KIND filtering produced an empty TRAIN_FILE list." >&2
        exit 1
    fi
fi

echo "[INFO] DATA_KIND=${data_kind}"
echo "[INFO] TRAIN_FILE=${TRAIN_FILE}"
if [[ -n "${VAL_FILE}" ]]; then
    if [[ ! -f "${VAL_FILE}" ]]; then
        echo "[ERROR] VAL_FILE not found: ${VAL_FILE}" >&2
        exit 1
    fi
    VALIDATION_OVERRIDES=(
        "data.val_files='${VAL_FILE}'"
        "async_training.use_trainer_do_validate=${async_use_trainer_do_validate}"
        "trainer.val_before_train=${trainer_val_before_train}"
        "trainer.test_freq=5"
    )
    echo "[INFO] VAL_FILE=${VAL_FILE}"
else
    VALIDATION_OVERRIDES=(
        "data.val_files=[]"
        "async_training.use_trainer_do_validate=False"
        "trainer.val_before_train=False"
        "trainer.test_freq=-1"
    )
    echo "[INFO] VAL_FILE disabled"
fi

if [[ -z "${debug_save_base_dir}" ]]; then
    debug_save_base_dir="${CKPTS_DIR}"
fi

TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-"${WORKING_DIR}/recipe/deepresearch/config/tools.yaml"}
EVAL_SCRIPTS_DIR=${EVAL_SCRIPTS_DIR:-"${WORKING_DIR}/recipe/deepresearch/eval_scripts"}

# Support both DeepResearch loop and built-in async-partial loop.
# - deepresearch_agent: original DeepResearch agent behavior.
# - deepresearch_async_partial_agent: DeepResearch agent with partial-rollout interruption/resume.
DEFAULT_AGENT_LOOP=${DEFAULT_AGENT_LOOP:-deepresearch_async_partial_agent}
AGENT_LOOP_CONFIG_PATH=${AGENT_LOOP_CONFIG_PATH:-recipe/deepresearch/config/agent_loop_config.yaml}

async_partial_rollout_norm="$(echo "${async_partial_rollout}" | tr '[:upper:]' '[:lower:]')"
if [[ "${async_partial_rollout_norm}" == "true" ]] && [[ "${DEFAULT_AGENT_LOOP}" == "deepresearch_agent" ]]; then
    echo "[WARN] ASYNC_PARTIAL_ROLLOUT=True requires interruption/resume support. "\
"Switching DEFAULT_AGENT_LOOP from deepresearch_agent to deepresearch_async_partial_agent."
    DEFAULT_AGENT_LOOP=deepresearch_async_partial_agent
fi
if [[ "${async_partial_rollout_norm}" != "true" ]] && [[ "${DEFAULT_AGENT_LOOP}" == "deepresearch_async_partial_agent" ]]; then
    echo "[WARN] DEFAULT_AGENT_LOOP=deepresearch_async_partial_agent is designed for ASYNC_PARTIAL_ROLLOUT=True."
fi

# DeepResearch multi-turn/memory knobs
# Align with proposer_v1/inference_2/run_react_infer.sh semantics:
# MAX_LLM_CALL_PER_RUN means the maximum number of assistant reasoning rounds.
max_llm_call_per_run=${MAX_LLM_CALL_PER_RUN:-50}
max_assistant_turns=${MAX_ASSISTANT_TURNS:-${max_llm_call_per_run}}
max_user_turns=${MAX_USER_TURNS:-${max_llm_call_per_run}}
# Each reasoning round usually adds one assistant turn and one environment/user turn.
# Keep a larger aggregate cap so assistant rounds are not truncated early by total turns.
max_turns=${MAX_TURNS:-$((max_assistant_turns + max_user_turns))}
max_parallel_calls=${MAX_PARALLEL_CALLS:-1}
max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH:-12000}
tool_response_truncate_side=${TOOL_RESPONSE_TRUNCATE_SIDE:-middle}
memory_enabled=${MEMORY_ENABLED:-True}

# vLLM eval LLM configuration (optional)
# IPs: read from config file at runtime; env var EVAL_LLM_IPS overrides if set.
EVAL_LLM_NODES_CONF="${EVAL_LLM_NODES_CONF:-${SCRIPT_DIR}/config/eval_llm_nodes.conf}"
if [[ -n "${EVAL_LLM_IPS:-}" ]]; then
    :  # env var takes precedence, keep as-is
elif [[ -f "${EVAL_LLM_NODES_CONF}" ]]; then
    # Supports both legacy one-IP-per-line format and sectioned format:
    # [obj] ips=..., [openended] ips=...
    EVAL_LLM_IPS="$(
python3 - "${EVAL_LLM_NODES_CONF}" <<'PY'
import re
import sys

path = sys.argv[1]
current = "obj"
ips = []

def norm_profile(name: str) -> str:
    n = (name or "").strip().lower()
    aliases = {
        "default": "obj",
        "eval": "obj",
        "objective": "obj",
        "obj": "obj",
        "main": "obj",
        "openended": "openended",
        "open-ended": "openended",
        "citation": "citation",
        "cite": "citation",
    }
    return aliases.get(n, n)

with open(path, "r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            current = norm_profile(line[1:-1].strip())
            continue

        if "=" in line:
            key, value = [x.strip() for x in line.split("=", 1)]
            key_l = key.lower()
            target_profile = current
            parsed_key = key_l
            if "." in key_l:
                maybe_profile, maybe_key = key_l.split(".", 1)
                profile = norm_profile(maybe_profile)
                if profile in {"obj", "openended", "citation"}:
                    target_profile = profile
                    parsed_key = maybe_key.strip().lower()
            elif "_" in key_l:
                maybe_profile, maybe_key = key_l.split("_", 1)
                profile = norm_profile(maybe_profile)
                if profile in {"obj", "openended", "citation"}:
                    target_profile = profile
                    parsed_key = maybe_key.strip().lower()

            if target_profile == "obj" and parsed_key in {"ip", "ips", "nodes", "hosts"}:
                ips.extend([x for x in re.split(r"[,\s]+", value) if x])
            continue

        if current == "obj":
            ips.extend([x for x in re.split(r"[,\s]+", line) if x])

seen = set()
ordered = []
for ip in ips:
    if ip in seen:
        continue
    seen.add(ip)
    ordered.append(ip)

print(",".join(ordered))
PY
)"
    if [[ -z "${EVAL_LLM_IPS}" ]]; then
        echo "[ERROR] ${EVAL_LLM_NODES_CONF} exists but contains no valid IPs." >&2
        exit 1
    fi
    echo "[INFO] Loaded EVAL_LLM_IPS from ${EVAL_LLM_NODES_CONF}: ${EVAL_LLM_IPS}"
else
    EVAL_LLM_IPS="a0002,a0004,a0001"
    echo "[WARN] ${EVAL_LLM_NODES_CONF} not found, using default EVAL_LLM_IPS=${EVAL_LLM_IPS}"
fi
EVAL_LLM_PORTS="${EVAL_LLM_PORTS:-6000,6001,6002,6003}"
EVAL_LLM_MODEL="${EVAL_LLM_MODEL:-${VLLM_JUDGE_MODEL:-eval_model}}"

# Search service URL configuration (hot-reloaded each call, like eval_llm_nodes.conf).
# Edit config/search_nodes.conf at runtime to switch the search service without restart.
SEARCH_NODES_CONF="${SEARCH_NODES_CONF:-${SCRIPT_DIR}/config/search_nodes.conf}"
export SEARCH_NODES_CONF
echo "[INFO] SEARCH_NODES_CONF=${SEARCH_NODES_CONF}"

# Scholar service URL configuration (hot-reloaded each call, like eval_llm_nodes.conf).
# Edit config/scholar_nodes.conf at runtime to switch the scholar service without restart.
SCHOLAR_NODES_CONF="${SCHOLAR_NODES_CONF:-${SCRIPT_DIR}/config/scholar_nodes.conf}"
export SCHOLAR_NODES_CONF
echo "[INFO] SCHOLAR_NODES_CONF=${SCHOLAR_NODES_CONF}"

# Python sandbox endpoint configuration (hot-reloaded each call).
# Edit config/python_nodes.conf at runtime to switch sandbox endpoints without restart.
PYTHON_NODES_CONF="${PYTHON_NODES_CONF:-${SCRIPT_DIR}/config/python_nodes.conf}"
export PYTHON_NODES_CONF
echo "[INFO] PYTHON_NODES_CONF=${PYTHON_NODES_CONF}"

# Runtime/perf knobs
model_trust_remote_code=${MODEL_TRUST_REMOTE_CODE:-False}
model_use_remove_padding=${MODEL_USE_REMOVE_PADDING:-True}
model_use_fused_kernels=${MODEL_USE_FUSED_KERNELS:-True}
actor_megatron_use_remove_padding=${ACTOR_MEGATRON_USE_REMOVE_PADDING:-True}
critic_megatron_use_remove_padding=${CRITIC_MEGATRON_USE_REMOVE_PADDING:-True}
actor_megatron_attention_backend=${ACTOR_MEGATRON_ATTENTION_BACKEND:-flash}
critic_megatron_attention_backend=${CRITIC_MEGATRON_ATTENTION_BACKEND:-flash}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
infer_micro_batch_size=${INFER_MICRO_BATCH_SIZE:-null}
train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE:-null}
offload=${OFFLOAD:-True}
param_offload=${PARAM_OFFLOAD:-${offload}}
grad_offload=${GRAD_OFFLOAD:-${offload}}
optimizer_offload=${OPTIMIZER_OFFLOAD:-True}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
rollout_tensor_parallel_size=${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-32768}
context_threshold=${CONTEXT_THRESHOLD:-16384}

qwen35_requires_no_thd=false
if [[ -f "${MODEL_PATH}/config.json" ]]; then
    if command -v rg >/dev/null 2>&1; then
        if rg -q '"model_type"\s*:\s*"qwen3_5(_moe)?' "${MODEL_PATH}/config.json" \
            || rg -q '"Qwen3_5(Moe)?ForConditionalGeneration"' "${MODEL_PATH}/config.json"; then
            qwen35_requires_no_thd=true
        fi
    elif grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"qwen3_5(_moe)?' "${MODEL_PATH}/config.json" \
        || grep -Eq '"Qwen3_5(Moe)?ForConditionalGeneration"' "${MODEL_PATH}/config.json"; then
        qwen35_requires_no_thd=true
    fi
fi
if [[ "${qwen35_requires_no_thd}" != "true" ]] && [[ "${MODEL_PATH}" == *Qwen3.5* || "${MODEL_PATH}" == *Qwen3_5* || "${MODEL_PATH}" == *qwen3.5* || "${MODEL_PATH}" == *qwen3_5* ]]; then
    qwen35_requires_no_thd=true
fi

if [[ "${qwen35_requires_no_thd}" == "true" ]]; then
    echo "[INFO] Detected Qwen3.5 model; disabling THD-dependent options for Megatron."
    model_trust_remote_code=True
    model_use_remove_padding=False
    model_use_fused_kernels=False
    actor_megatron_use_remove_padding=False
    critic_megatron_use_remove_padding=False
    actor_megatron_attention_backend=auto
    critic_megatron_attention_backend=auto
    if [[ -z "${ROLLOUT_TENSOR_PARALLEL_SIZE:-}" ]] && (( ROLLOUT_NGPUS_PER_NODE >= 2 )); then
        rollout_tensor_parallel_size=1
        echo "[INFO] Detected Qwen3.5 model; defaulting rollout TP to ${rollout_tensor_parallel_size} to reduce per-GPU vLLM memory pressure."
    fi
fi

if [[ "${use_dynamic_bsz}" == "False" || "${use_dynamic_bsz}" == "false" ]] && [[ "${train_micro_batch_size}" == "null" ]]; then
    train_micro_batch_size=1
    echo "[INFO] use_dynamic_bsz=False and TRAIN_MICRO_BATCH_SIZE is unset; defaulting actor micro batch size to ${train_micro_batch_size}."
fi

if [[ "${use_dynamic_bsz}" == "False" || "${use_dynamic_bsz}" == "false" ]] && [[ "${infer_micro_batch_size}" == "null" ]]; then
    infer_micro_batch_size=1
    echo "[INFO] use_dynamic_bsz=False and INFER_MICRO_BATCH_SIZE is unset; defaulting infer micro batch size to ${infer_micro_batch_size}."
fi

# Megatron parallelism/offload knobs for actor/ref/critic.
megatron_tp=${MEGATRON_TP:-4}
megatron_pp=${MEGATRON_PP:-2}
megatron_vpp=${MEGATRON_VPP:-null}
megatron_cp=${MEGATRON_CP:-1}
megatron_ep=${MEGATRON_EP:-2}
megatron_etp=${MEGATRON_ETP:-null}
megatron_use_mbridge=${MEGATRON_USE_MBRIDGE:-True}
megatron_vanilla_mbridge=${MEGATRON_VANILLA_MBRIDGE:-True}
megatron_use_dist_ckpt=${MEGATRON_USE_DIST_CKPT:-False}
actor_save_dist_opt_param_state=${ACTOR_SAVE_DIST_OPT_PARAM_STATE:-True}

ref_megatron_tp=${REF_MEGATRON_TP:-${megatron_tp}}
ref_megatron_pp=${REF_MEGATRON_PP:-${megatron_pp}}
ref_megatron_vpp=${REF_MEGATRON_VPP:-${megatron_vpp}}
ref_megatron_cp=${REF_MEGATRON_CP:-${megatron_cp}}
ref_megatron_ep=${REF_MEGATRON_EP:-${megatron_ep}}
ref_megatron_etp=${REF_MEGATRON_ETP:-${megatron_etp}}

critic_megatron_tp=${CRITIC_MEGATRON_TP:-${megatron_tp}}
critic_megatron_pp=${CRITIC_MEGATRON_PP:-${megatron_pp}}
critic_megatron_vpp=${CRITIC_MEGATRON_VPP:-${megatron_vpp}}
critic_megatron_cp=${CRITIC_MEGATRON_CP:-${megatron_cp}}
critic_megatron_ep=${CRITIC_MEGATRON_EP:-${megatron_ep}}
critic_megatron_etp=${CRITIC_MEGATRON_ETP:-${megatron_etp}}

# LoRA / PEFT (Megatron-Bridge only; requires vanilla_mbridge=False)
enable_lora=${ENABLE_LORA:-False}
lora_type=${LORA_TYPE:-lora}
lora_rank=${LORA_RANK:-32}
lora_alpha=${LORA_ALPHA:-64}
lora_dropout=${LORA_DROPOUT:-0.0}
lora_dtype=${LORA_DTYPE:-bfloat16}
lora_A_init_method=${LORA_A_INIT_METHOD:-kaiming}
lora_B_init_method=${LORA_B_INIT_METHOD:-zero}
lora_dropout_position=${LORA_DROPOUT_POSITION:-pre}
lora_exclude_modules=${LORA_EXCLUDE_MODULES:-"[]"}
lora_merge=${LORA_MERGE:-True}
actor_lr=${ACTOR_LR:-}

if [[ -z "${actor_lr}" ]]; then
    if [[ "${enable_lora,,}" == "true" ]]; then
        actor_lr=3e-6
    else
        actor_lr=1e-6
    fi
fi

LORA_OVERRIDES=()
if [[ "${enable_lora,,}" == "true" ]]; then
    if [[ "${megatron_use_mbridge}" != "True" && "${megatron_use_mbridge}" != "true" ]]; then
        echo "[ERROR] ENABLE_LORA=True requires MEGATRON_USE_MBRIDGE=True" >&2
        exit 1
    fi
    megatron_vanilla_mbridge=False
    LORA_OVERRIDES=(
        "actor_rollout_ref.model.lora.type='${lora_type}'"
        "actor_rollout_ref.model.lora.rank=${lora_rank}"
        "actor_rollout_ref.model.lora.alpha=${lora_alpha}"
        "actor_rollout_ref.model.lora.dropout=${lora_dropout}"
        "actor_rollout_ref.model.lora.dtype='${lora_dtype}'"
        "actor_rollout_ref.model.lora.lora_A_init_method='${lora_A_init_method}'"
        "actor_rollout_ref.model.lora.lora_B_init_method='${lora_B_init_method}'"
        "actor_rollout_ref.model.lora.dropout_position='${lora_dropout_position}'"
        "actor_rollout_ref.model.lora.exclude_modules=${lora_exclude_modules}"
        "actor_rollout_ref.model.lora.merge=${lora_merge}"
    )
    echo "[INFO] Enabling LoRA: type=${lora_type} rank=${lora_rank} alpha=${lora_alpha} dropout=${lora_dropout}"
fi
echo "[INFO] Actor LR set to ${actor_lr} (ENABLE_LORA=${enable_lora})"

export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# -----------------------------
# API keys (tools + summarizer)
# -----------------------------
export DEEPRESEARCH_PRINT_TURNS="${DEEPRESEARCH_PRINT_TURNS:-0}"
export SERPER_KEY_ID="${SERPER_KEY_ID:-}"
export JINA_API_KEY="${JINA_API_KEY:-}"
export JINA_API_KEYS="${JINA_API_KEYS:-${JINA_API_KEY}}"
export SANDBOX_FUSION_ENDPOINT="${SANDBOX_FUSION_ENDPOINT:-}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

export DEEPRESEARCH_PRINT_ROLLOUT="${DEEPRESEARCH_PRINT_ROLLOUT:-0}"
export DEEPRESEARCH_PRINT_ROLLOUT_MAX_SAMPLES="${DEEPRESEARCH_PRINT_ROLLOUT_MAX_SAMPLES:-2}"
export DEEPRESEARCH_PRINT_ROLLOUT_MAX_CHARS="${DEEPRESEARCH_PRINT_ROLLOUT_MAX_CHARS:-2000}"
export DEEPRESEARCH_DUMP_TRAJECTORY_JSONL="${DEEPRESEARCH_DUMP_TRAJECTORY_JSONL:-1}"
export DEEPRESEARCH_STREAM_DIR="${DEEPRESEARCH_STREAM_DIR:-${ROLLOUT_DATA_DIR}/stream}"

# Keep local eval-node failover responsive during hot updates.
export LOCAL_OPENAI_TIMEOUT_SECONDS="${LOCAL_OPENAI_TIMEOUT_SECONDS:-30}"
export LOCAL_OPENAI_MAX_RETRIES="${LOCAL_OPENAI_MAX_RETRIES:-25}"
export LOCAL_OPENAI_BUSY_COOLDOWN_SECONDS="${LOCAL_OPENAI_BUSY_COOLDOWN_SECONDS:-2.0}"
export LOCAL_OPENAI_RETRY_BACKOFF_SECONDS="${LOCAL_OPENAI_RETRY_BACKOFF_SECONDS:-0.2}"
export LOCAL_OPENAI_FALLBACK_ENABLED="${LOCAL_OPENAI_FALLBACK_ENABLED:-1}"
export LOCAL_OPENAI_FALLBACK_MAX_RETRIES="${LOCAL_OPENAI_FALLBACK_MAX_RETRIES:-10}"
export EVAL_PER_RESPONSE_TIMEOUT_SECONDS="${EVAL_PER_RESPONSE_TIMEOUT_SECONDS:-3600}"
export EVAL_VISIT_TIMEOUT_SECONDS="${EVAL_VISIT_TIMEOUT_SECONDS:-300}"
export DEEPRESEARCH_FILTER_PROMPT_BSZ="${DEEPRESEARCH_FILTER_PROMPT_BSZ:-${legacy_train_prompt_bsz}}"

if [[ -z "${JINA_API_KEY}" ]]; then
    echo "[WARN] JINA_API_KEY is empty. visit tool will fail and rollout can look abnormally fast."
fi

# Shared Azure/OpenAI-compatible API configuration.
# This launcher is Azure-first: OpenAI defaults are disabled, and compatibility
# variables below mirror the shared API/Azure settings for code paths that still
# read OPENAI_* names.
export API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
export API_BASE="${API_BASE:-}"
export AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-5-mini}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://jian-general.openai.azure.com/}"
export OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME:-${AZURE_OPENAI_DEPLOYMENT}}"
export SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-${OPENAI_MODEL_NAME}}"
export MEMORY_MODEL_NAME="${MEMORY_MODEL_NAME:-}"
export MEMORY_API_KEY="${MEMORY_API_KEY:-}"
export MEMORY_API_BASE="${MEMORY_API_BASE:-}"
export MEMORY_AZURE_ENDPOINT="${MEMORY_AZURE_ENDPOINT:-}"
export MEMORY_AZURE_API_VERSION="${MEMORY_AZURE_API_VERSION:-2024-12-01-preview}"
export MEMORY_AZURE_DEPLOYMENT="${MEMORY_AZURE_DEPLOYMENT:-}"
export MEMORY_TIMEOUT_SECONDS="${MEMORY_TIMEOUT_SECONDS:-120}"
export MEMORY_FALLBACK_MODEL_NAME="${MEMORY_FALLBACK_MODEL_NAME:-deepseek.v3.2}"
export MEMORY_FALLBACK_API_KEY="${MEMORY_FALLBACK_API_KEY:-}"
export MEMORY_FALLBACK_API_BASE="${MEMORY_FALLBACK_API_BASE:-https://bedrock-mantle.us-east-1.api.aws/v1}"
export MEMORY_FALLBACK_AZURE_ENDPOINT="${MEMORY_FALLBACK_AZURE_ENDPOINT:-}"
export MEMORY_FALLBACK_AZURE_API_VERSION="${MEMORY_FALLBACK_AZURE_API_VERSION:-2024-12-01-preview}"
export MEMORY_FALLBACK_AZURE_DEPLOYMENT="${MEMORY_FALLBACK_AZURE_DEPLOYMENT:-}"
export MEMORY_FALLBACK_TIMEOUT_SECONDS="${MEMORY_FALLBACK_TIMEOUT_SECONDS:-300}"
# Visit summarizer primary config: keep this chain independent from MEMORY_* / generic API_*.
export VISIT_SUMMARY_MODEL_NAME="${VISIT_SUMMARY_MODEL_NAME:-}"
export VISIT_SUMMARY_API_KEY="${VISIT_SUMMARY_API_KEY:-}"
export VISIT_SUMMARY_API_BASE="${VISIT_SUMMARY_API_BASE:-}"
export VISIT_SUMMARY_AZURE_ENDPOINT="${VISIT_SUMMARY_AZURE_ENDPOINT:-}"
export VISIT_SUMMARY_AZURE_API_VERSION="${VISIT_SUMMARY_AZURE_API_VERSION:-2024-12-01-preview}"
# Visit summarizer fallback config: also independent and explicit.
export VISIT_SUMMARY_FALLBACK_MODEL_NAME="${VISIT_SUMMARY_FALLBACK_MODEL_NAME:-}"
export VISIT_SUMMARY_FALLBACK_API_KEY="${VISIT_SUMMARY_FALLBACK_API_KEY:-}"
export VISIT_SUMMARY_FALLBACK_API_BASE="${VISIT_SUMMARY_FALLBACK_API_BASE:-}"
export VISIT_SUMMARY_FALLBACK_AZURE_ENDPOINT="${VISIT_SUMMARY_FALLBACK_AZURE_ENDPOINT:-}"
export VISIT_SUMMARY_FALLBACK_AZURE_API_VERSION="${VISIT_SUMMARY_FALLBACK_AZURE_API_VERSION:-2024-12-01-preview}"
export VISIT_SUMMARY_TIMEOUT_SECONDS="${VISIT_SUMMARY_TIMEOUT_SECONDS:-300}"
export MEMORY_LOCAL_FALLBACK_MODEL_NAME="${MEMORY_LOCAL_FALLBACK_MODEL_NAME:-}"
export MEMORY_LOCAL_FALLBACK_API_KEY="${MEMORY_LOCAL_FALLBACK_API_KEY:-}"
export MEMORY_LOCAL_FALLBACK_TIMEOUT_SECONDS="${MEMORY_LOCAL_FALLBACK_TIMEOUT_SECONDS:-300}"
export LOCAL_OPENAI_BASE_URLS="${LOCAL_OPENAI_BASE_URLS:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY}}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-${API_BASE}}"
export LOCAL_OPENAI_FALLBACK_API_KEY="${LOCAL_OPENAI_FALLBACK_API_KEY:-${MEMORY_FALLBACK_API_KEY}}"
export LOCAL_OPENAI_FALLBACK_API_BASE="${LOCAL_OPENAI_FALLBACK_API_BASE:-${MEMORY_FALLBACK_API_BASE}}"
export LOCAL_OPENAI_FALLBACK_MODEL_NAME="${LOCAL_OPENAI_FALLBACK_MODEL_NAME:-${MEMORY_FALLBACK_MODEL_NAME}}"
export LOCAL_OPENAI_FALLBACK_AZURE_ENDPOINT="${LOCAL_OPENAI_FALLBACK_AZURE_ENDPOINT:-}"
export LOCAL_OPENAI_FALLBACK_AZURE_API_VERSION="${LOCAL_OPENAI_FALLBACK_AZURE_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"
export LOCAL_OPENAI_FALLBACK_AZURE_DEPLOYMENT="${LOCAL_OPENAI_FALLBACK_AZURE_DEPLOYMENT:-${AZURE_OPENAI_DEPLOYMENT}}"
export LOCAL_OPENAI_SECONDARY_FALLBACK_API_KEY="${LOCAL_OPENAI_SECONDARY_FALLBACK_API_KEY:-${API_KEY}}"
export LOCAL_OPENAI_SECONDARY_FALLBACK_API_BASE="${LOCAL_OPENAI_SECONDARY_FALLBACK_API_BASE:-${API_BASE}}"
export LOCAL_OPENAI_SECONDARY_FALLBACK_MODEL_NAME="${LOCAL_OPENAI_SECONDARY_FALLBACK_MODEL_NAME:-${AZURE_OPENAI_DEPLOYMENT}}"
export LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_ENDPOINT="${LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_ENDPOINT:-${AZURE_OPENAI_ENDPOINT}}"
export LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_API_VERSION="${LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_API_VERSION:-${AZURE_OPENAI_API_VERSION}}"
export LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_DEPLOYMENT="${LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_DEPLOYMENT:-${AZURE_OPENAI_DEPLOYMENT}}"

# Eval LLM API-client configuration (for reward.py eval_llm).
export EVAL_LLM_PROVIDER="${EVAL_LLM_PROVIDER:-local_openai}"        # auto | vllm | local_openai | azure | openai
export EVAL_LLM_API_KEY="${EVAL_LLM_API_KEY:-}"
export EVAL_LLM_API_BASE="${EVAL_LLM_API_BASE:-}"
export EVAL_LLM_MODEL_NAME="${EVAL_LLM_MODEL_NAME:-}"
export EVAL_LLM_AZURE_ENDPOINT="${EVAL_LLM_AZURE_ENDPOINT:-}"
export EVAL_LLM_AZURE_API_VERSION="${EVAL_LLM_AZURE_API_VERSION:-2024-12-01-preview}"
export EVAL_LLM_AZURE_DEPLOYMENT="${EVAL_LLM_AZURE_DEPLOYMENT:-}"
export EVAL_LLM_FALLBACK_PROVIDER="${EVAL_LLM_FALLBACK_PROVIDER:-azure}"
export EVAL_LLM_FALLBACK_API_KEY="${EVAL_LLM_FALLBACK_API_KEY:-}"
export EVAL_LLM_FALLBACK_API_BASE="${EVAL_LLM_FALLBACK_API_BASE:-}"
export EVAL_LLM_FALLBACK_MODEL_NAME="${EVAL_LLM_FALLBACK_MODEL_NAME:-}"
export EVAL_LLM_FALLBACK_AZURE_ENDPOINT="${EVAL_LLM_FALLBACK_AZURE_ENDPOINT:-}"
export EVAL_LLM_FALLBACK_AZURE_API_VERSION="${EVAL_LLM_FALLBACK_AZURE_API_VERSION:-2024-12-01-preview}"
export EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT="${EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT:-}"
export EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME="${EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME:-}"
export EVAL_LLM_TIMEOUT_SECONDS="${EVAL_LLM_TIMEOUT_SECONDS:-120}"

# Optional per-task evaluators (reward.py -> openended_task_eval):
# - CITATION_EVAL_LLM_* for inline citation checks
# - OPENENDED_EVAL_LLM_* for open-ended rubric evaluation
# Defaults in this script:
# - citation: Azure gpt-5-mini -> Bedrock deepseek.v3.2 -> local eval model
# - open-ended: inherits shared eval defaults
export CITATION_EVAL_LLM_PROVIDER="${CITATION_EVAL_LLM_PROVIDER:-azure}"
export CITATION_EVAL_LLM_API_KEY="${CITATION_EVAL_LLM_API_KEY:-}"
export CITATION_EVAL_LLM_API_BASE="${CITATION_EVAL_LLM_API_BASE:-}"
export CITATION_EVAL_LLM_MODEL_NAME="${CITATION_EVAL_LLM_MODEL_NAME:-}"
export CITATION_EVAL_LLM_AZURE_ENDPOINT="${CITATION_EVAL_LLM_AZURE_ENDPOINT:-}"
export CITATION_EVAL_LLM_AZURE_API_VERSION="${CITATION_EVAL_LLM_AZURE_API_VERSION:-2024-12-01-preview}"
export CITATION_EVAL_LLM_AZURE_DEPLOYMENT="${CITATION_EVAL_LLM_AZURE_DEPLOYMENT:-}"
export CITATION_EVAL_LLM_FALLBACK_PROVIDER="${CITATION_EVAL_LLM_FALLBACK_PROVIDER:-api}"
export CITATION_EVAL_LLM_FALLBACK_API_KEY="${CITATION_EVAL_LLM_FALLBACK_API_KEY:-}"
export CITATION_EVAL_LLM_FALLBACK_API_BASE="${CITATION_EVAL_LLM_FALLBACK_API_BASE:-}"
export CITATION_EVAL_LLM_FALLBACK_MODEL_NAME="${CITATION_EVAL_LLM_FALLBACK_MODEL_NAME:-}"
export CITATION_EVAL_LLM_FALLBACK_AZURE_ENDPOINT="${CITATION_EVAL_LLM_FALLBACK_AZURE_ENDPOINT:-}"
export CITATION_EVAL_LLM_FALLBACK_AZURE_API_VERSION="${CITATION_EVAL_LLM_FALLBACK_AZURE_API_VERSION:-2024-12-01-preview}"
export CITATION_EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT="${CITATION_EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT:-}"
export CITATION_EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME="${CITATION_EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME:-}"
export CITATION_EVAL_LLM_TIMEOUT_SECONDS="${CITATION_EVAL_LLM_TIMEOUT_SECONDS:-120}"

export OPENENDED_EVAL_LLM_PROVIDER="${OPENENDED_EVAL_LLM_PROVIDER:-local_openai}"
export OPENENDED_EVAL_LLM_API_KEY="${OPENENDED_EVAL_LLM_API_KEY:-}"
export OPENENDED_EVAL_LLM_API_BASE="${OPENENDED_EVAL_LLM_API_BASE:-}"
export OPENENDED_EVAL_LLM_MODEL_NAME="${OPENENDED_EVAL_LLM_MODEL_NAME:-}"
export OPENENDED_EVAL_LLM_AZURE_ENDPOINT="${OPENENDED_EVAL_LLM_AZURE_ENDPOINT:-}"
export OPENENDED_EVAL_LLM_AZURE_API_VERSION="${OPENENDED_EVAL_LLM_AZURE_API_VERSION:-2024-12-01-preview}"
export OPENENDED_EVAL_LLM_AZURE_DEPLOYMENT="${OPENENDED_EVAL_LLM_AZURE_DEPLOYMENT:-}"
export OPENENDED_EVAL_LLM_FALLBACK_PROVIDER="${OPENENDED_EVAL_LLM_FALLBACK_PROVIDER:-azure}"
export OPENENDED_EVAL_LLM_FALLBACK_API_KEY="${OPENENDED_EVAL_LLM_FALLBACK_API_KEY:-}"
export OPENENDED_EVAL_LLM_FALLBACK_API_BASE="${OPENENDED_EVAL_LLM_FALLBACK_API_BASE:-}"
export OPENENDED_EVAL_LLM_FALLBACK_MODEL_NAME="${OPENENDED_EVAL_LLM_FALLBACK_MODEL_NAME:-}"
export OPENENDED_EVAL_LLM_FALLBACK_AZURE_ENDPOINT="${OPENENDED_EVAL_LLM_FALLBACK_AZURE_ENDPOINT:-}"
export OPENENDED_EVAL_LLM_FALLBACK_AZURE_API_VERSION="${OPENENDED_EVAL_LLM_FALLBACK_AZURE_API_VERSION:-2024-12-01-preview}"
export OPENENDED_EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT="${OPENENDED_EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT:-}"
export OPENENDED_EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME="${OPENENDED_EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME:-}"
export OPENENDED_EVAL_LLM_TIMEOUT_SECONDS="${OPENENDED_EVAL_LLM_TIMEOUT_SECONDS:-120}"

# Shared API key defaults used by summarizer, memory, and judges.
export GOOGLE_MAPS_API_KEY="${GOOGLE_MAPS_API_KEY:-}"

# HLE judge model (shared Azure/OpenAI-compatible API)
export HLE_JUDGE_MODEL_NAME="${HLE_JUDGE_MODEL_NAME:-${AZURE_OPENAI_DEPLOYMENT}}"

if [[ "${gen_prompt_bsz}" != "1" ]]; then
    echo "[WARN] GEN_PROMPT_BSZ=${gen_prompt_bsz}. fully_async streaming mode is usually tuned with GEN_PROMPT_BSZ=1."
fi

# Dynamic Curriculum Learning: sampler config
curriculum_norm="$(echo "${curriculum_enabled}" | tr '[:upper:]' '[:lower:]')"
if [[ "${curriculum_norm}" == "true" ]]; then
    CURRICULUM_SAMPLER_OVERRIDES=(
        "data.sampler.class_path='${WORKING_DIR}/recipe/deepresearch/curriculum_sampler.py'"
        "data.sampler.class_name='DynamicCurriculumSampler'"
        "data.dataloader_num_workers=0"
        "+data.curriculum.objective=${curriculum_objective}"
        "+data.curriculum.lr=${curriculum_lr}"
        "+data.curriculum.temperature=${curriculum_temperature}"
        "+data.curriculum.min_weight=${curriculum_min_weight}"
        "+data.curriculum.replacement=${curriculum_replacement}"
    )
    echo "[INFO] Dynamic Curriculum Learning ENABLED (objective=${curriculum_objective}, lr=${curriculum_lr}, temperature=${curriculum_temperature}, min_weight=${curriculum_min_weight}, replacement=${curriculum_replacement})"
else
    CURRICULUM_SAMPLER_OVERRIDES=()
    echo "[INFO] Dynamic Curriculum Learning DISABLED"
fi

# Non-schema Hydra extension overrides: always use '+' prefix.
NON_SCHEMA_OVERRIDES=(
    "+algorithm.filter_groups.enable=${enable_filter_groups}"
    "+algorithm.filter_groups.metric=${filter_groups_metric}"
    "+algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches}"
    "+custom_reward_function.reward_kwargs.eval_scripts_dir='${EVAL_SCRIPTS_DIR}'"
    "+custom_reward_function.reward_kwargs.eval_llm_ips='${EVAL_LLM_IPS}'"
    "+custom_reward_function.reward_kwargs.eval_llm_ports='${EVAL_LLM_PORTS}'"
    "+custom_reward_function.reward_kwargs.eval_llm_model='${EVAL_LLM_MODEL}'"
    "+custom_reward_function.reward_kwargs.eval_llm_nodes_conf='${EVAL_LLM_NODES_CONF}'"
)

cd "${WORKING_DIR}"
export PYTHONPATH="${WORKING_DIR}:${WORKING_DIR}/recipe/deepresearch:${PYTHONPATH:-}"
export RAY_ADDRESS

launcher_module="verl.experimental.fully_async_policy.fully_async_main"
DEBUG_OVERRIDES=()
debug_load_save_only_norm="$(echo "${debug_load_save_only}" | tr '[:upper:]' '[:lower:]')"
if [[ "${debug_load_save_only_norm}" == "true" ]]; then
    launcher_module="recipe.deepresearch.megatron_load_save_debug"
    DEBUG_OVERRIDES=(
        "+debug.global_step=${debug_global_step}"
        "+debug.load_checkpoint_path='${debug_load_checkpoint_path}'"
        "+debug.save_base_dir='${debug_save_base_dir}'"
    )
    if [[ -n "${debug_save_checkpoint_path}" ]]; then
        DEBUG_OVERRIDES+=("+debug.save_checkpoint_path='${debug_save_checkpoint_path}'")
    fi
    echo "[INFO] DEBUG_LOAD_SAVE_ONLY enabled; launcher_module=${launcher_module}"
    echo "[INFO] debug.load_checkpoint_path=${debug_load_checkpoint_path}"
    echo "[INFO] debug.save_base_dir=${debug_save_base_dir}"
    if [[ -n "${debug_save_checkpoint_path}" ]]; then
        echo "[INFO] debug.save_checkpoint_path=${debug_save_checkpoint_path}"
    fi
fi

python3 -m "${launcher_module}" \
    --config-name="${fully_async_config_name}" \
    "+ray_kwargs.ray_init.address='${RAY_ADDRESS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.SERPER_KEY_ID='${SERPER_KEY_ID}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.JINA_API_KEY='${JINA_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.JINA_API_KEYS='${JINA_API_KEYS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_FUSION_ENDPOINT='${SANDBOX_FUSION_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHON_NODES_CONF='${PYTHON_NODES_CONF}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.API_KEY='${API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.API_BASE='${API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.AZURE_OPENAI_ENDPOINT='${AZURE_OPENAI_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.AZURE_OPENAI_API_VERSION='${AZURE_OPENAI_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.AZURE_OPENAI_DEPLOYMENT='${AZURE_OPENAI_DEPLOYMENT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.SUMMARY_MODEL_NAME='${SUMMARY_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VISIT_SUMMARY_MODEL_NAME='${VISIT_SUMMARY_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VISIT_SUMMARY_API_KEY='${VISIT_SUMMARY_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VISIT_SUMMARY_API_BASE='${VISIT_SUMMARY_API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VISIT_SUMMARY_AZURE_ENDPOINT='${VISIT_SUMMARY_AZURE_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VISIT_SUMMARY_AZURE_API_VERSION='${VISIT_SUMMARY_AZURE_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VISIT_SUMMARY_FALLBACK_MODEL_NAME='${VISIT_SUMMARY_FALLBACK_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VISIT_SUMMARY_FALLBACK_API_KEY='${VISIT_SUMMARY_FALLBACK_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VISIT_SUMMARY_FALLBACK_API_BASE='${VISIT_SUMMARY_FALLBACK_API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VISIT_SUMMARY_FALLBACK_AZURE_ENDPOINT='${VISIT_SUMMARY_FALLBACK_AZURE_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VISIT_SUMMARY_FALLBACK_AZURE_API_VERSION='${VISIT_SUMMARY_FALLBACK_AZURE_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VISIT_SUMMARY_TIMEOUT_SECONDS='${VISIT_SUMMARY_TIMEOUT_SECONDS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_MODEL_NAME='${MEMORY_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_API_KEY='${MEMORY_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_API_BASE='${MEMORY_API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_AZURE_ENDPOINT='${MEMORY_AZURE_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_AZURE_API_VERSION='${MEMORY_AZURE_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_AZURE_DEPLOYMENT='${MEMORY_AZURE_DEPLOYMENT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_TIMEOUT_SECONDS='${MEMORY_TIMEOUT_SECONDS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_FALLBACK_MODEL_NAME='${MEMORY_FALLBACK_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_FALLBACK_API_KEY='${MEMORY_FALLBACK_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_FALLBACK_API_BASE='${MEMORY_FALLBACK_API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_FALLBACK_AZURE_ENDPOINT='${MEMORY_FALLBACK_AZURE_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_FALLBACK_AZURE_API_VERSION='${MEMORY_FALLBACK_AZURE_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_FALLBACK_AZURE_DEPLOYMENT='${MEMORY_FALLBACK_AZURE_DEPLOYMENT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_FALLBACK_TIMEOUT_SECONDS='${MEMORY_FALLBACK_TIMEOUT_SECONDS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_LOCAL_FALLBACK_MODEL_NAME='${MEMORY_LOCAL_FALLBACK_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_LOCAL_FALLBACK_API_KEY='${MEMORY_LOCAL_FALLBACK_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_LOCAL_FALLBACK_TIMEOUT_SECONDS='${MEMORY_LOCAL_FALLBACK_TIMEOUT_SECONDS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_BASE_URLS='${LOCAL_OPENAI_BASE_URLS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_IPS='${EVAL_LLM_IPS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_PORTS='${EVAL_LLM_PORTS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.DEEPRESEARCH_PRINT_ROLLOUT='${DEEPRESEARCH_PRINT_ROLLOUT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.DEEPRESEARCH_PRINT_ROLLOUT_MAX_SAMPLES='${DEEPRESEARCH_PRINT_ROLLOUT_MAX_SAMPLES}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.DEEPRESEARCH_PRINT_ROLLOUT_MAX_CHARS='${DEEPRESEARCH_PRINT_ROLLOUT_MAX_CHARS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.DEEPRESEARCH_FILTER_PROMPT_BSZ='${DEEPRESEARCH_FILTER_PROMPT_BSZ}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL='${VERL_LOGGING_LEVEL:-DEBUG}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.DEEPRESEARCH_PRINT_TURNS='${DEEPRESEARCH_PRINT_TURNS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.DEEPRESEARCH_DUMP_TRAJECTORY_JSONL='${DEEPRESEARCH_DUMP_TRAJECTORY_JSONL}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.DEEPRESEARCH_STREAM_DIR='${DEEPRESEARCH_STREAM_DIR}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1='${VLLM_USE_V1}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_ENABLE='${NCCL_CUMEM_ENABLE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG='${NCCL_DEBUG}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_ASYNC_ERROR_HANDLING='${NCCL_ASYNC_ERROR_HANDLING}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_NCCL_ASYNC_ERROR_HANDLING='${TORCH_NCCL_ASYNC_ERROR_HANDLING}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.TRITON_CACHE_DIR='${TRITON_CACHE_DIR}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.TORCHINDUCTOR_CACHE_DIR='${TORCHINDUCTOR_CACHE_DIR}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_KERNEL_CACHE_PATH='${PYTORCH_KERNEL_CACHE_PATH}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CC='${CC}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CXX='${CXX}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CUDAHOSTCXX='${CUDAHOSTCXX}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_HOME='${CUDA_HOME:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CUDACXX='${CUDACXX:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_NVCC_EXECUTABLE='${CUDA_NVCC_EXECUTABLE:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CPATH='${CPATH:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CPLUS_INCLUDE_PATH='${CPLUS_INCLUDE_PATH:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PATH='${PATH}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.XDG_CACHE_HOME='${XDG_CACHE_HOME:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR='${TMPDIR:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_CACHE_DIR='${FLASHINFER_CACHE_DIR}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_TMP_CACHE_DIR='${FLASHINFER_TMP_CACHE_DIR}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_EXTRA_CFLAGS='${FLASHINFER_EXTRA_CFLAGS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_EXTRA_CUDAFLAGS='${FLASHINFER_EXTRA_CUDAFLAGS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_TIMEOUT_SECONDS='${LOCAL_OPENAI_TIMEOUT_SECONDS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_MAX_RETRIES='${LOCAL_OPENAI_MAX_RETRIES}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_BUSY_COOLDOWN_SECONDS='${LOCAL_OPENAI_BUSY_COOLDOWN_SECONDS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_RETRY_BACKOFF_SECONDS='${LOCAL_OPENAI_RETRY_BACKOFF_SECONDS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_FALLBACK_ENABLED='${LOCAL_OPENAI_FALLBACK_ENABLED}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_FALLBACK_MAX_RETRIES='${LOCAL_OPENAI_FALLBACK_MAX_RETRIES}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_FALLBACK_API_KEY='${LOCAL_OPENAI_FALLBACK_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_FALLBACK_API_BASE='${LOCAL_OPENAI_FALLBACK_API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_FALLBACK_MODEL_NAME='${LOCAL_OPENAI_FALLBACK_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_FALLBACK_AZURE_ENDPOINT='${LOCAL_OPENAI_FALLBACK_AZURE_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_FALLBACK_AZURE_API_VERSION='${LOCAL_OPENAI_FALLBACK_AZURE_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_FALLBACK_AZURE_DEPLOYMENT='${LOCAL_OPENAI_FALLBACK_AZURE_DEPLOYMENT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_SECONDARY_FALLBACK_API_KEY='${LOCAL_OPENAI_SECONDARY_FALLBACK_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_SECONDARY_FALLBACK_API_BASE='${LOCAL_OPENAI_SECONDARY_FALLBACK_API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_SECONDARY_FALLBACK_MODEL_NAME='${LOCAL_OPENAI_SECONDARY_FALLBACK_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_ENDPOINT='${LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_API_VERSION='${LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_DEPLOYMENT='${LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_DEPLOYMENT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_PER_RESPONSE_TIMEOUT_SECONDS='${EVAL_PER_RESPONSE_TIMEOUT_SECONDS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_VISIT_TIMEOUT_SECONDS='${EVAL_VISIT_TIMEOUT_SECONDS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_PROVIDER='${EVAL_LLM_PROVIDER}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_API_KEY='${EVAL_LLM_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_API_BASE='${EVAL_LLM_API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_MODEL_NAME='${EVAL_LLM_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_AZURE_ENDPOINT='${EVAL_LLM_AZURE_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_AZURE_API_VERSION='${EVAL_LLM_AZURE_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_AZURE_DEPLOYMENT='${EVAL_LLM_AZURE_DEPLOYMENT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_FALLBACK_PROVIDER='${EVAL_LLM_FALLBACK_PROVIDER}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_FALLBACK_API_KEY='${EVAL_LLM_FALLBACK_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_FALLBACK_API_BASE='${EVAL_LLM_FALLBACK_API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_FALLBACK_MODEL_NAME='${EVAL_LLM_FALLBACK_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_FALLBACK_AZURE_ENDPOINT='${EVAL_LLM_FALLBACK_AZURE_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_FALLBACK_AZURE_API_VERSION='${EVAL_LLM_FALLBACK_AZURE_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT='${EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME='${EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EVAL_LLM_TIMEOUT_SECONDS='${EVAL_LLM_TIMEOUT_SECONDS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_PROVIDER='${CITATION_EVAL_LLM_PROVIDER}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_API_KEY='${CITATION_EVAL_LLM_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_API_BASE='${CITATION_EVAL_LLM_API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_MODEL_NAME='${CITATION_EVAL_LLM_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_AZURE_ENDPOINT='${CITATION_EVAL_LLM_AZURE_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_AZURE_API_VERSION='${CITATION_EVAL_LLM_AZURE_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_AZURE_DEPLOYMENT='${CITATION_EVAL_LLM_AZURE_DEPLOYMENT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_FALLBACK_PROVIDER='${CITATION_EVAL_LLM_FALLBACK_PROVIDER}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_FALLBACK_API_KEY='${CITATION_EVAL_LLM_FALLBACK_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_FALLBACK_API_BASE='${CITATION_EVAL_LLM_FALLBACK_API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_FALLBACK_MODEL_NAME='${CITATION_EVAL_LLM_FALLBACK_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_FALLBACK_AZURE_ENDPOINT='${CITATION_EVAL_LLM_FALLBACK_AZURE_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_FALLBACK_AZURE_API_VERSION='${CITATION_EVAL_LLM_FALLBACK_AZURE_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT='${CITATION_EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME='${CITATION_EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CITATION_EVAL_LLM_TIMEOUT_SECONDS='${CITATION_EVAL_LLM_TIMEOUT_SECONDS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_PROVIDER='${OPENENDED_EVAL_LLM_PROVIDER}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_API_KEY='${OPENENDED_EVAL_LLM_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_API_BASE='${OPENENDED_EVAL_LLM_API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_MODEL_NAME='${OPENENDED_EVAL_LLM_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_AZURE_ENDPOINT='${OPENENDED_EVAL_LLM_AZURE_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_AZURE_API_VERSION='${OPENENDED_EVAL_LLM_AZURE_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_AZURE_DEPLOYMENT='${OPENENDED_EVAL_LLM_AZURE_DEPLOYMENT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_FALLBACK_PROVIDER='${OPENENDED_EVAL_LLM_FALLBACK_PROVIDER}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_FALLBACK_API_KEY='${OPENENDED_EVAL_LLM_FALLBACK_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_FALLBACK_API_BASE='${OPENENDED_EVAL_LLM_FALLBACK_API_BASE}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_FALLBACK_MODEL_NAME='${OPENENDED_EVAL_LLM_FALLBACK_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_FALLBACK_AZURE_ENDPOINT='${OPENENDED_EVAL_LLM_FALLBACK_AZURE_ENDPOINT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_FALLBACK_AZURE_API_VERSION='${OPENENDED_EVAL_LLM_FALLBACK_AZURE_API_VERSION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT='${OPENENDED_EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME='${OPENENDED_EVAL_LLM_LOCAL_FALLBACK_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENENDED_EVAL_LLM_TIMEOUT_SECONDS='${OPENENDED_EVAL_LLM_TIMEOUT_SECONDS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENAI_API_KEY='${OPENAI_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.GOOGLE_MAPS_API_KEY='${GOOGLE_MAPS_API_KEY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.HLE_JUDGE_MODEL_NAME='${HLE_JUDGE_MODEL_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEMORY_ENABLED='${memory_enabled}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CONTEXT_THRESHOLD='${context_threshold}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MAX_TURN_RESPONSE_LENGTH='${max_turn_response_length}'" \
    "data.train_files='${TRAIN_FILE}'" \
    "${VALIDATION_OVERRIDES[@]}" \
    data.prompt_key=prompt \
    data.reward_fn_key=data_source \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=True \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.trust_remote_code=${model_trust_remote_code} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.hybrid_engine=False \
    "actor_rollout_ref.model.path='${MODEL_PATH}'" \
    actor_rollout_ref.model.trust_remote_code=${model_trust_remote_code} \
    actor_rollout_ref.model.use_remove_padding=${model_use_remove_padding} \
    actor_rollout_ref.model.use_fused_kernels=${model_use_fused_kernels} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=${actor_strategy} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.total_training_steps=${megatron_total_training_steps} \
    actor_rollout_ref.actor.optim.lr_decay_steps=${megatron_lr_decay_steps} \
    actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler=${use_checkpoint_opt_param_scheduler} \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_micro_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${full_sequence_token_budget} \
    actor_rollout_ref.actor.megatron.use_mbridge=${megatron_use_mbridge} \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=${megatron_vanilla_mbridge} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${megatron_use_dist_ckpt} \
    actor_rollout_ref.actor.megatron.use_remove_padding=${actor_megatron_use_remove_padding} \
    actor_rollout_ref.actor.megatron.param_offload=${param_offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${grad_offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${megatron_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${megatron_pp} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${megatron_vpp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${megatron_cp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${megatron_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${megatron_etp} \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=${actor_megatron_attention_backend} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.session_weight_correction=${session_weight_correction} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${full_sequence_token_budget} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size} \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=${megatron_vanilla_mbridge} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${megatron_use_dist_ckpt} \
    actor_rollout_ref.ref.megatron.param_offload=${param_offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ref_megatron_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ref_megatron_pp} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${ref_megatron_vpp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${ref_megatron_cp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ref_megatron_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ref_megatron_etp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    "actor_rollout_ref.rollout.agent.default_agent_loop='${DEFAULT_AGENT_LOOP}'" \
    "actor_rollout_ref.rollout.agent.agent_loop_config_path='${AGENT_LOOP_CONFIG_PATH}'" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_user_turns} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=${max_parallel_calls} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_response_length} \
    "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side='${tool_response_truncate_side}'" \
    "actor_rollout_ref.rollout.multi_turn.tool_config_path='${TOOL_CONFIG_PATH}'" \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tensor_parallel_size} \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${full_sequence_token_budget} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_generation_token_budget} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=${gdn_prefill_backend} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','extra','optimizer','hf_model']" \
    +actor_rollout_ref.actor.checkpoint.save_distributed_optimizer_parameter_state=${actor_save_dist_opt_param_state} \
    +actor_rollout_ref.actor.checkpoint.mbridge_config.distributed_filesystem=True \
    +actor_rollout_ref.actor.checkpoint.mbridge_config.memory_efficient=True \
    critic.strategy=${critic_strategy} \
    critic.optim.total_training_steps=${megatron_total_training_steps} \
    critic.optim.lr_decay_steps=${megatron_lr_decay_steps} \
    critic.optim.use_checkpoint_opt_param_scheduler=${use_checkpoint_opt_param_scheduler} \
    critic.megatron.vanilla_mbridge=${megatron_vanilla_mbridge} \
    critic.megatron.use_dist_checkpointing=${megatron_use_dist_ckpt} \
    critic.megatron.use_remove_padding=${critic_megatron_use_remove_padding} \
    critic.megatron.param_offload=${param_offload} \
    critic.megatron.grad_offload=${grad_offload} \
    critic.megatron.optimizer_offload=${optimizer_offload} \
    critic.megatron.tensor_model_parallel_size=${critic_megatron_tp} \
    critic.megatron.pipeline_model_parallel_size=${critic_megatron_pp} \
    critic.megatron.virtual_pipeline_model_parallel_size=${critic_megatron_vpp} \
    critic.megatron.context_parallel_size=${critic_megatron_cp} \
    critic.megatron.expert_model_parallel_size=${critic_megatron_ep} \
    critic.megatron.expert_tensor_parallel_size=${critic_megatron_etp} \
    ++critic.megatron.override_transformer_config.attention_backend=${critic_megatron_attention_backend} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.rollout_correction.bypass_mode=${rollout_correction_bypass_mode} \
    async_training.require_batches=${async_require_batches} \
    async_training.trigger_parameter_sync_step=${async_trigger_parameter_sync_step} \
    async_training.staleness_threshold=${async_staleness_threshold} \
    async_training.partial_rollout=${async_partial_rollout} \
    +async_training.reward_cancel_mode=${async_reward_cancel_mode} \
    async_training.checkpoint_engine.enable=${async_ckpt_enable} \
    async_training.checkpoint_engine.overlap_broadcast_and_consume=${async_ckpt_overlap} \
    async_training.checkpoint_engine.device_buffer_size_M=${async_ckpt_device_buffer_size_m} \
    rollout.nnodes=${NNODES} \
    rollout.n_gpus_per_node=${ROLLOUT_NGPUS_PER_NODE} \
    rollout.total_rollout_steps=${total_rollout_steps} \
    rollout.total_epochs=${rollout_total_epochs} \
    rollout.test_freq=${rollout_test_freq} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${TRAINER_NGPUS_PER_NODE} \
    trainer.save_freq=5 \
    trainer.total_epochs=10 \
    "trainer.default_local_dir='${CKPTS_DIR}'" \
    "trainer.rollout_data_dir='${ROLLOUT_DATA_DIR}'" \
    trainer.resume_mode=${resume_mode} \
    "${RESUME_PATH_OVERRIDES[@]}" \
    trainer.logger='["console","wandb"]' \
    "trainer.project_name='${project_name}'" \
    "trainer.experiment_name='${exp_name}'" \
    "custom_reward_function.path='${WORKING_DIR}/recipe/deepresearch/reward.py'" \
    "${DEBUG_OVERRIDES[@]}" \
    "${NON_SCHEMA_OVERRIDES[@]}" \
    "${CURRICULUM_SAMPLER_OVERRIDES[@]}" \
    "${LORA_OVERRIDES[@]}" \
    reward_manager.source=importlib \
    reward_manager.name=DeepResearchRewardManager \
    reward_manager.module.path=pkg://recipe.deepresearch.reward_manager \
    reward_model.reward_manager=deepresearch \
    reward_model.reward_loop_source=importlib \
    reward_model.reward_loop_module_path=pkg://recipe.deepresearch.reward_loop_manager \
    reward_model.reward_loop_class_name=DeepResearchRewardLoopManager
