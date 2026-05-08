#!/bin/bash
# WideSearch QUEST-35B-SFT+Midtrain+RL run.
# Uses gpt-5-mini for visit summaries / memory condensation and keeps API
# credentials in api_config.yaml or the caller environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${INFERENCE_DIR}/.." && pwd)"
QUEST_INFER_DIR="${QUEST_INFER_DIR:-${INFERENCE_DIR}}"
WIDESEARCH_DIR="${WIDESEARCH_DIR:-${REPO_ROOT}/evaluation/widesearch}"

load_api_config() {
    local config_file="$1"
    if [ ! -f "$config_file" ]; then
        echo "Error: API config file not found: ${config_file}" >&2
        exit 1
    fi

    eval "$(
        python3 - "$config_file" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = json.load(f)

for key, value in config.get("common", {}).items():
    if value is None:
        value = ""
    elif isinstance(value, bool):
        value = "true" if value else "false"
    else:
        value = str(value)
    print(f"export {key}={shlex.quote(value)}")
PY
    )"
}

export API_CONFIG_FILE="${API_CONFIG_FILE:-${INFERENCE_DIR}/api_config.yaml}"
load_api_config "$API_CONFIG_FILE"
echo "Loaded API config from ${API_CONFIG_FILE}"

export WIDESEARCH_LANGS="${WIDESEARCH_LANGS:-en}"
export WIDESEARCH_RUN_SLOT="${WIDESEARCH_RUN_SLOT:-1}"
export ROLLOUT_COUNT="${ROLLOUT_COUNT:-1}"

export MODEL_NAME="${MODEL_NAME:-deepresearch}"
export MODEL_PATH="${MODEL_PATH:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
export MEMORY_TOKENIZER_PATH="${MEMORY_TOKENIZER_PATH:-${MODEL_PATH}}"
export SERVER_ENDPOINTS_FILE="${SERVER_ENDPOINTS_FILE:-${INFERENCE_DIR}/server_endpoints.conf}"

export MAX_TURN="${MAX_TURN:-400}"
export MAX_LLM_CALL_PER_RUN="${MAX_LLM_CALL_PER_RUN:-${MAX_TURN}}"
export MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-80000}"
export MEMORY_CONTEXT_THRESHOLD="${MEMORY_CONTEXT_THRESHOLD:-80000}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-32000}"
export ENABLE_PYTHON_TOOL="${ENABLE_PYTHON_TOOL:-false}"
export ENABLE_SCHOLAR_TOOL="${ENABLE_SCHOLAR_TOOL:-true}"
export MEMORY_ENABLED="${MEMORY_ENABLED:-true}"
export MEMORY_STRATEGY="${MEMORY_STRATEGY:-condenser}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.1}"

# Keep summary / memory on gpt-5-mini.
export AZURE_OPENAI_DEPLOYMENT="gpt-5-mini"
export SUMMARY_MODEL_NAME="gpt-5-mini"
export MEMORY_MODEL_NAME="gpt-5-mini"
export MEMORY_OPENAI_API_KEY="${MEMORY_OPENAI_API_KEY:-${MEMORY_API_KEY:-${API_KEY:-}}}"
export EVAL_OPENAI_API_KEY="${EVAL_OPENAI_API_KEY:-${API_KEY:-}}"

CONFIG="35b_sft_midtrain_rl_mem80000_out32000_turn400_run_gpt-5-mini_slot${WIDESEARCH_RUN_SLOT}"
SWEEP_ROOT="${WIDESEARCH_SWEEP_ROOT:-${INFERENCE_DIR}/outputs/widesearch}"

_WL=$(echo "$WIDESEARCH_LANGS" | tr '[:upper:]' '[:lower:]' | tr ',' '_')
case "$_WL" in
    en)
        WIDESEARCH_QUEST_BASENAME="widesearch_en_input.jsonl"
        WIDESEARCH_EXPECTED_N=100
        ;;
    zh)
        WIDESEARCH_QUEST_BASENAME="widesearch_zh_input.jsonl"
        WIDESEARCH_EXPECTED_N=100
        ;;
    both|en_zh)
        WIDESEARCH_LANGS=both
        WIDESEARCH_QUEST_BASENAME="widesearch_en_zh_input.jsonl"
        WIDESEARCH_EXPECTED_N=200
        ;;
    *)
        echo "Invalid WIDESEARCH_LANGS=${WIDESEARCH_LANGS}; use en, zh, or both" >&2
        exit 1
        ;;
esac
export WIDESEARCH_LANGS
export WIDESEARCH_EXPECTED_N
export QUEST_INPUT="${QUEST_INPUT:-${WIDESEARCH_DIR}/${WIDESEARCH_QUEST_BASENAME}}"
export QUEST_OUTPUT="${QUEST_OUTPUT:-${SWEEP_ROOT}/${CONFIG}/quest_output}"
export RESPONSE_DIR="${RESPONSE_DIR:-${SWEEP_ROOT}/${CONFIG}/responses}"
export RESULT_DIR="${RESULT_DIR:-${SWEEP_ROOT}/${CONFIG}/results}"
export CACHE_DIR="${CACHE_DIR:-${SWEEP_ROOT}/${CONFIG}/cache}"
LOG_DIR="${LOG_DIR:-${SWEEP_ROOT}/${CONFIG}/logs}"

export TASK_LOG_DIR="${TASK_LOG_DIR:-${LOG_DIR}}"
export SEARCH_CACHE_FILE="${SEARCH_CACHE_FILE:-${CACHE_DIR}/search_cache.db}"
export VISIT_CACHE_FILE="${VISIT_CACHE_FILE:-${CACHE_DIR}/visit_cache.db}"

mkdir -p "$QUEST_OUTPUT" "$RESPONSE_DIR" "$RESULT_DIR" "$CACHE_DIR" "$LOG_DIR"

echo "============================================"
echo "  QUEST-35B-SFT+Midtrain+RL (out32K)"
echo "  sweep: ${CONFIG}"
echo "  WideSearch subset: ${WIDESEARCH_LANGS} (expect ${WIDESEARCH_EXPECTED_N})"
echo "  summary/memory: gpt-5-mini"
echo "  QUEST_INPUT: ${QUEST_INPUT}"
echo "  QUEST_OUTPUT: ${QUEST_OUTPUT}"
echo "============================================"

if [ ! -f "$QUEST_INPUT" ]; then
    cd "$WIDESEARCH_DIR"
    mkdir -p "$SWEEP_ROOT"
    PYTHONPATH="$WIDESEARCH_DIR" WIDESEARCH_LANGS="$WIDESEARCH_LANGS" QUEST_INPUT="$QUEST_INPUT" uv run python3 -c "
import json
import os
from src.evaluation.data_loader import WideSearchDataLoaderHF
loader = WideSearchDataLoaderHF()
langs = os.environ.get('WIDESEARCH_LANGS', 'en').strip().lower().replace(',', '_')
if langs in ('both', 'en_zh'):
    prefixes = ('ws_en_', 'ws_zh_')
elif langs == 'zh':
    prefixes = ('ws_zh_',)
else:
    prefixes = ('ws_en_',)
ids = sorted(iid for iid in loader.get_instance_id_list() if any(iid.startswith(p) for p in prefixes))
with open(os.environ['QUEST_INPUT'], 'w', encoding='utf-8') as f:
    for iid in ids:
        q = loader.load_query_by_instance_id(iid)
        f.write(json.dumps({'question': q.query, 'answer': '', 'filename': iid}, ensure_ascii=False) + '\n')
print(f'{len(ids)} queries written -> {os.environ[\"QUEST_INPUT\"]}')
"
fi

infer_done=1
dataset_stem="${WIDESEARCH_QUEST_BASENAME%.jsonl}"
for ((r = 1; r <= ROLLOUT_COUNT; r++)); do
    iter_file="$QUEST_OUTPUT/$MODEL_NAME/$dataset_stem/iter${r}.jsonl"
    cnt=0
    if [ -f "$iter_file" ]; then cnt=$(wc -l < "$iter_file"); fi
    if [ "$cnt" -lt "$WIDESEARCH_EXPECTED_N" ]; then infer_done=0; break; fi
done

if [ "$infer_done" -eq 1 ]; then
    echo "[Skip] Inference done for ${ROLLOUT_COUNT} rollout(s)."
else
    INFER_MAX_WORKERS="${INFER_MAX_WORKERS:-32}"
    echo "[Infer] Running with max_workers=${INFER_MAX_WORKERS}..."
    cd "$QUEST_INFER_DIR"
    python -u run_multi_react.py \
        --dataset "$QUEST_INPUT" \
        --output "$QUEST_OUTPUT" \
        --max_workers "$INFER_MAX_WORKERS" \
        --model "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --temperature "$TEMPERATURE" \
        --presence_penalty "$PRESENCE_PENALTY" \
        --roll_out_count "$ROLLOUT_COUNT" \
        2>&1 | tee "$LOG_DIR/infer.log"
fi

echo "[Convert]..."
cd "$WIDESEARCH_DIR"
PYTHONPATH="$WIDESEARCH_DIR" uv run python3 << 'PYEOF'
import glob
import json
import os
import re

quest_output = os.environ["QUEST_OUTPUT"]
model_name = os.environ["MODEL_NAME"]
response_dir = os.environ["RESPONSE_DIR"]
patterns = [
    os.path.join(quest_output, model_name, "**", "iter1*.jsonl"),
    os.path.join(quest_output, "**", "iter1*.jsonl"),
    os.path.join(quest_output, model_name, "**", "recovered_iter1_from_trajectories.jsonl"),
    os.path.join(quest_output, "**", "recovered_iter1_from_trajectories.jsonl"),
]
files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
seen = set()
converted = 0
for path in files:
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        iid = d.get("filename") or d.get("instance_id", "")
        if not iid or iid in seen:
            continue
        seen.add(iid)
        pred = d.get("prediction", "")
        if len(pred.strip()) <= 1:
            for msg in reversed(d.get("messages", [])):
                if msg.get("role") == "assistant":
                    pred = str(msg.get("content", ""))
                    break
        match = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL)
        resp = match.group(1).strip() if match else pred
        ws = {
            "instance_id": iid,
            "response": resp,
            "messages": [
                {"role": "user", "content": d.get("question", "")},
                {"role": "assistant", "content": {
                    "step_status": "FINISHED",
                    "content": resp,
                    "reasoning_content": None,
                    "signature": None,
                    "tool_calls": [],
                    "tool_call_results": [],
                    "error_marker": None,
                }},
            ],
            "trial_idx": 0,
        }
        with open(os.path.join(response_dir, f"quest_{iid}_0_response.jsonl"), "w", encoding="utf-8") as out:
            out.write(json.dumps(ws, ensure_ascii=False) + "\n")
        converted += 1
print(f"  Converted {converted}")
PYEOF

if [ "${RUN_WIDESEARCH_EVAL:-true}" = "true" ]; then
    if [ -z "${EVAL_OPENAI_API_KEY:-}" ]; then
        echo "[Eval] Skipped because EVAL_OPENAI_API_KEY/API_KEY is not set."
        exit 0
    fi
    echo "[Eval]..."
    iids=$(WIDESEARCH_LANGS="${WIDESEARCH_LANGS}" python3 -c "
import os
langs = os.environ.get('WIDESEARCH_LANGS', 'en').strip().lower().replace(',', '_')
ids = []
if langs in ('en', 'both', 'en_zh'):
    ids += [f'ws_en_{i:03d}' for i in range(1, 101)]
if langs in ('zh', 'both', 'en_zh'):
    ids += [f'ws_zh_{i:03d}' for i in range(1, 101)]
print(','.join(ids))
")
    export OPENAI_API_KEY="$EVAL_OPENAI_API_KEY"
    PYTHONPATH="$WIDESEARCH_DIR" python3 "$WIDESEARCH_DIR/run_widesearch_eval.py" \
        --model_config_name=quest --instance_id="$iids" --trial_num=1 \
        --eval_model_config_name=gpt-5-mini-eval --response_root="$RESPONSE_DIR" \
        --result_save_root="$RESULT_DIR" --thread_num=4 --use_cache \
        2>&1 | tee "$LOG_DIR/eval.log" | tail -5
fi

summary_file="$RESULT_DIR/quest_trial_num_1_summary.json"
if [ -f "$summary_file" ]; then
    python3 -c "import json; d=json.load(open('$summary_file')); print(f'item F1 = {d[\"f1_by_item\"][\"avg_n\"]*100:.1f}%')"
fi
