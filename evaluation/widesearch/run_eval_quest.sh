#!/bin/bash
# Convert QUEST WideSearch outputs and evaluate them with the WideSearch scorer.
# Credentials are loaded from inference/api_config.yaml or the caller environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WIDESEARCH_DIR="${WIDESEARCH_DIR:-${REPO_ROOT}/evaluation/widesearch}"

load_api_config() {
    local config_file="$1"
    if [ ! -f "$config_file" ]; then
        echo "Error: API config file not found" >&2
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

export API_CONFIG_FILE="${API_CONFIG_FILE:-${REPO_ROOT}/inference/api_config.yaml}"
load_api_config "$API_CONFIG_FILE"
echo "Loaded API config"

export AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-5-mini}"
export API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY:-}}"

export WIDESEARCH_LANGS="${WIDESEARCH_LANGS:-en}"
export WIDESEARCH_RUN_SLOT="${WIDESEARCH_RUN_SLOT:-1}"
export MODEL_NAME="${MODEL_NAME:-deepresearch}"
export ROLLOUT_COUNT="${ROLLOUT_COUNT:-1}"

SWEEP_ROOT="${WIDESEARCH_SWEEP_ROOT:-${REPO_ROOT}/inference/outputs/widesearch}"
CONFIG="${CONFIG:-}"
if [ -z "$CONFIG" ]; then
    CONFIG="$(
        python3 - "$SWEEP_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit(1)

candidates = [p for p in root.iterdir() if p.is_dir() and (p / "quest_output").is_dir()]
if not candidates:
    raise SystemExit(1)

candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(candidates[0].name)
PY
    )" || {
        echo "Error: no evaluation input found; set CONFIG explicitly" >&2
        exit 1
    }
fi

QUEST_OUTPUT="${QUEST_OUTPUT:-${SWEEP_ROOT}/${CONFIG}/quest_output}"
RESPONSE_DIR="${RESPONSE_DIR:-${SWEEP_ROOT}/${CONFIG}/responses}"
RESULT_DIR="${RESULT_DIR:-${SWEEP_ROOT}/${CONFIG}/results}"
JUDGE_CONFIG="${JUDGE_CONFIG:-default_eval_config}"
THREAD_NUM="${THREAD_NUM:-4}"

_WL=$(echo "$WIDESEARCH_LANGS" | tr '[:upper:]' '[:lower:]' | tr ',' '_')
case "$_WL" in
    en)
        WIDESEARCH_QUEST_BASENAME="widesearch_en_input.jsonl"
        EXPECTED_IDS=$(python3 -c "print(','.join([f'ws_en_{i:03d}' for i in range(1, 101)]))")
        ;;
    zh)
        WIDESEARCH_QUEST_BASENAME="widesearch_zh_input.jsonl"
        EXPECTED_IDS=$(python3 -c "print(','.join([f'ws_zh_{i:03d}' for i in range(1, 101)]))")
        ;;
    both|en_zh)
        WIDESEARCH_QUEST_BASENAME="widesearch_en_zh_input.jsonl"
        EXPECTED_IDS=$(python3 -c "print(','.join([f'ws_en_{i:03d}' for i in range(1, 101)] + [f'ws_zh_{i:03d}' for i in range(1, 101)]))")
        ;;
    *)
        echo "Invalid WIDESEARCH_LANGS; use en, zh, or both" >&2
        exit 1
        ;;
esac

if [ "${SKIP_CONVERT:-0}" != "1" ] || [ "${SKIP_EVAL:-0}" != "1" ]; then
    mkdir -p "$RESPONSE_DIR" "$RESULT_DIR"
fi

echo "============================================"
echo "  WideSearch QUEST eval"
echo "  subset: ${WIDESEARCH_LANGS}"
echo "============================================"

if [ "${SKIP_CONVERT:-0}" != "1" ]; then
    echo "[Convert] QUEST iter files -> WideSearch response files"
    export QUEST_OUTPUT RESPONSE_DIR MODEL_NAME
    PYTHONPATH="$SCRIPT_DIR" python3 << 'PY'
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
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            iid = d.get("filename") or d.get("instance_id", "")
            if not iid or iid in seen:
                continue
            seen.add(iid)
            pred = d.get("prediction", "") or ""
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
            out = os.path.join(response_dir, f"quest_{iid}_0_response.jsonl")
            with open(out, "w", encoding="utf-8") as wf:
                wf.write(json.dumps(ws, ensure_ascii=False) + "\n")
            converted += 1
print(f"Converted {converted} responses")
PY
fi

if [ "${SKIP_EVAL:-0}" = "1" ]; then
    echo "SKIP_EVAL=1; conversion finished."
    exit 0
fi

PYTHONPATH="$SCRIPT_DIR" python3 "$SCRIPT_DIR/run_widesearch_eval.py" \
    --model_config_name=quest \
    --instance_id="$EXPECTED_IDS" \
    --trial_num=1 \
    --eval_model_config_name="$JUDGE_CONFIG" \
    --response_root="$RESPONSE_DIR" \
    --result_save_root="$RESULT_DIR" \
    --thread_num="$THREAD_NUM" \
    --use_cache

summary_file="$RESULT_DIR/quest_trial_num_1_summary.json"
if [ -f "$summary_file" ]; then
    python3 -c "import json; d=json.load(open('$summary_file')); print(f'item F1 = {d[\"f1_by_item\"][\"avg_n\"]*100:.1f}%')"
fi
