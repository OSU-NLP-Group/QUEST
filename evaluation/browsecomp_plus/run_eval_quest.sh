#!/bin/bash
# Convert QUEST BrowseComp-Plus outputs and run the BrowseComp-Plus judge.
# Credentials are loaded from inference/api_config.yaml or the caller environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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

EVAL_FORCE_ARGS=()
if [ "${FORCE_EVAL:-0}" = "1" ] || [ "${FORCE_REEVAL:-0}" = "1" ]; then
    EVAL_FORCE_ARGS=(--force)
fi

resolve_run_root() {
    if [ -n "${RUN_ROOT:-}" ]; then
        printf '%s\n' "$RUN_ROOT"
        return 0
    fi

    python3 - "$REPO_ROOT" <<'PY'
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
base = repo_root / "inference" / "outputs" / "browsecomp_plus"
if not base.is_dir():
    raise SystemExit(1)

candidates = []
for path in base.iterdir():
    if not path.is_dir():
        continue
    data_dir = path / "deepresearch" / "browsecomp_plus_quest_130"
    if any(data_dir.glob("iter*.jsonl")):
        candidates.append(path)

if not candidates:
    raise SystemExit(1)

candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(candidates[0])
PY
}

RUN_ROOT="$(resolve_run_root)" || {
    echo "Error: no evaluation input found; set RUN_ROOT explicitly" >&2
    exit 1
}
DATASET_NAME="${DATASET_NAME:-browsecomp_plus_quest_130}"
JSONL_DIR="${JSONL_DIR:-${RUN_ROOT}/deepresearch/${DATASET_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-${SCRIPT_DIR}/converted_runs/quest}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${SCRIPT_DIR}/evals/quest}"
GROUND_TRUTH="${GROUND_TRUTH:-${REPO_ROOT}/data/browsecomp_plus/browsecomp_plus_decrypted.jsonl}"
ROLLOUT_COUNT="${ROLLOUT_COUNT:-3}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini}"

if [ ! -f "$GROUND_TRUTH" ]; then
    echo "Error: ground truth file not found" >&2
    exit 1
fi

for ITER in $(seq 1 "$ROLLOUT_COUNT"); do
    INPUT_JSONL="${JSONL_DIR}/iter${ITER}.jsonl"
    EVAL_DIR="${EVAL_ROOT}/iter${ITER}"

    if [ ! -f "$INPUT_JSONL" ]; then
        echo "=== Skipping iter${ITER}: input not found ==="
        continue
    fi

    echo "=== Convert iter${ITER} to judge format ==="
    python3 - "$INPUT_JSONL" "$EVAL_DIR" <<'PY'
import json
import os
import sys

jsonl_path, eval_dir = sys.argv[1], sys.argv[2]
os.makedirs(eval_dir, exist_ok=True)
for name in os.listdir(eval_dir):
    if name.endswith(".json"):
        os.remove(os.path.join(eval_dir, name))

count = 0
with open(jsonl_path, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        qid = item["filename"]
        prediction = item.get("prediction", "") or ""
        out = {
            "query_id": qid,
            "tool_call_counts": {"search": item.get("num_rounds", 0)},
            "status": "completed" if prediction.strip() else "incomplete",
            "retrieved_docids": [],
            "result": [{"type": "output_text", "output": prediction}],
        }
        with open(os.path.join(eval_dir, f"run_{qid}.json"), "w", encoding="utf-8") as wf:
            json.dump(out, wf, ensure_ascii=False)
        count += 1
print(f"Converted {count} files")
PY

    if [ "${SKIP_EVAL:-0}" = "1" ]; then
        continue
    fi

    echo "=== Evaluate iter${ITER} ==="
    cd "$SCRIPT_DIR"
    uv run python scripts_evaluation/evaluate_with_openai.py \
        --input_dir "$EVAL_DIR" \
        --ground_truth "$GROUND_TRUTH" \
        --eval_dir "$EVAL_OUTPUT_ROOT" \
        --model "$JUDGE_MODEL" \
        "${EVAL_FORCE_ARGS[@]}"
done
