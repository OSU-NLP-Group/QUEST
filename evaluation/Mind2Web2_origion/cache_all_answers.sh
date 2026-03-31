#!/usr/bin/env bash
set -euo pipefail

# Avoid iterating a literal "*" when no match
shopt -s nullglob

# Agent can be passed as the first arg; default to example
AGENT="${1:-example}"

# Resolve repo-relative answers root (works on any machine)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ANS_ROOT="${SCRIPT_DIR}/answers/${AGENT}"

# Collect task directories
dirs=( "${ANS_ROOT}"/*/ )
if [ ${#dirs[@]} -eq 0 ]; then
  echo "No tasks found under ${ANS_ROOT}. Check agent name and path." >&2
  exit 1
fi

for d in "${dirs[@]}"; do
  task_id="$(basename "$d")"
  echo ">>> Running: ${AGENT}/${task_id}"
  python batch_answer_cache.py "${AGENT}" "${task_id}" --headless
done
