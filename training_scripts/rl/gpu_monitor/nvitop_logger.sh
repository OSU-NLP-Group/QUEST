#!/usr/bin/env bash
# 每 5 秒执行一次 nvitop，结果写入本目录

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${1:-$SCRIPT_DIR}"
mkdir -p "$LOGDIR"
LOG_FILE="$LOGDIR/nvitop_$(date +%Y%m%d).log"

echo "nvitop 每 5s 记录一次，输出到: $LOG_FILE (按 Ctrl+C 停止)"

while true; do
  echo "" >> "$LOG_FILE"
  echo "========== $(date -Iseconds) ==========" >> "$LOG_FILE"
  nvitop -1 >> "$LOG_FILE" 2>&1
  sleep 5
done
