#!/usr/bin/env bash
# 下载评测数据集（CMMLU 小学相关科目）
#
# 用法:
#   bash download_datasets.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

echo "=============================================="
echo "下载 CMMLU 数据集（含小学语文/数学/常识等）"
echo "=============================================="

"$PYTHON" -m modelscope download \
    --dataset opencompass/cmmlu \
    --local_dir ./datasets/cmmlu

echo
echo "完成: ./datasets/cmmlu"
echo "查看小学相关科目:"
ls ./datasets/cmmlu 2>/dev/null | grep -i elementary || true
