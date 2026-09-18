#!/usr/bin/env bash
# 一键下载本项目所需的模型权重（走 ModelScope，国内速度较好）
#
# 用法:
#   bash download_models.sh              # 下载全部
#   bash download_models.sh ocr          # 只下 GLM-OCR
#   bash download_models.sh math         # 只下 Qwen2.5-Math

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 允许通过环境变量覆盖 Python 解释器
PYTHON="${PYTHON:-python3}"

# 检查 modelscope 是否可用
if ! "$PYTHON" -c "import modelscope" 2>/dev/null; then
    echo "未检测到 modelscope，正在安装..."
    "$PYTHON" -m pip install modelscope
fi

TARGET="${1:-all}"

download_ocr() {
    echo "=============================================="
    echo "下载 GLM-OCR（约 2.5GB）"
    echo "=============================================="
    "$PYTHON" -m modelscope download \
        --model ZhipuAI/GLM-OCR \
        --local_dir ./GLM-OCR
    echo "完成: ./GLM-OCR"
}

download_math() {
    echo "=============================================="
    echo "下载 Qwen2.5-Math-1.5B-Instruct（约 2.9GB）"
    echo "=============================================="
    "$PYTHON" -m modelscope download \
        --model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --local_dir ./Qwen2.5-Math-1.5B-Instruct
    echo "完成: ./Qwen2.5-Math-1.5B-Instruct"
}

case "$TARGET" in
    ocr)  download_ocr ;;
    math) download_math ;;
    all)  download_ocr; download_math ;;
    *)
        echo "未知参数: $TARGET"
        echo "用法: bash download_models.sh [all|ocr|math]"
        exit 1
        ;;
esac

echo
echo "全部完成。可以运行: $PYTHON solve_math.py <图片路径>"
