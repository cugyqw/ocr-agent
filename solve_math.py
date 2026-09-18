"""拍照解题 —— 命令行入口。

用法：
    # 单张图
    python solve_math.py /tmp/math1.png

    # 多张图
    python solve_math.py /tmp/math1.png /tmp/math2.png

    # 附加要求
    python solve_math.py /tmp/math1.png -i "请只给出最终答案"

流程：图片 → GLM-OCR 识别（文字+公式）→ Qwen2.5-Math 解题
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from solver import MathSolverPipeline


def main():
    parser = argparse.ArgumentParser(description="拍照解题（OCR + 数学推理）")
    parser.add_argument("images", nargs="+", help="题目图片路径")
    parser.add_argument("-i", "--instruction", default="", help="附加给解题模型的要求")
    parser.add_argument("--no-formula", action="store_true",
                        help="跳过公式识别（更快，但可能漏公式）")
    args = parser.parse_args()

    # 校验文件存在
    for p in args.images:
        if not p.startswith("http") and not Path(p).exists():
            print(f"错误：文件不存在 {p}", file=sys.stderr)
            sys.exit(1)

    print("=" * 70)
    print("初始化流水线（GLM-OCR + Qwen2.5-Math，两模型常驻显存）")
    print("=" * 70)

    pipeline = MathSolverPipeline(use_formula_prompt=not args.no_formula)
    pipeline.load()

    try:
        if len(args.images) == 1:
            results = [pipeline.solve(args.images[0], args.instruction)]
        else:
            results = pipeline.solve_many(args.images, args.instruction)

        for i, r in enumerate(results, 1):
            print()
            print("=" * 70)
            print(f"[{i}/{len(results)}] {r.image}")
            print("=" * 70)
            print("\n--- OCR 识别（文字）---")
            print(r.ocr_text or "(无)")
            if r.ocr_formula:
                print("\n--- OCR 识别（公式）---")
                print(r.ocr_formula)
            print("\n--- 解答 ---")
            print(r.answer)
            print("\n--- 耗时 ---")
            for k, v in r.timings.items():
                print(f"  {k:<14} {v:.2f}s")
    finally:
        pipeline.unload()


if __name__ == "__main__":
    main()
