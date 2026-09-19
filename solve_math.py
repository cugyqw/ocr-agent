"""拍照解题 —— 命令行入口。

支持小学阶段各科：数学、语文、英语。
自动识别科目并路由到对应模型。

用法：
    # 单张图（自动识别科目）
    python solve_math.py /tmp/problem.png

    # 多张图
    python solve_math.py q1.png q2.png q3.png

    # 强制指定科目（跳过自动识别）
    python solve_math.py problem.png --subject math

    # 附加要求
    python solve_math.py problem.png -i "只给出最终答案"

流程：图片 → GLM-OCR 识别 → 科目识别 → 对应模型解题
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
    parser = argparse.ArgumentParser(description="拍照解题（OCR + 多科目推理）")
    parser.add_argument("images", nargs="+", help="题目图片路径")
    parser.add_argument("-i", "--instruction", default="", help="附加给解题模型的要求")
    parser.add_argument("--no-formula", action="store_true",
                        help="跳过公式识别（更快，但可能漏公式）")
    parser.add_argument("--subject", choices=["math", "chinese", "english", "general"],
                        help="强制指定科目，默认自动识别")
    args = parser.parse_args()

    for p in args.images:
        if not p.startswith("http") and not Path(p).exists():
            print(f"错误：文件不存在 {p}", file=sys.stderr)
            sys.exit(1)

    print("=" * 70)
    print("初始化流水线")
    print("  OCR: GLM-OCR（常驻）")
    print("  解题: 数学→Qwen2.5-Math / 其他→Qwen2.5 通用（按需加载）")
    print("=" * 70)

    pipeline = MathSolverPipeline(
        use_formula_prompt=not args.no_formula,
        force_subject=args.subject,
    )
    pipeline.load()

    try:
        if len(args.images) == 1:
            results = [pipeline.solve(args.images[0], args.instruction)]
        else:
            results = pipeline.solve_many(args.images, args.instruction)

        for i, r in enumerate(results, 1):
            print()
            print("=" * 70)
            print(f"[{i}/{len(results)}] {r.image}   科目: {r.subject_name}")
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
