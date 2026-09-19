"""清洗后数据集的评测脚本。

对 CMMLU 小学科目（清洗后 835 题）做自动评测：
    - 四选一单选题，比对模型输出与标准答案
    - 记录准确率与耗时
    - 支持知识库开关（为后续 RAG A/B 对比预留）

用法:
    # 全量评测
    python bench_clean.py

    # 只测某一科
    python bench_clean.py --subject chinese

    # 抽样评测（快速验证）
    python bench_clean.py --limit 50

    # 换模型
    python bench_clean.py --model math
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.WARNING)

CLEAN_DIR = Path(__file__).parent / "datasets" / "clean"

MODELS = {
    "general": "/home/yqw/桌面/ocr_agent/Qwen2.5-1.5B-Instruct",
    "math": "/home/yqw/桌面/ocr_agent/Qwen2.5-Math-1.5B-Instruct",
}

SUBJECTS = {
    "chinese": "elementary_chinese.csv",
    "math": "elementary_mathematics.csv",
    "commonsense": "elementary_commonsense.csv",
    "it": "elementary_information_and_technology.csv",
}

# 提示词：要求模型先推理再作答。
# 注意两个坑：
#   1. 如果只要求输出答案，模型会退化成"直接猜字母"，耗时降到 0.06s，评测失真。
#   2. 如果说"简要分析"，模型会敷衍两句就选，仍是猜。
# 因此要求它完整解题、必要时计算，再给结论。
SYSTEM_PROMPT = """你是一位知识渊博的小学老师，请认真回答下面的单项选择题。

要求：
1. 仔细读懂题目，逐项分析四个选项，说明判断依据
2. 涉及计算的要实际算一遍，不要靠感觉猜
3. 最后一行输出答案，格式严格为：答案：X
   其中 X 是 A、B、C、D 中的一个字母

示例：
题目：中国的首都是哪里？
A. 上海  B. 北京  C. 广州  D. 深圳
分析：中国的首都是北京。A 上海是直辖市但非首都，C 广州、D 深圳均为南方城市。
答案：B"""


def build_prompt(q: dict) -> str:
    """把 CSV 行转成题目文本。"""
    return (
        f"题目：{q['Question']}\n"
        f"A. {q['A']}  B. {q['B']}  C. {q['C']}  D. {q['D']}"
    )


def parse_answer(text: str) -> str | None:
    """从模型输出里提取选项字母。"""
    # 优先匹配 "答案：X" 格式
    m = re.search(r"答案\s*[:：]\s*([ABCD])\b", text)
    if m:
        return m.group(1)

    # 兜底：找最后一个独立的 A/B/C/D
    m = re.findall(r"(?:^|[\s：:（(])([ABCD])(?:[\s。，,、）)]|$)", text)
    if m:
        return m[-1]

    # 再兜底：任何位置的孤立字母
    m = re.findall(r"\b([ABCD])\b", text)
    return m[-1] if m else None


def load_subject(name: str, limit: int | None, seed: int = 42) -> list[dict]:
    path = CLEAN_DIR / SUBJECTS[name]
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit and limit < len(rows):
        random.seed(seed)
        rows = random.sample(rows, limit)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default="general")
    parser.add_argument("--subject", choices=list(SUBJECTS), default=None,
                        help="只测某一科，默认测全部")
    parser.add_argument("--limit", type=int, default=None, help="每科最多测多少题")
    parser.add_argument("--out", default=None, help="结果保存路径(JSON)")
    parser.add_argument("--verbose", action="store_true", help="打印每题的作答")
    args = parser.parse_args()

    subjects = [args.subject] if args.subject else list(SUBJECTS)

    # ---- 加载模型 ----
    model_path = MODELS[args.model]
    print("=" * 74)
    print(f"加载模型: {model_path}")
    print("=" * 74)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    all_records = []
    summary = {}

    for subj in subjects:
        rows = load_subject(subj, args.limit)
        print(f"\n{'=' * 74}")
        print(f"评测科目: {subj}  ({len(rows)} 题)")
        print("=" * 74)

        correct = 0
        unparsed = 0
        total_time = 0.0
        records = []

        for i, r in enumerate(rows, 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(r)},
            ]
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tok(text, return_tensors="pt").to("cuda")

            t0 = time.time()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    # 给足空间让模型完整推理。设太小会截断，导致答案缺失、
                    # 耗时统计失真，评测结果不可信。
                    max_new_tokens=768,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            el = time.time() - t0
            total_time += el

            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            reply = tok.decode(new_tokens, skip_special_tokens=True).strip()

            pred = parse_answer(reply)
            gold = r["Answer"].strip()
            ok = pred == gold
            if pred is None:
                unparsed += 1
            if ok:
                correct += 1

            records.append({
                "subject": subj,
                "question": r["Question"],
                "options": {k: r[k] for k in "ABCD"},
                "gold": gold,
                "pred": pred,
                "correct": ok,
                "time": el,
                "raw_reply": reply,
            })

            if args.verbose:
                mark = "✓" if ok else ("?" if pred is None else "✗")
                print(f"[{i}/{len(rows)}] {mark} gold={gold} pred={pred}")
                if not ok:
                    print(f"     {r['Question'][:60]}")

            # 进度
            if i % 50 == 0:
                print(f"  进度 {i}/{len(rows)}  准确率 {correct / i * 100:.1f}%")

        acc = correct / len(rows) * 100 if rows else 0
        avg_t = total_time / len(rows) if rows else 0
        summary[subj] = {
            "n": len(rows),
            "correct": correct,
            "accuracy": acc,
            "unparsed": unparsed,
            "avg_time": avg_t,
            "total_time": total_time,
        }
        all_records.extend(records)

        print(f"\n  {subj}: {correct}/{len(rows)} = {acc:.1f}%  "
              f"(未解析 {unparsed}, 平均 {avg_t:.2f}s/题)")

    # ---- 汇总 ----
    print(f"\n{'=' * 74}")
    print(f"汇总  模型={args.model}")
    print("=" * 74)
    print(f"{'科目':<28}{'题数':>6}{'正确':>6}{'准确率':>10}{'平均耗时':>12}")
    print("-" * 74)

    tot_n = tot_c = 0
    for subj, s in summary.items():
        print(f"{subj:<28}{s['n']:>6}{s['correct']:>6}"
              f"{s['accuracy']:>9.1f}%{s['avg_time']:>11.2f}s")
        tot_n += s["n"]
        tot_c += s["correct"]

    print("-" * 74)
    overall = tot_c / tot_n * 100 if tot_n else 0
    print(f"{'合计':<28}{tot_n:>6}{tot_c:>6}{overall:>9.1f}%")
    print()

    if args.out:
        Path(args.out).write_text(
            json.dumps({"summary": summary, "records": all_records},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
