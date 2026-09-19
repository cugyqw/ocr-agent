"""基线实测：评估解题模型在小学题目上的表现。

目的：
    1. 对比 Math 版 vs 通用版在小学语文/英语题上的差异
    2. 记录错误清单，作为 RAG 知识库的数据需求依据

用法：
    python bench_baseline.py                # 测通用版
    python bench_baseline.py --model math   # 测 Math 版
    python bench_baseline.py --only poem    # 只测古诗类
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.WARNING)

MODELS = {
    "general": "/home/yqw/桌面/ocr_agent/Qwen2.5-1.5B-Instruct",
    "math": "/home/yqw/桌面/ocr_agent/Qwen2.5-Math-1.5B-Instruct",
}

# ---------------------------------------------------------------------------
# 测试题库：小学六年级以内
# 每题标注参考答案关键点（用于人工核对，不做自动判分——
# 因为语言题答案多样，自动判分不可靠）
# ---------------------------------------------------------------------------
PROBLEMS = [
    # ---- 古诗记忆 ----
    ("poem", "古诗-默写", "请默写《静夜思》全诗，并说明作者和朝代。",
     "李白（唐）；床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
    ("poem", "古诗-作者", "《春晓》的作者是谁？请写出这首诗。",
     "孟浩然（唐）；春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"),
    ("poem", "古诗-理解", "「谁知盘中餐，粒粒皆辛苦」出自哪首诗？作者是谁？表达了什么道理？",
     "《悯农》李绅（唐）；珍惜粮食、尊重劳动。"),

    # ---- 文学常识 ----
    ("literature", "常识-作品", "《西游记》的作者是谁？主要讲了什么故事？",
     "吴承恩（明）；唐僧师徒四人西天取经。"),
    ("literature", "常识-成语", "请解释成语「守株待兔」的意思，并说明它告诉我们什么道理。",
     "守着树桩等兔子；比喻死守经验不知变通，不能靠侥幸。"),

    # ---- 语文基础 ----
    ("chinese", "语文-近义词", "请写出下列词语的近义词：\n1. 高兴\n2. 立刻\n3. 仔细\n4. 美丽",
     "高兴→开心/快乐；立刻→马上/立即；仔细→认真/细心；美丽→漂亮/好看。"),
    ("chinese", "语文-反义词", "请写出下列词语的反义词：\n1. 骄傲\n2. 危险\n3. 节约\n4. 熟悉",
     "骄傲→谦虚；危险→安全；节约→浪费；熟悉→陌生。"),
    ("chinese", "语文-修辞", "指出下列句子使用的修辞手法：\n1. 弯弯的月亮像一只小船。\n2. 小鸟在枝头唱歌。",
     "1. 比喻；2. 拟人。"),

    # ---- 英语 ----
    ("english", "英语-语法", "Fill in the blank with the correct form:\n1. I ___ (be) a student.\n2. She ___ (go) to school every day.",
     "1. am；2. goes。"),
    ("english", "英语-翻译", "请把下面的英文翻译成中文：\n1. I like playing football with my friends.\n2. What time do you get up?",
     "1. 我喜欢和朋友们一起踢足球。2. 你几点起床？"),
    ("english", "英语-单词", "请写出下列单词的中文意思：\n1. apple  2. teacher  3. beautiful  4. run", 
     "1. 苹果；2. 老师；3. 美丽的；4. 跑。"),

    # ---- 数学（对照，预期 Math 版更好）----
    ("math", "数学-方程", "解方程：2x + 5 = 13，并写出解题步骤。", "x = 4"),
    ("math", "数学-应用题", "一个长方形的长是 8 厘米，宽是 5 厘米。求它的周长和面积。",
     "周长 26 厘米，面积 40 平方厘米。" ),
    ("math", "数学-分数", "计算：1/2 + 1/3 = ?", "5/6"),
]

SYSTEM_PROMPT = """你是一位小学老师，负责解答小学六年级以内的题目。
请准确作答，解答要清晰，符合小学生的理解水平。使用中文作答。"""


def load(model_kind: str):
    path = MODELS[model_kind]
    logging.warning("加载模型: %s", path)
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda"
    )
    mdl.eval()
    return mdl, tok


def ask(model, tok, question: str, max_new_tokens: int = 512) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    elapsed = time.time() - t0

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip(), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["general", "math"], default="general")
    parser.add_argument("--only", default=None, help="只测某类: poem/literature/chinese/english/math")
    parser.add_argument("--out", default=None, help="结果保存路径(JSON)")
    args = parser.parse_args()

    problems = PROBLEMS
    if args.only:
        problems = [p for p in PROBLEMS if p[0] == args.only]
        if not problems:
            print(f"没有类别: {args.only}")
            return

    model, tok = load(args.model)

    print("=" * 74)
    print(f"基线实测  模型={args.model}  题数={len(problems)}")
    print("=" * 74)

    records = []
    total_time = 0.0
    by_cat: dict[str, list] = {}

    for i, (cat, name, q, ref) in enumerate(problems, 1):
        print(f"\n{'=' * 74}")
        print(f"[{i}/{len(problems)}] {name}")
        print("=" * 74)
        print(f"题目: {q}")
        print("-" * 74)

        ans, el = ask(model, tok, q)
        total_time += el

        print(f"模型回答:\n{ans}")
        print("-" * 74)
        print(f"参考答案要点: {ref}")
        print(f"耗时: {el:.2f}s")

        records.append({
            "category": cat, "name": name, "question": q,
            "answer": ans, "reference": ref, "time": el,
        })
        by_cat.setdefault(cat, []).append(el)

    print("\n" + "=" * 74)
    print("汇总")
    print("=" * 74)
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均耗时: {total_time / len(problems):.2f}s/题")
    print()
    print(f"{'类别':<14}{'题数':>6}{'平均耗时':>12}")
    print("-" * 74)
    for cat, times in by_cat.items():
        print(f"{cat:<14}{len(times):>6}{sum(times) / len(times):>11.2f}s")

    if args.out:
        Path(args.out).write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
