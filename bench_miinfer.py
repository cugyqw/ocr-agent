"""miniinfer vs 朴素 transformers 的吞吐对比。

基准场景：同一张图，N 条不同问句并发请求。
对比对象：
  A) 朴素方式：每条请求各跑一次 model.generate()（相当于 run.py 的写法）
  B) miniinfer：连续批处理 + 批量 decode

指标：总耗时、forward 次数、吞吐（tokens/s）
"""

import logging
import time

import torch

logging.basicConfig(level=logging.WARNING)

MODEL_PATH = "/home/yqw/桌面/ocr_agent/GLM-OCR"
IMAGE = "/tmp/test_ocr.png"
MAX_NEW_TOKENS = 24

QUESTIONS = [
    "Text Recognition:",
    "What is the total amount?",
    "Text Recognition:",
    "Text Recognition:",
]


def build_messages(q):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "url": IMAGE},
            {"type": "text", "text": q},
        ],
    }]


def bench_naive(model, processor, warmup=True):
    """朴素方式：逐条 generate，模型常驻但无批处理。

    这是与 miniinfer 最公平的对比基线：两者都只加载一次模型，
    差别只在于"是否把并发请求批处理到一起"。

    Args:
        warmup: 先跑一次预热，排除首次 kernel 编译等一次性开销。
    """
    if warmup:
        inputs = processor.apply_chat_template(
            build_messages(QUESTIONS[0]),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    t0 = time.time()
    n_tokens = 0
    for q in QUESTIONS:
        inputs = processor.apply_chat_template(
            build_messages(q),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        n_tokens += out.shape[1] - inputs["input_ids"].shape[1]
    return time.time() - t0, n_tokens


def bench_miniinfer():
    """miniinfer：连续批处理。"""
    from miniinfer import EngineConfig, MiniInferEngine

    engine = MiniInferEngine(EngineConfig(
        model_path=MODEL_PATH,
        max_batch_size=4,
        max_new_tokens=MAX_NEW_TOKENS,
    ))
    engine.load_model()

    # 预热一次，排除首次编译/缓存影响
    warm = engine.build_sequence(build_messages(QUESTIONS[0]))
    engine.add_request(warm)
    engine.run_until_done()

    engine.stats = {k: 0 for k in engine.stats}

    seqs = []
    t0 = time.time()
    for q in QUESTIONS:
        s = engine.build_sequence(build_messages(q))
        engine.add_request(s)
        seqs.append(s)
    engine.run_until_done()
    elapsed = time.time() - t0

    n_tokens = sum(len(s.output_ids) for s in seqs)
    stats = dict(engine.stats)
    results = [engine.decode(s) for s in seqs]
    engine.shutdown()
    return elapsed, n_tokens, stats, results


def main():
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    print("=" * 66)
    print("加载模型（两种方式共用同一份权重）")
    print("=" * 66)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    print("\n" + "=" * 66)
    print("A) 朴素方式：逐条 model.generate()")
    print("=" * 66)
    t_naive, n_naive = bench_naive(model, processor)
    print(f"总耗时:      {t_naive:.2f}s")
    print(f"生成 tokens: {n_naive}")
    print(f"吞吐:        {n_naive / t_naive:.1f} tokens/s")

    # 释放显存，避免影响下一轮
    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 66)
    print("B) miniinfer：连续批处理 + 批量 decode")
    print("=" * 66)
    t_mini, n_mini, stats, results = bench_miniinfer()
    print(f"总耗时:      {t_mini:.2f}s")
    print(f"生成 tokens: {n_mini}")
    print(f"吞吐:        {n_mini / t_mini:.1f} tokens/s")
    print(f"forward 次数: {stats['forward_calls']}")
    print(f"输出样例:    {results[0]!r}")

    print("\n" + "=" * 66)
    print("对比")
    print("=" * 66)
    print(f"{'指标':<16}{'朴素':>12}{'miniinfer':>14}{'提升':>12}")
    print("-" * 66)
    print(f"{'总耗时(s)':<16}{t_naive:>12.2f}{t_mini:>14.2f}"
          f"{t_naive / t_mini:>11.2f}x")
    print(f"{'吞吐(tok/s)':<16}{n_naive / t_naive:>12.1f}{n_mini / t_mini:>14.1f}"
          f"{(n_mini / t_mini) / (n_naive / t_naive):>11.2f}x")


if __name__ == "__main__":
    main()
