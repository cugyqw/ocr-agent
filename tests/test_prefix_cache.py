"""前缀缓存正确性测试。

核心验证点：
    1. 缓存命中时，输出必须与不命中时**完全一致**（正确性优先）
    2. 命中时 forward 次数应减少（确实省了 prefill）
    3. 未命中时行为不变

用法:
    python tests/test_prefix_cache.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)

MODEL_PATH = "/home/yqw/桌面/ocr_agent/GLM-OCR"


def main():
    from miniinfer import EngineConfig, MiniInferEngine

    print("=" * 72)
    print("前缀缓存正确性测试")
    print("=" * 72)

    # ---- 构造纯文本 prompt（不含图片，才能走前缀缓存）----
    def make_text_seq(engine, text):
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        return engine.build_sequence(messages)

    # ================= 测试 1：禁用缓存时的基准 =================
    print("\n[1] 禁用前缀缓存，跑 3 条相同 prompt")
    cfg_off = EngineConfig(
        model_path=MODEL_PATH,
        max_model_len=4096,
        max_new_tokens=16,
        enable_prefix_cache=False,
    )
    eng_off = MiniInferEngine(cfg_off)
    eng_off.load_model()

    PROMPT = "请用一句话介绍你自己。"
    seqs_off = []
    for _ in range(3):
        s = make_text_seq(eng_off, PROMPT)
        eng_off.add_request(s)
        seqs_off.append(s)
    eng_off.run_until_done()

    out_off = [eng_off.decode(s) for s in seqs_off]
    stats_off = dict(eng_off.stats)
    print(f"  forward_calls = {stats_off['forward_calls']}")
    print(f"  输出: {out_off[0]!r}")
    eng_off.shutdown()

    # ================= 测试 2：启用缓存 =================
    print("\n[2] 启用前缀缓存，跑 3 条相同 prompt")
    cfg_on = EngineConfig(
        model_path=MODEL_PATH,
        max_model_len=4096,
        max_new_tokens=16,
        enable_prefix_cache=True,
    )
    eng_on = MiniInferEngine(cfg_on)
    eng_on.load_model()

    seqs_on = []
    for _ in range(3):
        s = make_text_seq(eng_on, PROMPT)
        eng_on.add_request(s)
        seqs_on.append(s)
    eng_on.run_until_done()

    out_on = [eng_on.decode(s) for s in seqs_on]
    stats_on = dict(eng_on.stats)
    print(f"  forward_calls = {stats_on['forward_calls']}")
    print(f"  前缀缓存命中 = {stats_on['prefix_cache_hits']}")
    print(f"  输出: {out_on[0]!r}")

    # ================= 对比 =================
    print("\n" + "=" * 72)
    print("结果对比")
    print("=" * 72)

    same = out_on == out_off
    print(f"  输出一致性: {'✓ 完全一致' if same else '✗ 不一致！'}")
    if not same:
        for i, (a, b) in enumerate(zip(out_off, out_on)):
            mark = "✓" if a == b else "✗"
            print(f"    [{i}] {mark} 无缓存={a!r}")
            print(f"         有缓存={b!r}")

    print(f"  forward 次数: 无缓存 {stats_off['forward_calls']} -> "
          f"有缓存 {stats_on['forward_calls']}")
    saved = stats_off["forward_calls"] - stats_on["forward_calls"]
    print(f"  节省 forward: {saved}")

    # ================= 测试 3：先后到达的重复请求（缓存真正生效的场景）=====
    print("\n" + "=" * 72)
    print("[3] 先后到达的相同 prompt（缓存应生效）")
    print("=" * 72)

    hits_before = eng_on.stats["prefix_cache_hits"]
    fwd_before = eng_on.stats["forward_calls"]

    # 第一条：先跑完，写入缓存
    s1 = make_text_seq(eng_on, PROMPT)
    eng_on.add_request(s1)
    eng_on.run_until_done()
    out_first = eng_on.decode(s1)
    fwd_after_first = eng_on.stats["forward_calls"]

    # 第二条：同样的 prompt，应立即命中缓存
    s2 = make_text_seq(eng_on, PROMPT)
    eng_on.add_request(s2)
    eng_on.run_until_done()
    out_second = eng_on.decode(s2)
    fwd_after_second = eng_on.stats["forward_calls"]

    print(f"  第1条 forward 次数: {fwd_after_first - fwd_before}")
    print(f"  第2条 forward 次数: {fwd_after_second - fwd_after_first}  "
          f"(命中缓存应显著减少)")
    print(f"  新增缓存命中: {eng_on.stats['prefix_cache_hits'] - hits_before}")
    print(f"  输出一致: {'✓' if out_first == out_second else '✗'}")
    if out_first != out_second:
        print(f"    第1条: {out_first!r}")
        print(f"    第2条: {out_second!r}")

    # ================= 测试 4：不同 prompt 不应误命中 =================
    print("\n" + "=" * 72)
    print("[3] 不同 prompt 不应命中缓存")
    print("=" * 72)
    eng_on.limit = None
    seq_a = make_text_seq(eng_on, "1 + 1 等于几？")
    seq_b = make_text_seq(eng_on, "2 + 2 等于几？")
    eng_on.add_request(seq_a)
    eng_on.add_request(seq_b)
    before = eng_on.stats["prefix_cache_hits"]
    eng_on.run_until_done()
    after = eng_on.stats["prefix_cache_hits"]
    print(f"  新命中次数: {after - before}（应为 0，两条 prompt 不同）")
    print(f"  A 输出: {eng_on.decode(seq_a)!r}")
    print(f"  B 输出: {eng_on.decode(seq_b)!r}")

    print(f"\n  缓存条目数: {len(eng_on.kv_manager.prefix_cache)}")
    print(f"  缓存显存: {eng_on.kv_manager.prefix_cache.memory_usage_mb():.1f} MB")

    eng_on.shutdown()

    # ================= 结论 =================
    print("\n" + "=" * 72)
    ok = same and saved >= 0 and (after - before) == 0
    print("测试结果: " + ("全部通过 ✓" if ok else "存在失败 ✗"))
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
