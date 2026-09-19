"""推理引擎主体。

职责：
1. 模型常驻显存（解决 run.py 每次重载 2.5G 的问题）
2. 驱动 step 循环：调度 -> forward -> 采样 -> 更新状态
3. 混合 prefill/decode：一个 step 里新进序列走 prefill，老序列走 decode

对比 vLLM：vLLM 有专门的 prefill/decode 分离（chunked prefill），
这里简化为同一个 batch 内混合处理，靠 attention_mask 屏蔽。

关于 CUDA 层：本引擎不自己写 kernel，注意力直接走模型内置的
SDPA/FlashAttention 分支，由 PyTorch 提供。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers.cache_utils import DynamicCache

from .config import EngineConfig
from .kv_cache import BatchKVCache
from .sampler import Sampler
from .scheduler import Scheduler
from .sequence import Sequence

logger = logging.getLogger(__name__)


class MiniInferEngine:
    """轻量推理引擎。

    典型用法::

        engine = MiniInferEngine(EngineConfig())
        engine.load_model()
        seq = engine.build_sequence(messages)
        engine.add_request(seq)
        engine.run_until_done()
        print(engine.decode(seq))
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = torch.device(self.config.device)

        self.scheduler = Scheduler(self.config)
        self.sampler = Sampler(
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )
        self.kv_manager: Optional[BatchKVCache] = None

        self._seq_counter = 0
        self._id_lock = threading.Lock()
        self._eos_token_ids: List[int] = []

        # 统计
        self.stats: Dict[str, float] = {
            "steps": 0,
            "forward_calls": 0,
            "tokens_generated": 0,
            "prefill_batches": 0,
        }

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """加载模型并常驻显存。只调用一次。"""
        if self.model is not None:
            return

        t0 = time.time()
        logger.info("加载模型: %s", self.config.model_path)

        dtype = getattr(torch, self.config.dtype)
        self.tokenizer = AutoProcessor.from_pretrained(self.config.model_path)
        # AutoProcessor 对多模态模型返回 processor；纯文本模型退回 tokenizer
        self.processor = self.tokenizer

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_path,
            dtype=dtype,
            device_map=self.config.device if self.config.device != "cpu" else None,
        )
        self.model.eval()

        if self.config.use_torch_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:  # noqa: BLE001
                logger.warning("torch.compile 失败，回退 eager: %s", e)

        # 收集 EOS token
        gen_cfg = getattr(self.model, "generation_config", None)
        eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg else None
        if eos is None:
            eos = getattr(self.model.config, "eos_token_id", None)
        if eos is None:
            eos = 59253
        self._eos_token_ids = eos if isinstance(eos, (list, tuple)) else [eos]

        # KV Cache 管理器
        self.kv_manager = BatchKVCache(self.config, self.model)

        logger.info("模型加载完成，耗时 %.1fs，EOS=%s", time.time() - t0, self._eos_token_ids)

    # ------------------------------------------------------------------
    # 请求构建
    # ------------------------------------------------------------------
    def next_seq_id(self) -> int:
        """生成自增序列 id。"""
        with self._id_lock:
            self._seq_counter += 1
            return self._seq_counter

    def build_sequence(
        self,
        messages: list,
        max_new_tokens: Optional[int] = None,
    ) -> Sequence:
        """把 chat messages 转成 Sequence。

        Args:
            messages: OpenAI 风格的 messages 列表，content 里可含 image/text
            max_new_tokens: 覆盖默认生成上限

        Returns:
            构建好的 Sequence（状态为 WAITING）
        """
        if self.model is None:
            raise RuntimeError("请先调用 load_model()")

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][0].tolist()
        seq = Sequence(
            seq_id=self.next_seq_id(),
            input_ids=input_ids,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
        )

        # 多模态字段
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            seq.pixel_values = pixel_values.to(self.device)
        grid = inputs.get("image_grid_thw")
        if grid is not None:
            seq.image_grid_thw = grid.to(self.device)
        mm_types = inputs.get("mm_token_type_ids")
        if mm_types is not None:
            seq.mm_token_type_ids = mm_types.to(self.device)

        return seq

    def add_request(self, seq: Sequence) -> bool:
        """提交请求到队列。"""
        return self.scheduler.add_request(seq)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run_until_done(self, max_steps: Optional[int] = None) -> None:
        """驱动调度循环，直到所有请求完成。"""
        step = 0
        while self.scheduler.has_work():
            self.step()
            step += 1
            if max_steps is not None and step >= max_steps:
                logger.warning("达到 max_steps=%d，提前退出", max_steps)
                break

    def step(self) -> None:
        """执行一个调度步。

        这是连续批处理的核心：每次调度都可能引入新序列、移除已完成序列。
        """
        # 1) 调度：回收 + 补位
        active = self.scheduler.schedule()
        if not active:
            return

        self.stats["steps"] += 1

        # 2) 区分 prefill / decode
        prefill_seqs = [s for s in active if not s.output_ids]
        decode_seqs = [s for s in active if s.output_ids]

        if prefill_seqs:
            self._run_prefill(prefill_seqs)
        if decode_seqs:
            self._run_decode(decode_seqs)

    # ------------------------------------------------------------------
    # Prefill：处理整段 prompt
    # ------------------------------------------------------------------
    def _run_prefill(self, seqs: List[Sequence]) -> None:
        """对一个批次的新序列做 prefill。

        策略：先按 prompt 长度分桶，同一桶内的序列长度一致，
        可以直接拼成一个 batch 做一次 forward（无需 padding 对齐），
        这样 prefill 也享受批处理收益。

        长度不一致的序列各自成桶（退化为逐条）。
        """
        self.stats["prefill_batches"] += 1
        ctx = torch.inference_mode if self.config.use_inference_mode else torch.no_grad

        buckets: Dict[int, List[Sequence]] = {}
        for seq in seqs:
            if seq.is_finished:
                continue
            buckets.setdefault(seq.prompt_len, []).append(seq)

        with ctx():
            for length, group in buckets.items():
                # 同一长度且视觉输入形状一致时，才真正合并成 batch
                if len(group) > 1 and self._same_vision(group):
                    self._prefill_batch(group)
                else:
                    for seq in group:
                        self._prefill_single(seq)

    @staticmethod
    def _same_vision(group: List[Sequence]) -> bool:
        """判断一组序列的视觉输入是否能拼成同一个 batch。

        要求每条的 pixel_values 第一维（patch 数）之和与 grid 兼容，
        这里采用保守判断：只有全部无图或图片形状完全一致的才合并。
        """
        first = group[0]
        if first.pixel_values is None:
            return all(s.pixel_values is None for s in group)
        shapes = {tuple(s.pixel_values.shape) for s in group if s.pixel_values is not None}
        grids = {tuple(s.image_grid_thw.flatten().tolist()) for s in group
                 if s.image_grid_thw is not None}
        return len(shapes) == 1 and len(grids) <= 1

    def _prefill_batch(self, group: List[Sequence]) -> None:
        """同长度的多条序列一次 forward 完成 prefill。

        先把各序列的 prompt 拼成 batch 前向，再把 batch cache 拆回各自槽位，
        这样后续 decode 阶段每步只需处理 1 个 token，且能按长度分桶批量执行。
        """
        batch = len(group)
        device = self.device
        L = group[0].prompt_len

        input_ids = torch.tensor([s.input_ids for s in group], device=device)
        attn = torch.ones((batch, L), dtype=torch.long, device=device)

        merged = DynamicCache()
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "past_key_values": merged,
            "use_cache": True,
            "logits_to_keep": 1,
        }

        # 视觉输入在 batch 维拼接
        pv = group[0].pixel_values
        if pv is not None:
            kwargs["pixel_values"] = torch.cat([s.pixel_values for s in group], dim=0)
            kwargs["image_grid_thw"] = torch.cat(
                [s.image_grid_thw for s in group], dim=0
            )
            kwargs["mm_token_type_ids"] = torch.stack(
                [s.mm_token_type_ids.flatten() for s in group], dim=0
            )

        try:
            out = self.model(**kwargs)
        except Exception as e:  # noqa: BLE001
            logger.warning("批量 prefill 失败，退化为逐条: %s", e)
            for seq in group:
                self._prefill_single(seq)
            return

        self.stats["forward_calls"] += 1

        # 把 batch cache 拆回各槽位（等长，无需 padding）
        for i, seq in enumerate(group):
            slot_cache = self.kv_manager.ensure_slot(seq.slot)
            for layer_idx in range(len(merged.layers)):
                layer = merged.layers[layer_idx]
                if layer is None or layer.keys is None:
                    continue
                k_i = layer.keys[i : i + 1].contiguous()
                v_i = layer.values[i : i + 1].contiguous()
                slot_cache.update(k_i, v_i, layer_idx)

        logits = out.logits[:, -1, :]
        for i, seq in enumerate(group):
            if seq.is_finished:
                continue
            token = self._sample_one(seq, logits[i : i + 1])
            self._handle_new_token(seq, token)

    def _prefill_single(self, seq: Sequence) -> None:
        """单条序列的 prefill（作为兜底与长尾处理）。"""
        inputs = {
            "input_ids": torch.tensor([seq.input_ids], device=self.device),
            "attention_mask": torch.ones(
                (1, len(seq.input_ids)), dtype=torch.long, device=self.device
            ),
        }
        if seq.pixel_values is not None:
            inputs["pixel_values"] = seq.pixel_values
        if seq.image_grid_thw is not None:
            inputs["image_grid_thw"] = seq.image_grid_thw
        if seq.mm_token_type_ids is not None:
            inputs["mm_token_type_ids"] = seq.mm_token_type_ids

        cache = self.kv_manager.ensure_slot(seq.slot)
        try:
            out = self.model(
                **inputs, past_key_values=cache, use_cache=True, logits_to_keep=1
            )
        except Exception as e:  # noqa: BLE001
            logger.error("prefill 失败 seq=%s: %s", seq.seq_id, e)
            seq.mark_aborted()
            self.kv_manager.reset_slot(seq.slot)
            return

        self.stats["forward_calls"] += 1
        token = self._sample_one(seq, out.logits[:, -1, :])
        self._handle_new_token(seq, token)

    # ------------------------------------------------------------------
    # Decode：整批一起生成下一个 token
    # ------------------------------------------------------------------
    def _run_decode(self, seqs: List[Sequence]) -> None:
        """对一个批次的序列做 decode —— 一次 forward 处理整个 batch。

        这是吞吐量的关键：整个 batch 只需要一次 forward，GPU 吃到
        batch_size 倍的并行度（这正是 vLLM 连续批处理的核心收益）。

        策略：按 total_len 分桶，只有长度相同的序列才合并成一次 forward。
        - 长度相同 → 无需 padding，位置编码天然对齐，结果与逐条严格一致；
        - 长度不同 → 各自成桶（退化为逐条）。

        【重要】为什么不合并不同长度的序列？
        GLM-OCR 使用 3D 位置编码 (mrope)，其 KV Cache 与 attention_mask 的
        padding 处理是自定义的。实测即使显式传入 position_ids 与
        mm_token_type_ids，对"cache 长度不一致 + padding"的 batch 仍会算出
        错误 logits（输出乱码）。因此长度对齐是这个模型批处理的硬约束，
        而非性能上的妥协。这与 vLLM 用 PagedAttention 任意拼装序列的做法
        有本质差异——后者要求模型侧配合。

        实践上这仍能拿到大部分收益：同一批提交、prompt 长度相同的请求
        （例如批量 OCR 同一模板的文档）会全程同长，持续享有批量加速。
        """
        active = [s for s in seqs if not s.is_finished]
        if not active:
            return

        ctx = torch.inference_mode if self.config.use_inference_mode else torch.no_grad

        buckets: Dict[int, List[Sequence]] = {}
        for seq in active:
            buckets.setdefault(seq.total_len, []).append(seq)

        with ctx():
            for length, group in buckets.items():
                if len(group) > 1:
                    try:
                        self._decode_batch(group)
                        continue
                    except Exception as e:  # noqa: BLE001
                        logger.warning("批量 decode 失败，退化为逐条: %s", e)
                for seq in group:
                    self._decode_one_by_one([seq])

    def _decode_batch(self, group: List[Sequence]) -> None:
        """等长序列的批量 decode：一次 forward 处理整组。"""
        batch = len(group)
        device = self.device
        L = group[0].total_len  # 组内长度一致

        input_ids = torch.tensor(
            [[s.output_ids[-1] if s.output_ids else s.input_ids[-1]] for s in group],
            device=device,
        )
        attn = torch.ones((batch, L), dtype=torch.long, device=device)
        pos_ids = torch.tensor([[L - 1]] * batch, dtype=torch.long, device=device)

        # KV Cache：组内每条序列各自持有，按层拼成 batch 维度
        merged = self.kv_manager.merge_caches(group)

        fwd_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=pos_ids,
            past_key_values=merged,
            use_cache=True,
            logits_to_keep=1,
        )

        # mm_token_type_ids：mrope 在 batch 场景必须知道模态分布
        if any(s.mm_token_type_ids is not None for s in group):
            mm = torch.zeros((batch, L), dtype=torch.long, device=device)
            for i, s in enumerate(group):
                if s.mm_token_type_ids is None:
                    continue
                hist = s.mm_token_type_ids.flatten()[:L]
                mm[i, : hist.shape[0]] = hist
            fwd_kwargs["mm_token_type_ids"] = mm

        try:
            out = self.model(**fwd_kwargs)
        except Exception:
            self.kv_manager.release_merged()
            raise

        self.stats["forward_calls"] += 1
        self.kv_manager.split_back(group, merged)

        logits = out.logits[:, -1, :]
        for i, seq in enumerate(group):
            if seq.is_finished:
                continue
            token = self._sample_one(seq, logits[i : i + 1])
            self._handle_new_token(seq, token)

    def _decode_one_by_one(self, seqs: List[Sequence]) -> None:
        """逐条 decode（批量失败时的兜底路径）。"""
        for seq in seqs:
            if seq.is_finished:
                continue
            last_token = seq.output_ids[-1] if seq.output_ids else seq.input_ids[-1]
            total_len = seq.total_len

            inputs = {
                "input_ids": torch.tensor([[last_token]], device=self.device),
                "attention_mask": torch.ones(
                    (1, total_len), dtype=torch.long, device=self.device
                ),
                "position_ids": torch.tensor(
                    [[total_len - 1]], dtype=torch.long, device=self.device
                ),
            }
            if seq.mm_token_type_ids is not None:
                hist = seq.mm_token_type_ids.flatten()
                mm = torch.zeros((1, total_len), dtype=torch.long, device=self.device)
                take = min(hist.shape[0], total_len)
                mm[0, :take] = hist[:take]
                inputs["mm_token_type_ids"] = mm
            cache = self.kv_manager.get_slot(seq.slot)
            try:
                out = self.model(
                    **inputs, past_key_values=cache, use_cache=True, logits_to_keep=1
                )
            except Exception as e:  # noqa: BLE001
                logger.error("decode 失败 seq=%s: %s", seq.seq_id, e)
                seq.mark_aborted()
                self.kv_manager.reset_slot(seq.slot)
                continue

            self.stats["forward_calls"] += 1
            self._handle_new_token(seq, self._sample_one(seq, out.logits[:, -1, :]))

    # ------------------------------------------------------------------
    # 采样与状态更新
    # ------------------------------------------------------------------
    def _sample_one(self, seq: Sequence, logits: torch.Tensor) -> int:
        """对单条序列采样一个 token。"""
        params = None
        if seq.temperature is not None or seq.top_p is not None or seq.top_k is not None:
            params = [{
                "temperature": seq.temperature if seq.temperature is not None else self.config.temperature,
                "top_p": seq.top_p if seq.top_p is not None else self.config.top_p,
                "top_k": seq.top_k if seq.top_k is not None else self.config.top_k,
            }]
        token = self.sampler(logits, params)
        return int(token.item())

    def _handle_new_token(self, seq: Sequence, token: int) -> None:
        """处理新生成的 token：判停、记录、释放资源。"""
        self.stats["tokens_generated"] += 1

        # EOS 判定
        if token in self._eos_token_ids:
            seq.mark_finished()
            self.kv_manager.reset_slot(seq.slot)
            return

        seq.append_token(token)

        # 长度上限
        if seq.is_max_tokens_reached():
            seq.mark_finished()
            self.kv_manager.reset_slot(seq.slot)

    # ------------------------------------------------------------------
    # 结果获取
    # ------------------------------------------------------------------
    def decode(self, seq: Sequence) -> str:
        """把生成的 token 解码为文本。"""
        return self.processor.decode(seq.output_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # 资源
    # ------------------------------------------------------------------
    def memory_usage(self) -> Dict[str, float]:
        """返回显存占用统计（MB）。"""
        result = {}
        if torch.cuda.is_available():
            result["allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            result["reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        if self.kv_manager is not None:
            result["kv_cache_mb"] = self.kv_manager.memory_usage_mb()
        return result

    def shutdown(self) -> None:
        """释放模型与缓存。"""
        if self.kv_manager is not None:
            self.kv_manager.clear()
            self.kv_manager.prefix_cache.clear()
        self.model = None
        self.processor = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
