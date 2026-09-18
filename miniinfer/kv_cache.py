"""KV Cache 管理 + 前缀缓存。

对比 vLLM 的 PagedAttention：
- vLLM：把 KV Cache 切成分页，用页表拼装，几乎不浪费显存，但需要 CUDA kernel。
- 本实现：为每个 batch 槽位预分配一整块 StaticCache，用显存浪费换实现简单。
  8GB 卡上 GLM-OCR(0.9B) 够用；后续需要更大 batch 时可替换本模块。

前缀缓存（对应 SGLang 的 RadixAttention）：
- vLLM 用哈希表匹配 block；SGLang 用基数树匹配任意长度前缀。
- 本实现用哈希表匹配"完整 prompt 前缀"，命中则直接复用已算好的 KV，
  跳过 prefill。批量 OCR 场景下所有请求共享长 prompt 模板，收益明显。
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache, StaticCache

from .config import EngineConfig

logger = logging.getLogger(__name__)


def hash_tokens(token_ids: List[int]) -> str:
    """对 token 序列做哈希，用于前缀缓存索引。"""
    raw = ",".join(map(str, token_ids)).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


class PrefixCache:
    """前缀缓存：按 prompt 前缀哈希缓存 KV Cache。

    线程安全性：单进程单线程假设，引擎的 step 循环里串行访问。
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.enabled = config.enable_prefix_cache
        # 有序字典实现 LRU
        self._store: "OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

    def get(self, token_ids: List[int]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """查询前缀缓存。

        Returns:
            (key_cache, value_cache) 命中时返回，未命中返回 None。
        """
        if not self.enabled:
            return None
        key = hash_tokens(token_ids)
        if key not in self._store:
            return None
        # 命中则移到末尾（LRU）
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, token_ids: List[int], k: torch.Tensor, v: torch.Tensor) -> None:
        """写入前缀缓存。k/v 应为 [num_layers, batch, heads, seq, dim] 之类，
        这里按层拆分后逐层存储更省内存，简化起见先存整块。"""
        if not self.enabled:
            return
        key = hash_tokens(token_ids)
        # 分离计算图，避免拖住显存
        self._store[key] = (k.detach().clone(), v.detach().clone())
        self._store.move_to_end(key)
        # LRU 淘汰
        while len(self._store) > self.config.prefix_cache_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """清空前缀缓存。"""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class BatchKVCache:
    """管理一个 batch 的 KV Cache。

    为 max_batch_size 个槽位预分配 StaticCache，每步以"所有序列当前总长度
    的最大值"作为 cache 位置推进。不同长度的序列靠 attention_mask 屏蔽。

    说明：这里采用"整块预分配"策略，显存占用 = max_batch_size * max_model_len，
    属于用空间换简单。若显存吃紧，可改为按需分配/分页。
    """

    def __init__(self, config: EngineConfig, model):
        self.config = config
        self.model = model
        self.prefix_cache = PrefixCache(config)

        text_cfg = getattr(model.config, "text_config", model.config)
        self.num_layers = getattr(text_cfg, "num_hidden_layers", 16)
        self.num_kv_heads = getattr(text_cfg, "num_key_value_heads", 8)
        self.head_dim = getattr(text_cfg, "head_dim", 128)

        # 延迟到第一次使用时分配（需要知道实际 dtype/device）
        self._caches: List[Optional[DynamicCache]] = [None] * config.max_batch_size
        # 批量 forward 时记录合并后的最大长度，供 split_back 使用
        self._merged_len: Optional[int] = None
        # 每条序列 merge 前的真实 cache 长度
        self._merged_lens: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # 槽位生命周期
    # ------------------------------------------------------------------
    def reset_slot(self, slot: int) -> None:
        """清空某个槽位的缓存（序列结束/新序列占用时调用）。"""
        if 0 <= slot < len(self._caches):
            self._caches[slot] = None

    def get_slot(self, slot: int) -> Optional[DynamicCache]:
        """取某个槽位的 cache 对象。"""
        return self._caches[slot] if 0 <= slot < len(self._caches) else None

    def ensure_slot(self, slot: int) -> DynamicCache:
        """确保槽位已分配 cache。"""
        if self._caches[slot] is None:
            self._caches[slot] = DynamicCache()
        return self._caches[slot]

    def clear(self) -> None:
        """清空所有槽位。"""
        for i in range(len(self._caches)):
            self._caches[i] = None

    # ------------------------------------------------------------------
    # 批量 forward 支持：把多个槽位的 cache 合并成一个大 batch
    # ------------------------------------------------------------------
    def merge_caches(self, seqs: List["Sequence"]) -> DynamicCache:  # noqa: F821
        """把多条序列各自的 KV Cache 拼成一个 batch 维度的 cache。

        前提：组内序列的 cache 长度必须一致（引擎按 total_len 分桶保证）。
        长度一致时无需 padding，位置编码天然对齐，结果与逐条推理等价。

        Returns:
            合并后的 DynamicCache，batch 维等于 len(seqs)。
        """
        merged = DynamicCache()
        if not seqs:
            return merged

        batch = len(seqs)

        template = self.get_slot(seqs[0].slot)
        if template is None or len(template.layers) == 0:
            raise RuntimeError(f"序列 {seqs[0].seq_id} 的 KV Cache 为空，无法合并")

        num_layers = len(template.layers)
        self._merged_lens = []

        for layer_idx in range(num_layers):
            layer = template.layers[layer_idx]
            if layer is None or layer.keys is None:
                continue
            tpl_k = layer.keys
            heads, dim = tpl_k.shape[1], tpl_k.shape[3]
            dtype, device = tpl_k.dtype, tpl_k.device
            L = tpl_k.shape[2]

            k_buf = torch.zeros((batch, heads, L, dim), dtype=dtype, device=device)
            v_buf = torch.zeros_like(k_buf)

            for i, seq in enumerate(seqs):
                cache = self.get_slot(seq.slot)
                if cache is None or layer_idx >= len(cache.layers):
                    continue
                lyr = cache.layers[layer_idx]
                if lyr is None or lyr.keys is None:
                    continue
                ck, cv = lyr.keys, lyr.values
                if ck.shape[2] != L:
                    raise RuntimeError(
                        f"序列 {seq.seq_id} cache 长度 {ck.shape[2]} 与组内长度 {L} 不一致"
                    )
                k_buf[i] = ck.to(device)
                v_buf[i] = cv.to(device)

            merged.update(k_buf, v_buf, layer_idx)

        for seq in seqs:
            c = self.get_slot(seq.slot)
            if (c is None or len(c.layers) == 0 or c.layers[0] is None
                    or c.layers[0].keys is None):
                self._merged_lens.append(0)
            else:
                self._merged_lens.append(c.layers[0].keys.shape[2])
        return merged

    def split_back(self, seqs: List["Sequence"], merged: DynamicCache) -> None:  # noqa: F821
        """把合并 cache 的结果拆回各槽位。

        forward 会在 merged cache 右侧追加这 1 个新 token 的 KV，
        因此有效长度 = merge 前长度 + 1。
        """
        if not seqs or len(merged.layers) == 0:
            return
        merged_lens = getattr(self, "_merged_lens", None)

        for i, seq in enumerate(seqs):
            cur = self.get_slot(seq.slot)
            if cur is None:
                continue
            prev_len = merged_lens[i] if merged_lens and i < len(merged_lens) else 0
            new_len = prev_len + 1
            for layer_idx in range(len(merged.layers)):
                layer = merged.layers[layer_idx]
                if layer is None or layer.keys is None:
                    continue
                k, v = layer.keys, layer.values
                # 等长组、左侧无 padding，直接取前 new_len 个位置
                k_i = k[i : i + 1, :, :new_len, :].contiguous()
                v_i = v[i : i + 1, :, :new_len, :].contiguous()
                if layer_idx < len(cur.layers) and cur.layers[layer_idx] is not None:
                    cur.layers[layer_idx].keys = k_i
                    cur.layers[layer_idx].values = v_i
                else:
                    cur.update(k_i, v_i, layer_idx)

        self._merged_len = None
        self._merged_lens = None

    def release_merged(self) -> None:
        """释放临时合并对象引用。"""
        self._merged_len = None
        self._merged_lens = None

    # ------------------------------------------------------------------
    # 显存统计
    # ------------------------------------------------------------------
    def memory_usage_mb(self) -> float:
        """粗略统计当前 KV Cache 占用（MB）。"""
        total = 0
        for cache in self._caches:
            if cache is None:
                continue
            try:
                for layer in cache.layers:
                    if layer is None:
                        continue
                    for t in (layer.keys, layer.values):
                        if isinstance(t, torch.Tensor):
                            total += t.numel() * t.element_size()
            except Exception:
                pass
        return total / 1024 / 1024
