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
    """前缀缓存：按 token 序列哈希缓存 KV Cache。

    对应 vLLM 的 Prefix Caching / SGLang 的 RadixAttention：
    多条请求若共享相同的前缀，其 KV Cache 可以复用，跳过重复的 prefill。

    适用场景：批量处理同一模板的文档时，prompt 前缀（如系统提示、
    固定指令）完全相同，命中率很高。

    实现说明：
        缓存的是"完整序列的 KV"（而非任意长度前缀）。这样实现简单、
        语义清晰，代价是只有序列完全一致才命中。对 OCR/批量处理场景
        已经足够——它们的 prompt 模板高度统一。

        缓存粒度按"层"存储：DynamicCache 是多层结构，逐层保存。

    线程安全性：单进程单线程假设，引擎的 step 循环里串行访问。
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.enabled = config.enable_prefix_cache
        # 有序字典实现 LRU：key -> list[(k, v)]（按层）
        self._store: "OrderedDict[str, List[Tuple[torch.Tensor, torch.Tensor]]]" = \
            OrderedDict()
        # 统计
        self.hits = 0
        self.misses = 0

    def get(self, token_ids: List[int]) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """查询前缀缓存。

        Args:
            token_ids: 序列的 token id 列表（作为缓存键）

        Returns:
            命中时返回按层组织的 [(k, v), ...]，未命中返回 None。
            k/v 形状为 [1, heads, seq, dim]（单序列，无 batch 维扩展）。
        """
        if not self.enabled:
            return None
        key = hash_tokens(token_ids)
        if key not in self._store:
            self.misses += 1
            return None
        # 命中则移到末尾（LRU）
        self._store.move_to_end(key)
        self.hits += 1
        return self._store[key]

    def put(self, token_ids: List[int], cache) -> None:
        """写入前缀缓存。

        Args:
            token_ids: 序列的 token id 列表（作为缓存键）
            cache: DynamicCache 对象，取其各层的 keys/values
        """
        if not self.enabled or cache is None:
            return

        layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in getattr(cache, "layers", []):
            if layer is None or layer.keys is None:
                continue
            # 分离计算图并复制，避免拖住显存
            layers.append((
                layer.keys.detach().clone(),
                layer.values.detach().clone(),
            ))

        if not layers:
            return

        key = hash_tokens(token_ids)
        self._store[key] = layers
        self._store.move_to_end(key)

        # LRU 淘汰
        while len(self._store) > self.config.prefix_cache_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """清空前缀缓存。"""
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def memory_usage_mb(self) -> float:
        """统计缓存占用的显存（MB）。"""
        total = 0
        for layers in self._store.values():
            for k, v in layers:
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total / 1024 / 1024

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

    # ------------------------------------------------------------------
    # 前缀缓存对接
    # ------------------------------------------------------------------
    def try_restore_prefix(
        self, slot: int, token_ids: List[int]
    ) -> bool:
        """尝试从前缀缓存恢复某个槽位的 KV Cache。

        命中时直接把缓存的 KV 写入该槽位，调用方即可跳过 prefill
        （只需处理最后 1 个 token 来产出第一个输出）。

        Args:
            slot: 目标槽位
            token_ids: 该序列的 token id 列表（作缓存键）

        Returns:
            True 表示命中并已恢复；False 表示未命中，需正常 prefill。
        """
        cached = self.prefix_cache.get(token_ids)
        if cached is None:
            return False

        # 按层恢复。缓存里存的是单序列 KV（batch 维为 1），
        # 与槽位一一对应，直接写入即可。
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(cached):
            cache.update(k, v, layer_idx)
        self._caches[slot] = cache
        return True

    def save_prefix(self, slot: int, token_ids: List[int]) -> None:
        """把某个槽位当前的 KV Cache 存入前缀缓存。

        通常在 prefill 完成后调用——此时该槽位保存的正是完整 prompt
        的 KV，后续请求若 prompt 相同即可直接复用。
        """
        cache = self.get_slot(slot)
        if cache is None:
            return
        self.prefix_cache.put(token_ids, cache)

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
