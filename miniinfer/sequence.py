"""序列状态机。

连续批处理的核心数据结构：每条请求在引擎内部表示为一个 Sequence，
它的生命周期是 WAITING -> RUNNING -> FINISHED。

对比 vLLM：vLLM 的 Sequence 还要管 block table（PagedAttention 的页表），
这里简化为"整条序列占一块连续 KV Cache 区域"，用显存浪费换实现简单。
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch


class SeqStatus(enum.Enum):
    """序列状态。"""

    WAITING = "waiting"      # 在队列里等待被调度
    RUNNING = "running"      # 正在参与 batch 生成
    FINISHED = "finished"    # 生成结束（自然停止或达到上限）
    ABORTED = "aborted"      # 被主动取消


@dataclass
class Sequence:
    """一条推理请求的完整状态。

    Attributes:
        seq_id: 唯一标识
        input_ids: prompt 的 token id 列表（不含 padding）
        pixel_values / image_grid_thw: 视觉输入（多模态）
        max_new_tokens: 本条请求最多生成多少 token
        output_ids: 已生成的 token id
        status: 当前状态
    """

    seq_id: int
    input_ids: List[int]
    max_new_tokens: int = 512

    # 多模态输入（None 表示纯文本）
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    # mm_token_type_ids: 标记每个 token 是文本(0) 还是图片(1)。
    # GLM-OCR 的 3D 位置编码(mrope)在 batch 推理时依赖它来区分模态，
    # 必须在 prefill 和 decode 阶段都传入，否则位置编码错位、输出乱码。
    mm_token_type_ids: Optional[torch.Tensor] = None

    # 采样参数（None 表示用引擎默认）
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    # ---- 运行时状态 ----
    status: SeqStatus = SeqStatus.WAITING
    output_ids: List[int] = field(default_factory=list)

    # 在 batch 中的槽位索引（-1 表示未分配）
    slot: int = -1

    #  KV Cache 起始偏移（前缀缓存命中时 > 0）
    cache_offset: int = 0

    # 计时
    enqueue_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def prompt_len(self) -> int:
        """prompt 长度。"""
        return len(self.input_ids)

    @property
    def total_len(self) -> int:
        """当前总长度（prompt + 已生成）。"""
        return len(self.input_ids) + len(self.output_ids)

    @property
    def is_finished(self) -> bool:
        """是否已结束。"""
        return self.status in (SeqStatus.FINISHED, SeqStatus.ABORTED)

    def append_token(self, token_id: int) -> None:
        """追加一个生成的 token。"""
        self.output_ids.append(token_id)

    def mark_running(self) -> None:
        """标记为运行中。"""
        if self.status == SeqStatus.WAITING:
            self.status = SeqStatus.RUNNING
            self.start_time = time.time()

    def mark_finished(self) -> None:
        """标记为完成。"""
        self.status = SeqStatus.FINISHED
        self.finish_time = time.time()

    def mark_aborted(self) -> None:
        """标记为取消。"""
        self.status = SeqStatus.ABORTED
        self.finish_time = time.time()

    def is_max_tokens_reached(self) -> bool:
        """是否已达到生成长度上限。"""
        return len(self.output_ids) >= self.max_new_tokens

    def latency(self) -> Optional[float]:
        """端到端延迟（秒）。"""
        if self.finish_time is None or self.start_time is None:
            return None
        return self.finish_time - self.start_time

    def ttft(self) -> Optional[float]:
        """首 token 延迟（秒）。"""
        if self.start_time is None:
            return None
        return self.start_time - self.enqueue_time
