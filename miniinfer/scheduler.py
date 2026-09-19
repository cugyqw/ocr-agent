"""调度器 —— 连续批处理（Continuous Batching）的核心实现。

对比 vLLM/SGLang：
- 静态批处理：一批请求必须同进同出，短请求被长请求拖住，GPU 空转。
- 连续批处理：每生成一个 step 就重新调度一次，完成的立刻出队、
  等待的立刻补位，GPU 每个时刻都是满的。

本调度器每个 step 做两件事：
1. 把已完成的序列移出 running 集合
2. 从 waiting 队列取新序列填补空位
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List, Optional

from .config import EngineConfig
from .sequence import Sequence, SeqStatus

logger = logging.getLogger(__name__)


class Scheduler:
    """连续批处理调度器。

    维护两个集合：
    - waiting: 等待调度的请求队列
    - running: 当前正在生成、占用 batch 槽位的请求
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Optional[Sequence]] = [None] * config.max_batch_size

        # 统计：累计被调度过的序列数（只增不减），
        # 由 engine 汇总进 stats 对外暴露
        self.total_scheduled = 0

    # ------------------------------------------------------------------
    # 入队
    # ------------------------------------------------------------------
    def add_request(self, seq: Sequence) -> bool:
        """把请求加入等待队列。

        Returns:
            True 表示入队成功，False 表示队列已满。
        """
        if len(self.waiting) >= self.config.max_queue_size:
            logger.warning("队列已满(%d)，拒绝请求 seq_id=%s",
                           self.config.max_queue_size, seq.seq_id)
            return False
        self.waiting.append(seq)
        return True

    # ------------------------------------------------------------------
    # 调度
    # ------------------------------------------------------------------
    def schedule(self) -> List[Sequence]:
        """执行一次调度：回收完成的槽位，再用等待队列填补。

        Returns:
            本次需要实际参与 forward 的序列列表（即 running 中的非空项）。
            注意：一个 step 里，有些序列可能只需要处理 1 个 token（decode），
            有些需要处理整个 prompt（prefill），这个区分由引擎层处理。
        """
        # 1) 回收已完成的槽位
        self._free_finished_slots()

        # 2) 用等待队列填补空位
        self._fill_empty_slots()

        # 3) 返回当前在跑的序列
        return [s for s in self.running if s is not None]

    def _free_finished_slots(self) -> None:
        """把已完成/已取消的序列移出运行槽位。"""
        for i, seq in enumerate(self.running):
            if seq is not None and seq.is_finished:
                self.running[i] = None

    def _fill_empty_slots(self) -> None:
        """从等待队列取请求填补空槽位。"""
        for i, seq in enumerate(self.running):
            if seq is not None:
                continue
            if not self.waiting:
                break

            candidate = self.waiting.popleft()

            # 长度校验：超过最大上下文则直接拒绝
            if candidate.total_len > self.config.max_model_len:
                logger.warning(
                    "序列 %s 长度 %d 超过上限 %d，拒绝",
                    candidate.seq_id, candidate.total_len, self.config.max_model_len,
                )
                candidate.mark_aborted()
                continue

            candidate.slot = i
            candidate.mark_running()
            self.running[i] = candidate
            self.total_scheduled += 1

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def has_work(self) -> bool:
        """是否还有未完成的工作。"""
        return bool(self.waiting) or any(s is not None for s in self.running)

    # ------------------------------------------------------------------
    # 取消
    # ------------------------------------------------------------------
    def abort(self, seq_id: int) -> bool:
        """取消指定请求（无论它在等待还是在跑）。"""
        # 在 waiting 里找
        for seq in list(self.waiting):
            if seq.seq_id == seq_id:
                self.waiting.remove(seq)
                seq.mark_aborted()
                return True
        # 在 running 里找
        for i, seq in enumerate(self.running):
            if seq is not None and seq.seq_id == seq_id:
                self.running[i] = None
                seq.mark_aborted()
                return True
        return False
