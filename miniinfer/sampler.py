"""采样器。

GLM-OCR 默认是 greedy（generation_config.json 里 do_sample=false），
OCR 任务也确实该用 greedy：结果确定、可复现。

这里同时支持 temperature / top-k / top-p，方便以后换别的模型。
所有操作都在 [batch, vocab] 的 logits 上向量化完成，一次处理整个 batch。
"""

from __future__ import annotations

from typing import List, Optional

import torch


class Sampler:
    """批量采样器。"""

    def __init__(
        self,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def __call__(
        self,
        logits: torch.Tensor,
        per_seq_params: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """从 logits 采样出 token。

        Args:
            logits: [batch, vocab] 形状的未归一化分数
            per_seq_params: 每条序列的独立采样参数（覆盖默认值）

        Returns:
            [batch] 形状的 token id
        """
        if not self.do_sample:
            return torch.argmax(logits, dim=-1)

        # 逐序列应用不同参数（batch 内参数可能不同）
        if per_seq_params is not None and len(per_seq_params) == logits.shape[0]:
            probs = self._sample_with_params(logits, per_seq_params)
        else:
            probs = self._apply_sampling(logits, self.temperature, self.top_k, self.top_p)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _sample_with_params(
        self, logits: torch.Tensor, params: List[dict]
    ) -> torch.Tensor:
        """逐条应用各自的采样参数。"""
        outputs = []
        for i, p in enumerate(params):
            row = logits[i : i + 1]
            probs = self._apply_sampling(
                row,
                p.get("temperature", self.temperature),
                p.get("top_k", self.top_k),
                p.get("top_p", self.top_p),
            )
            outputs.append(probs)
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _apply_sampling(
        logits: torch.Tensor, temperature: float, top_k: int, top_p: float
    ) -> torch.Tensor:
        """对一组 logits 应用 temperature + top-k + top-p，返回概率。"""
        # 数值稳定：先转 float32 再算
        logits = logits.float()

        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        # top-k 过滤
        if top_k and top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = torch.where(
                logits < kth, torch.full_like(logits, float("-inf")), logits
            )

        probs = torch.softmax(logits, dim=-1)

        # top-p (nucleus) 过滤
        if top_p and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # 累积概率超过 top_p 的位置全部屏蔽（保留第一个超过的）
            mask = cumulative - sorted_probs > top_p
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            # 归一化
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)

        return probs
