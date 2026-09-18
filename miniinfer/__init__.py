"""miniinfer —— 轻量 LLM 推理引擎。

在 transformers 之上实现 vLLM/SGLang 的核心优势：
- 连续批处理（Continuous Batching）
- KV Cache 复用 / 前缀缓存
- 模型常驻显存
- CUDA 层复用 PyTorch SDPA，不自己写 kernel

设计目标是"够用、可读、可替换"，而非追求极致性能。
"""

from .config import EngineConfig
from .engine import MiniInferEngine
from .sequence import Sequence, SeqStatus
from .scheduler import Scheduler
from .sampler import Sampler

__all__ = [
    "EngineConfig",
    "MiniInferEngine",
    "Sequence",
    "SeqStatus",
    "Scheduler",
    "Sampler",
]

__version__ = "0.1.0"
