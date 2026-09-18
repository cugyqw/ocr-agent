"""miniinfer 配置。

设计原则：所有可调参数集中在这里，便于后续按需扩展。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineConfig:
    """推理引擎配置。"""

    # ---- 模型 ----
    model_path: str = "/home/yqw/桌面/ocr_agent/GLM-OCR"
    device: str = "cuda"
    dtype: str = "bfloat16"

    # ---- 批处理 ----
    # 单步最多同时处理多少条序列。8GB 显存下先给保守值。
    max_batch_size: int = 4
    # 单条序列的最大总长度（prompt + 生成），决定 KV Cache 预分配大小
    max_model_len: int = 4096
    # 默认最大生成 token 数
    max_new_tokens: int = 512

    # ---- KV Cache ----
    # 是否启用前缀缓存（多条请求共享 prompt 前缀时复用 KV）
    enable_prefix_cache: bool = True
    # 前缀缓存最多保留多少个前缀条目
    prefix_cache_size: int = 32

    # ---- 调度 ----
    # 队列最大等待请求数，超过则拒绝
    max_queue_size: int = 128
    # 调度策略: "fcfs"（先来先服务）
    schedule_policy: str = "fcfs"

    # ---- 采样 ----
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    # ---- 视觉输入 ----
    # 图片最长边像素上限。原模型默认 9633792，太大，限制后显著省显存。
    # 设为 None 表示不限制。
    max_image_pixels: Optional[int] = 1280 * 1280

    # ---- 性能 ----
    # 是否使用 torch.inference_mode（关闭梯度，省显存）
    use_inference_mode: bool = True
    # 是否启用 torch.compile（首次编译慢，后续更快）
    # 注意：Blackwell(SM12.0) + Triton 需要 python3-dev，环境不满足时保持 False
    use_torch_compile: bool = False
