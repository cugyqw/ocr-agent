"""阶段一：OCR 识别。

职责：把图片转成文本。数学题的关键是公式要转成 LaTeX，
GLM-OCR 的 "Formula Recognition:" prompt 正好负责这件事。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from miniinfer import EngineConfig, MiniInferEngine

logger = logging.getLogger(__name__)

# GLM-OCR 支持的解析 prompt（来自官方 README）
PROMPT_TEXT = "Text Recognition:"
PROMPT_FORMULA = "Formula Recognition:"
PROMPT_TABLE = "Table Recognition:"


class OCRStage:
    """OCR 阶段：图片 → 文本。"""

    def __init__(
        self,
        model_path: str = "/home/yqw/桌面/ocr_agent/GLM-OCR",
        max_new_tokens: int = 1024,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.engine: MiniInferEngine | None = None

    def load(self) -> None:
        """加载并常驻 GLM-OCR。"""
        self.engine = MiniInferEngine(EngineConfig(
            model_path=self.model_path,
            max_batch_size=2,       # OCR 阶段并发需求低，省显存
            max_new_tokens=self.max_new_tokens,
            max_model_len=4096,
        ))
        self.engine.load_model()
        logger.info("OCR 阶段就绪")

    def run(self, image, prompt: str = PROMPT_TEXT) -> str:
        """识别一张图片。

        Args:
            image: 图片路径或 URL
            prompt: GLM-OCR 的解析指令

        Returns:
            识别出的文本（公式会以 LaTeX 形式给出）
        """
        if self.engine is None:
            raise RuntimeError("请先调用 load()")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": str(image)},
                {"type": "text", "text": prompt},
            ],
        }]
        seq = self.engine.build_sequence(messages)
        self.engine.add_request(seq)
        self.engine.run_until_done()
        return self.engine.decode(seq).strip()

    def run_multi(self, image, prompts: list[str]) -> dict[str, str]:
        """用多个 prompt 识别同一张图，结果合并。

        数学题里文字和公式是混排的，用单一 prompt 可能漏内容，
        所以分别跑 Text / Formula，再把结果拼起来。
        """
        results = {}
        for p in prompts:
            results[p] = self.run(image, p)
        return results

    def run_batch(self, images: list, prompt: str = PROMPT_TEXT) -> list[str]:
        """批量识别多张图（利用引擎的连续批处理）。"""
        if self.engine is None:
            raise RuntimeError("请先调用 load()")

        seqs = []
        for img in images:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "url": str(img)},
                    {"type": "text", "text": prompt},
                ],
            }]
            seq = self.engine.build_sequence(messages)
            self.engine.add_request(seq)
            seqs.append(seq)

        self.engine.run_until_done()
        return [self.engine.decode(s).strip() for s in seqs]

    def unload(self) -> None:
        """释放显存。"""
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None


def clean_ocr_text(text: str) -> str:
    """清理 OCR 输出中的噪声。

    GLM-OCR 有时会带上对话模板残留（如 <|user|>）或多余空行。
    """
    # 去掉特殊 token 残留
    text = re.sub(r"<\|[^|]*\|>", "", text)
    # 合并多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
