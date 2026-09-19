"""阶段一：OCR 识别。

职责：把图片转成文本。数学题的关键是公式要转成 LaTeX，
GLM-OCR 的 "Formula Recognition:" prompt 正好负责这件事。
"""

from __future__ import annotations

import logging
import re

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
        max_model_len: int = 4096,
    ):
        """
        Args:
            max_model_len: 单条序列最大长度（prompt + 生成）。
                GLM-OCR 的图片 token 数约等于像素数/1000：
                    单张图片(A4, 150DPI) 约 2800 token
                    PDF 整页扫描件可能更大
                PDF 场景建议调到 8192。
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.engine: MiniInferEngine | None = None

    def load(self) -> None:
        """加载并常驻 GLM-OCR。"""
        self.engine = MiniInferEngine(EngineConfig(
            model_path=self.model_path,
            max_batch_size=2,       # OCR 阶段并发需求低，省显存
            max_new_tokens=self.max_new_tokens,
            max_model_len=self.max_model_len,
        ))
        self.engine.load_model()
        logger.info("OCR 阶段就绪（max_model_len=%d）", self.max_model_len)

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
        ok = self.engine.add_request(seq)
        self.engine.run_until_done()

        # 注意：入队失败或序列被中途拒绝时，output 为空。
        # 之前这里直接返回空串，导致调用方完全看不出失败原因，
        # 排查起来很费劲。这里改为抛出异常并说明原因。
        if not ok:
            raise RuntimeError(f"请求入队失败（队列已满）: {image}")

        text = self.engine.decode(seq)
        if not text.strip():
            raise RuntimeError(
                f"OCR 未产出内容（seq_id={seq.seq_id}, "
                f"prompt_len={seq.prompt_len}, 图片可能过大导致 token 超限）: {image}"
            )
        return text.strip()

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
