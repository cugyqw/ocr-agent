"""拍照解题流水线：图片 → OCR → 解题。

两个模型都常驻显存，全程只加载一次。

显存预算（RTX 5060 Ti 8GB）：
    GLM-OCR (0.9B, bf16)      ~2.1 GB
    Qwen2.5-Math (1.5B, bf16) ~3.1 GB
    合计                      ~5.2 GB
    加上 CUDA 上下文与 KV Cache，峰值约 6~7 GB
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .ocr_stage import OCRStage, PROMPT_FORMULA, PROMPT_TEXT, clean_ocr_text
from .solve_stage import SolveStage

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """一次完整求解的结果。"""

    image: str
    ocr_text: str = ""
    ocr_formula: str = ""
    answer: str = ""
    timings: dict = field(default_factory=dict)

    @property
    def problem(self) -> str:
        """合并后的题目文本（文字 + 公式）。"""
        parts = []
        if self.ocr_text:
            parts.append(self.ocr_text)
        if self.ocr_formula:
            parts.append(f"\n[公式识别结果]\n{self.ocr_formula}")
        return "\n".join(parts).strip()


class MathSolverPipeline:
    """拍照解题流水线。"""

    def __init__(
        self,
        ocr_model_path: str = "/home/yqw/桌面/ocr_agent/GLM-OCR",
        solve_model_path: str = "/home/yqw/桌面/ocr_agent/Qwen2.5-Math-1.5B-Instruct",
        use_formula_prompt: bool = True,
    ):
        self.ocr = OCRStage(model_path=ocr_model_path)
        self.solver = SolveStage(model_path=solve_model_path)
        # 数学题建议同时跑 Text 和 Formula 两个 prompt，避免漏掉公式
        self.use_formula_prompt = use_formula_prompt

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载两个模型（各一次，常驻显存）。"""
        t0 = time.time()
        self.ocr.load()
        self.solver.load()
        logger.info("流水线就绪，总共耗时 %.1fs", time.time() - t0)
        if torch.cuda.is_available():
            logger.info("显存占用: %.1f GB",
                        torch.cuda.memory_allocated() / 1024**3)

    def unload(self) -> None:
        """释放两个模型。"""
        self.ocr.unload()
        self.solver.unload()

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def solve(
        self,
        image,
        extra_instruction: str = "",
        history: list[dict] | None = None,
    ) -> SolveResult:
        """对一张图片完成"识别 + 解题"。

        Args:
            image: 图片路径或 URL
            extra_instruction: 附加给解题模型的额外要求
            history: 多轮对话历史（用于追问）

        Returns:
            SolveResult
        """
        image = str(image)
        result = SolveResult(image=image)

        # ---- 阶段一：OCR ----
        t0 = time.time()
        result.ocr_text = clean_ocr_text(self.ocr.run(image, PROMPT_TEXT))
        result.timings["ocr_text"] = time.time() - t0

        if self.use_formula_prompt:
            t0 = time.time()
            result.ocr_formula = clean_ocr_text(self.ocr.run(image, PROMPT_FORMULA))
            result.timings["ocr_formula"] = time.time() - t0

        # ---- 阶段二：解题 ----
        problem = result.problem
        if extra_instruction:
            problem = f"{problem}\n\n{extra_instruction}"

        if not problem:
            result.answer = "[OCR 未识别出内容，请检查图片是否清晰]"
            return result

        t0 = time.time()
        result.answer = self.solver.run(problem, history=history)
        result.timings["solve"] = time.time() - t0

        result.timings["total"] = sum(
            v for k, v in result.timings.items() if k != "total"
        )
        return result

    def solve_many(self, images: list, extra_instruction: str = "") -> list[SolveResult]:
        """批量解题。

        说明：OCR 阶段会批处理（利用连续批处理），
        解题阶段目前逐条（生成 token 多，批量收益受等长约束）。
        """
        images = [str(i) for i in images]
        results = [SolveResult(image=img) for img in images]

        # OCR 批量
        t0 = time.time()
        texts = self.ocr.run_batch(images, PROMPT_TEXT)
        for r, t in zip(results, texts):
            r.ocr_text = clean_ocr_text(t)
        ocr_time = time.time() - t0

        if self.use_formula_prompt:
            formulas = [self.ocr.run(img, PROMPT_FORMULA) for img in images]
            for r, f in zip(results, formulas):
                r.ocr_formula = clean_ocr_text(f)

        for r in results:
            r.timings["ocr_batch"] = ocr_time / max(len(images), 1)

        # 解题逐条
        for r in results:
            problem = r.problem
            if extra_instruction:
                problem = f"{problem}\n\n{extra_instruction}"
            if not problem:
                r.answer = "[OCR 未识别出内容]"
                continue
            t0 = time.time()
            r.answer = self.solver.run(problem)
            r.timings["solve"] = time.time() - t0

        return results
