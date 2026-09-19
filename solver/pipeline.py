"""拍照解题流水线：图片 → OCR → 科目识别 → 解题。

架构：
    GLM-OCR 常驻显存（识图）
        ↓
    科目识别（规则判断，零成本）
        ↓
    解题模型按需加载：
        数学  → Qwen2.5-Math-1.5B
        其他  → Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

from .ocr_stage import OCRStage, PROMPT_FORMULA, PROMPT_TEXT, clean_ocr_text
from .solve_stage import SolveStage
from .subject import describe, detect_subject

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """一次完整求解的结果。"""

    image: str
    ocr_text: str = ""
    ocr_formula: str = ""
    subject: str = "general"
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

    @property
    def subject_name(self) -> str:
        """科目中文名。"""
        return describe(self.subject)


class MathSolverPipeline:
    """拍照解题流水线（兼容旧名，现已支持多科目）。"""

    def __init__(
        self,
        ocr_model_path: str = "/home/yqw/桌面/ocr_agent/GLM-OCR",
        math_model_path: str = "/home/yqw/桌面/ocr_agent/Qwen2.5-Math-1.5B-Instruct",
        general_model_path: str = "/home/yqw/桌面/ocr_agent/Qwen2.5-1.5B-Instruct",
        use_formula_prompt: bool = True,
        force_subject: str | None = None,
    ):
        """
        Args:
            use_formula_prompt: 是否额外跑公式识别 prompt
            force_subject: 强制指定科目（跳过自动识别），便于调试
        """
        self.ocr = OCRStage(model_path=ocr_model_path)
        self.solver = SolveStage(
            math_model_path=math_model_path,
            general_model_path=general_model_path,
        )
        self.use_formula_prompt = use_formula_prompt
        self.force_subject = force_subject

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载 OCR 模型（解题模型按需加载）。"""
        t0 = time.time()
        self.ocr.load()
        self.solver.load()
        logger.info("流水线就绪，耗时 %.1fs", time.time() - t0)

    def unload(self) -> None:
        """释放所有模型。"""
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
        """对一张图片完成"识别 + 解题"。"""
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

        # ---- 科目识别 ----
        t0 = time.time()
        if self.force_subject:
            result.subject = self.force_subject
        else:
            result.subject = detect_subject(result.problem)
        result.timings["subject"] = time.time() - t0
        logger.info("科目识别: %s", result.subject_name)

        # ---- 阶段二：解题 ----
        problem = result.problem
        if extra_instruction:
            problem = f"{problem}\n\n{extra_instruction}"

        if not problem.strip():
            result.answer = "[OCR 未识别出内容，请检查图片是否清晰]"
            return result

        t0 = time.time()
        result.answer = self.solver.run(problem, subject=result.subject, history=history)
        result.timings["solve"] = time.time() - t0

        result.timings["total"] = sum(
            v for k, v in result.timings.items() if k != "total"
        )
        return result

    def solve_many(self, images: list, extra_instruction: str = "") -> list[SolveResult]:
        """批量解题。OCR 阶段批处理，解题阶段按科目逐个处理。"""
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
            r.subject = self.force_subject or detect_subject(r.problem)

        # 解题：按科目分组，同科目的连续处理可减少模型切换
        order = sorted(range(len(results)), key=lambda i: results[i].subject)
        for i in order:
            r = results[i]
            problem = r.problem
            if extra_instruction:
                problem = f"{problem}\n\n{extra_instruction}"
            if not problem.strip():
                r.answer = "[OCR 未识别出内容]"
                continue
            t0 = time.time()
            r.answer = self.solver.run(problem, subject=r.subject)
            r.timings["solve"] = time.time() - t0

        return results
