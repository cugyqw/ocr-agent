"""拍照解题流水线。

架构：两阶段串联，各司其职
    图片 → [GLM-OCR: 识图] → 文本(含LaTeX公式) → [Qwen2.5-Math: 解题] → 解答

两个模型都常驻显存，避免反复加载。
"""

from .pipeline import MathSolverPipeline
from .ocr_stage import OCRStage
from .solve_stage import SolveStage

__all__ = ["MathSolverPipeline", "OCRStage", "SolveStage"]
