"""拍照解题流水线。

架构：两阶段串联 + 科目路由
    图片 → [GLM-OCR: 识图] → 文本(含LaTeX)
         → [科目识别]
         → 数学: Qwen2.5-Math / 其他: Qwen2.5 通用
         → 解答

OCR 模型常驻显存；解题模型按科目按需加载。
"""

from .pipeline import MathSolverPipeline, SolveResult
from .ocr_stage import OCRStage
from .solve_stage import SolveStage
from .subject import detect_subject, describe

__all__ = [
    "MathSolverPipeline",
    "SolveResult",
    "OCRStage",
    "SolveStage",
    "detect_subject",
    "describe",
]
