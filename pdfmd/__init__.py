"""PDF 转换模块：PDF → Word / Markdown。

两条处理路径，自动判断：
    PDF → 有文本层? ── 是 → 直接提取（快、准、零成本）
                    └─ 否 → GLM-OCR 识别（扫描件）

输出支持 Markdown 与 DOCX。
"""

from .converter import PdfConverter, ConversionResult
from .detector import detect_pdf_type, PdfType

__all__ = [
    "PdfConverter",
    "ConversionResult",
    "detect_pdf_type",
    "PdfType",
]
