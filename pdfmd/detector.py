"""PDF 类型检测。

判断 PDF 是"电子版"（含文本层）还是"扫描版"（纯图片），
这决定了走哪条处理路径：

    电子版 → 直接提取文字（100% 准确，无需 OCR）
    扫描版 → 走 OCR（GLM-OCR）

为什么这个判断很重要：
    对电子版 PDF 用 OCR 是"降级"——本来文字是精确的，
    经过 OCR 反而会引入识别错误。
"""

from __future__ import annotations

import enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PdfType(enum.Enum):
    """PDF 类型。"""

    TEXT = "text"        # 电子版，含文本层
    SCANNED = "scanned"  # 扫描版，需要 OCR
    MIXED = "mixed"      # 混合：部分页有文本，部分没有
    EMPTY = "empty"      # 空文档或无有效内容


def detect_pdf_type(
    pdf_path: str | Path,
    sample_pages: int = 5,
    min_chars: int = 20,
) -> tuple[PdfType, dict]:
    """检测 PDF 类型。

    策略：抽样前若干页，统计每页可提取的字符数。
    电子版每页通常有几百字符，扫描版接近 0。

    Args:
        pdf_path: PDF 文件路径
        sample_pages: 抽样页数
        min_chars: 一页至少多少字符才算"有文本"

    Returns:
        (PdfType, 详情字典)
    """
    import pdfplumber

    pdf_path = Path(pdf_path)
    detail = {
        "path": str(pdf_path),
        "total_pages": 0,
        "sampled_pages": 0,
        "text_pages": 0,
        "scanned_pages": 0,
        "chars_per_page": [],
    }

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        detail["total_pages"] = total
        if total == 0:
            return PdfType.EMPTY, detail

        n = min(sample_pages, total)
        detail["sampled_pages"] = n

        for i in range(n):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            n_chars = len(text.strip())
            detail["chars_per_page"].append(n_chars)
            if n_chars >= min_chars:
                detail["text_pages"] += 1
            else:
                detail["scanned_pages"] += 1

    if detail["text_pages"] == n:
        pdf_type = PdfType.TEXT
    elif detail["text_pages"] == 0:
        pdf_type = PdfType.SCANNED
    else:
        pdf_type = PdfType.MIXED

    logger.info(
        "PDF 类型: %s（抽样 %d 页，有文本 %d 页，每页字符数 %s）",
        pdf_type.value, n, detail["text_pages"], detail["chars_per_page"],
    )
    return pdf_type, detail


def describe_pdf_type(pdf_type: PdfType) -> str:
    """类型的中文描述。"""
    return {
        PdfType.TEXT: "电子版（可直接提取文字）",
        PdfType.SCANNED: "扫描版（需要 OCR）",
        PdfType.MIXED: "混合版（部分页需 OCR）",
        PdfType.EMPTY: "空文档",
    }.get(pdf_type, "未知")
