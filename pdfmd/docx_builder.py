"""DOCX 生成：把页面内容写入 Word 文档。

将 Markdown 的层级语义映射到 Word 的样式：
    #      → Heading 0（标题）
    ##     → Heading 1
    ###    → Heading 2
    段落    → Normal
    | 表格 | → Word 表格

关于版面还原的说明：
    Word 是"流式文档"，PDF 是"固定版面文档"，两者模型不同。
    本模块保留的是**结构**（标题层级、段落、表格），
    而非**绝对位置**（字体、分栏、图片坐标）。
    要做到后者需要版面分析模型，不在轻量方案范围内。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .text_rules import is_meaningful_formula, looks_like_heading

logger = logging.getLogger(__name__)


# XML 不允许的控制字符（除 \t \n \r 外，其余 0x00-0x1F 都非法）
_ILLEGAL_XML = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    r"\u200b-\u200f\u2028\u2029\ufeff]"
)


def sanitize(text: str) -> str:
    """清洗不可见字符。

    PDF 提取的文字里常含零宽空格、控制字符等，直接写入 Word
    会报 "All strings must be XML compatible"。必须清理。
    """
    if not text:
        return ""
    return _ILLEGAL_XML.sub("", text)


def _set_cjk_font(doc, font_name: str = "宋体", size_pt: int | None = None) -> None:
    """设置文档默认字体，并正确处理中文字体。

    python-docx 设置中文字体需要同时设置 eastasia，
    否则中文会回退到默认字体。
    """
    from docx.oxml.ns import qn

    style = doc.styles["Normal"]
    style.font.name = font_name
    if size_pt:
        from docx.shared import Pt
        style.font.size = Pt(size_pt)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), font_name)


def pages_to_docx(
    pages: list[dict],
    out_path: str | Path,
    title: str = "",
    ocr_mode: bool = False,
) -> Path:
    """把页面内容写入 DOCX。

    Args:
        pages: extract_text_pages 或 ocr_pages 的输出
        out_path: 输出 .docx 路径
        title: 文档标题
        ocr_mode: 是否来自 OCR

    Returns:
        输出文件路径
    """
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    _set_cjk_font(doc, "宋体", 11)

    # ---- 标题 ----
    if title:
        doc.add_heading(sanitize(title), level=0)

    for p in pages:
        page_no = p.get("page", 0)
        logger.info("写入第 %d 页", page_no)

        # 分页符（首页不加）
        if page_no > 1:
            doc.add_page_break()

        text = sanitize(p.get("text") or "").strip()
        if text:
            for raw in text.split("\n"):
                line = sanitize(raw).rstrip()
                if not line.strip():
                    continue

                if not ocr_mode and looks_like_heading(line):
                    doc.add_heading(line.strip(), level=1)
                else:
                    par = doc.add_paragraph()
                    _add_runs_with_emphasis(par, line)

        # 公式。无效结果（一堆空 $$）直接丢弃。
        formula = sanitize(p.get("formula") or "").strip()
        if formula and is_meaningful_formula(formula):
            doc.add_heading("公式", level=2)
            for fl in formula.split("\n"):
                fl = sanitize(fl).strip()
                if fl:
                    doc.add_paragraph(fl)

        # 表格
        for table in (p.get("tables") or []):
            if not table:
                continue
            _add_table(doc, table)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    logger.info("已保存: %s", out_path)
    return out_path



def _add_runs_with_emphasis(par, text: str) -> None:
    """把 **粗体** 语法转成 Word 的加粗 run。"""
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = par.add_run(part[2:-2])
            run.bold = True
        else:
            par.add_run(part)


def _add_table(doc, table: list[list]) -> None:
    """把二维列表写成 Word 表格。"""
    rows = [[sanitize("" if c is None else str(c).replace("\n", " ")).strip()
             for c in row] for row in table]
    if not rows:
        return

    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]

    word_table = doc.add_table(rows=len(rows), cols=ncol)
    word_table.style = "Table Grid"

    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            word_table.cell(i, j).text = val

    doc.add_paragraph()
