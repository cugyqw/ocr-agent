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

                if not ocr_mode and _looks_like_heading(line):
                    doc.add_heading(line.strip(), level=1)
                else:
                    par = doc.add_paragraph()
                    _add_runs_with_emphasis(par, line)

        # 公式。无效结果（一堆空 $$）直接丢弃。
        formula = sanitize(p.get("formula") or "").strip()
        if formula and _is_meaningful_formula(formula):
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


def _is_meaningful_formula(text: str) -> bool:
    """判断公式识别结果是否有效（与 md_builder 保持一致的逻辑）。

    过滤两类噪声：
    1. 无公式页面输出的大量空 $$ 标记
    2. 把普通文本强行包进 $$ / \\mathrm{} 的"伪公式"
    """
    if not text:
        return False
    cleaned = re.sub(r"\$\$|\\begin\{[^}]*\}|\\end\{[^}]*\}", "", text)
    cleaned = re.sub(r"[\s\[\]{}]", "", cleaned)
    if len(cleaned) < 4:
        return False

    n_latex_math = len(re.findall(
        r"\\(frac|sqrt|sum|int|lim|alpha|beta|gamma|theta|pi|times|cdot|"
        r"div|pm|mp|leq|geq|neq|approx|infty|partial|nabla|log|ln|sin|cos|tan)",
        text,
    ))
    n_math_chars = len(re.findall(r"[+\-*/=^_<>]", cleaned))
    n_digits = len(re.findall(r"[0-9]", cleaned))

    if n_latex_math >= 1:
        return True
    if n_math_chars >= 1 and n_digits >= 1:
        return True
    if n_math_chars >= 3:
        return True
    # 表格/矩阵结构
    if re.search(r"\\begin\{(array|matrix|pmatrix|bmatrix|cases)", text) \
            and "&" in text and "\\\\" in text:
        return True
    return False


def _contains_math(s: str) -> bool:
    """判断一行是否包含数学公式（与 md_builder 保持一致）。"""
    if s.count("=") >= 1 and len(s) < 60:
        if re.search(r"[+\-*/^_\\]|sqrt|frac|sum|int|pi\b|log|sin|cos|tan", s):
            return True
    if re.search(r"\\[a-zA-Z]+", s):
        return True
    if len(s) <= 30 and re.fullmatch(r"[\s0-9a-zA-Z+\-*/=^_(){}\[\].,<>]+", s) \
            and re.search(r"[+\-*/=^]", s):
        return True
    return False


def _looks_like_heading(line: str) -> bool:
    """与 md_builder 保持一致的标题启发式判断。"""
    s = line.strip()
    if not s or len(s) > 40:
        return False
    if s[-1] in "。，；：、,.;:":
        return False
    # 公式行不是标题
    if _contains_math(s):
        return False
    if re.match(r"^(第[一二三四五六七八九十百\d]+[章节讲部分课]|[\d]+[\.、]\s*\S)", s):
        return True
    if len(s) <= 20 and not any(c in s for c in "。，；：、,.;:！？!?"):
        return True
    return False


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
