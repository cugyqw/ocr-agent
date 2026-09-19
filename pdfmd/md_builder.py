"""Markdown 构建：从 PDF 提取内容，产出 Markdown。

两条路径：
    1. extract_text_pages() —— 电子版，用 pdfplumber 提取文字与表格
    2. ocr_pages()          —— 扫描版，转图片后交给 GLM-OCR

关于版面还原的说明：
    本模块产出的是"内容级"Markdown（标题、段落、表格），
    不保留字体、颜色、精确位置。要做到像素级还原需要
    版面分析模型（如 MinerU），不在轻量方案范围内。
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from .text_rules import is_meaningful_formula, looks_like_heading

logger = logging.getLogger(__name__)



def render_pdf_pages(
    pdf_path: str | Path,
    out_dir: str | Path,
    dpi: int = 150,
    max_pages: Optional[int] = None,
) -> list[Path]:
    """用 pdftoppm 把 PDF 每页渲染成 PNG。

    pdftoppm 是 poppler-utils 提供的工具，系统已安装。

    【DPI 选择很重要】
    GLM-OCR 的图片 token 数约等于像素数 / 1000，上限 4096。
    A4 页面(8.27x11.69 inch) 在不同 DPI 下的实测 token 数：

        200 DPI -> 1654x2339 = 3.87MP -> 4968 token  ❌ 超限
        150 DPI -> 1240x1754 = 2.17MP -> 2784 token  ✅ 可用
        120 DPI ->  992x1403 = 1.39MP -> 1762 token  ✅ 安全

    所以默认用 150 DPI。若页面更大（如 A3）或引擎 max_model_len 更小，
    需要进一步降低。

    Args:
        dpi: 渲染分辨率，默认 150（兼顾清晰度与 token 上限）
        max_pages: 最多渲染多少页

    Returns:
        生成的图片路径列表（按页序）
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = out_dir / "page"
    cmd = ["pdftoppm", "-png", "-r", str(dpi)]
    if max_pages:
        cmd += ["-f", "1", "-l", str(max_pages)]
    cmd += [str(pdf_path), str(prefix)]

    logger.info("渲染 PDF: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm 失败: {result.stderr}")

    # pdftoppm 输出形如 page-1.png / page-01.png，按数字排序
    imgs = sorted(
        out_dir.glob("page-*.png"),
        key=lambda p: int(p.stem.split("-")[-1]),
    )
    logger.info("生成 %d 张图片", len(imgs))
    return imgs


def extract_text_pages(pdf_path: str | Path) -> list[dict]:
    """提取电子版 PDF 的每页文字与表格。

    Returns:
        [{"page": 1, "text": "...", "tables": [[[...]]]}, ...]
    """
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            tables = []
            try:
                for t in page.extract_tables():
                    if t:
                        tables.append(t)
            except Exception as e:  # noqa: BLE001
                logger.warning("第 %d 页表格提取失败: %s", i, e)
            pages.append({"page": i, "text": text, "tables": tables})
    return pages


def ocr_pages(
    image_paths: list[Path],
    ocr_func: Callable[[str, str], str],
    prompts: list[str] | None = None,
) -> list[dict]:
    """对每页图片做 OCR。

    Args:
        image_paths: 页面图片
        ocr_func: OCR 调用函数，签名 (图片路径, prompt) -> 文本
        prompts: 要跑的 prompt 列表（默认只跑 Text Recognition）

    Returns:
        [{"page": 1, "text": "...", "formula": "..."}, ...]
    """
    from solver.ocr_stage import PROMPT_FORMULA, PROMPT_TEXT

    if prompts is None:
        prompts = [PROMPT_TEXT, PROMPT_FORMULA]

    want_formula = PROMPT_FORMULA in prompts

    results = []
    for i, img in enumerate(image_paths, 1):
        text = ocr_func(str(img), PROMPT_TEXT)
        formula = ocr_func(str(img), PROMPT_FORMULA) if want_formula else ""
        results.append({"page": i, "text": text, "formula": formula})
        logger.info("OCR 第 %d/%d 页完成", i, len(image_paths))
    return results



def pages_to_markdown(pages: list[dict], source: str = "", ocr_mode: bool = False) -> str:
    """把页面内容拼成 Markdown。

    Args:
        pages: extract_text_pages 或 ocr_pages 的输出
        source: 源文件名，写入文档头
        ocr_mode: 是否来自 OCR（OCR 输出已含 LaTeX，不再加工）
    """
    lines = []
    if source:
        lines.append(f"# {Path(source).stem}")
        lines.append("")

    for p in pages:
        lines.append(f"<!-- 第 {p['page']} 页 -->")
        lines.append("")

        text = (p.get("text") or "").strip()
        if text:
            for raw in text.split("\n"):
                line = raw.rstrip()
                if not line.strip():
                    lines.append("")
                    continue
                if not ocr_mode and looks_like_heading(line):
                    lines.append(f"## {line.strip()}")
                else:
                    lines.append(line)
            lines.append("")

        # 公式（OCR 模式）。无效结果（一堆空 $$）直接丢弃。
        formula = (p.get("formula") or "").strip()
        if formula and is_meaningful_formula(formula):
            lines.append("### 公式")
            lines.append("")
            lines.append(formula)
            lines.append("")

        # 表格（电子版）
        for ti, table in enumerate(p.get("tables") or [], 1):
            if not table:
                continue
            lines.append(f"**表格 {ti}**")
            lines.append("")
            lines.extend(_table_to_md(table))
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _table_to_md(table: list[list]) -> list[str]:
    """把 pdfplumber 的表格转成 Markdown 表格。"""
    out = []
    rows = [[("" if c is None else str(c).replace("\n", " ")).strip() for c in row]
            for row in table]
    if not rows:
        return out

    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]

    out.append("| " + " | ".join(rows[0]) + " |")
    out.append("|" + "---|" * ncol)
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return out
