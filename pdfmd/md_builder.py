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
import tempfile
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# 中文字体判断：含较多 CJK 字符的行更可能是标题
_CJK_RANGE = ("\u4e00", "\u9fff")


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


def _is_meaningful_formula(text: str) -> bool:
    """判断公式识别结果是否有效。

    背景：对没有公式的页面硬跑 "Formula Recognition:"，模型有两种
    错误行为，都必须过滤：

    1. 输出一堆空的 $$ 标记（实测能刷出几百行）
    2. 把普通文本强行包进 $$ 里，例如把"会议纪要/时间/地点"整段
       套上 \\mathrm{} 和 $$，这不是公式，是模型在"完成任务的表演"

    判断标准：去掉标记后，内容里必须有真正的数学特征
    （运算符、LaTeX 数学命令），否则判为无效。
    """
    if not text:
        return False

    # 去掉 $$ 与环境标记
    cleaned = re.sub(r"\$\$|\\begin\{[^}]*\}|\\end\{[^}]*\}", "", text)
    cleaned = re.sub(r"[\s\[\]{}]", "", cleaned)

    if len(cleaned) < 4:
        return False

    # LaTeX 数学命令（注意排除 \mathrm \text \begin \end \left \right
    # 这类"排版壳"，它们常被用来包装普通文本）
    n_latex_math = len(re.findall(
        r"\\(frac|sqrt|sum|int|lim|alpha|beta|gamma|theta|pi|times|cdot|"
        r"div|pm|mp|leq|geq|neq|approx|infty|partial|nabla|log|ln|sin|cos|tan)",
        text,
    ))

    # 数学运算符与数字（数字单独出现不算，需配合运算符）
    n_math_chars = len(re.findall(r"[+\-*/=^_<>]", cleaned))
    n_digits = len(re.findall(r"[0-9]", cleaned))

    if n_latex_math >= 1:
        return True
    if n_math_chars >= 1 and n_digits >= 1:
        return True
    if n_math_chars >= 3:
        return True
    # 表格/矩阵结构：array/matrix 环境且有列分隔符与换行符
    if re.search(r"\\begin\{(array|matrix|pmatrix|bmatrix|cases)", text) \
            and "&" in text and "\\\\" in text:
        return True

    return False


def _contains_math(s: str) -> bool:
    """判断一行是否包含数学公式。

    背景：公式行往往"短、无标点"，会被标题启发式误判。
    实测 "S = pi * r^2" 就被错当成标题了。
    """
    # 等号、运算符密集
    if s.count("=") >= 1 and len(s) < 60:
        # 排除"答案是 X"这类正常句子
        if re.search(r"[+\-*/^_\\]|sqrt|frac|sum|int|pi\b|log|sin|cos|tan", s):
            return True
    # LaTeX 命令
    if re.search(r"\\[a-zA-Z]+", s):
        return True
    # 纯符号/数字构成的短行
    if len(s) <= 30 and re.fullmatch(r"[\s0-9a-zA-Z+\-*/=^_(){}\[\].,<>]+", s) \
            and re.search(r"[+\-*/=^]", s):
        return True
    return False


def _looks_like_heading(line: str) -> bool:
    """粗略判断一行是否像标题。

    规则：短、不含句末标点、行首常见标题特征。
    这是启发式判断，不追求完美——真正的版面分析需要模型。
    """
    s = line.strip()
    if not s or len(s) > 40:
        return False
    # 以句号/逗号结尾的多半是正文
    if s[-1] in "。，；：、,.;:":
        return False
    # 公式行不是标题
    if _contains_math(s):
        return False
    # 常见标题模式
    if re.match(r"^(第[一二三四五六七八九十百\d]+[章节讲部分课]|[\d]+[\.、]\s*\S)", s):
        return True
    # 纯短行且无标点
    if len(s) <= 20 and not any(c in s for c in "。，；：、,.;:！？!?"):
        return True
    return False


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
                if not ocr_mode and _looks_like_heading(line):
                    lines.append(f"## {line.strip()}")
                else:
                    lines.append(line)
            lines.append("")

        # 公式（OCR 模式）。无效结果（一堆空 $$）直接丢弃。
        formula = (p.get("formula") or "").strip()
        if formula and _is_meaningful_formula(formula):
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
