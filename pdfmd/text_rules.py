"""文本判断规则（Markdown 与 Word 构建共用）。

原本 md_builder.py 和 docx_builder.py 各自实现了一份完全相同的
_is_meaningful_formula / _contains_math / _looks_like_heading，
维护时需要同步改两处，容易漂移。这里抽成公共模块。

这些是**启发式规则**，不追求完美——真正的版面分析需要模型。
但它们在电子版 PDF 上效果不错（实测能正确识别"第一章 xxx"、
"3.1 xxx"等标题，并排除公式行）。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 公式判断
# ---------------------------------------------------------------------------

# LaTeX 数学命令。注意排除 \mathrm \text \begin \end \left \right 这类
# "排版壳"——它们常被模型用来包装普通文本，不能作为"是公式"的依据。
_LATEX_MATH = re.compile(
    r"\\(frac|sqrt|sum|int|lim|alpha|beta|gamma|theta|pi|times|cdot|"
    r"div|pm|mp|leq|geq|neq|approx|infty|partial|nabla|log|ln|sin|cos|tan)"
)

# 数学运算符
_MATH_CHARS = re.compile(r"[+\-*/=^_<>]")
_DIGITS = re.compile(r"[0-9]")

# 矩阵/表格环境
_MATRIX_ENV = re.compile(r"\\begin\{(array|matrix|pmatrix|bmatrix|cases)")


def is_meaningful_formula(text: str) -> bool:
    """判断公式识别结果是否有效。

    背景：对没有公式的页面硬跑 "Formula Recognition:"，模型有两种
    错误行为，都必须过滤：

    1. 输出一堆空的 $$ 标记（实测能刷出几百行）
    2. 把普通文本强行包进 $$ / \\mathrm{}，例如把"会议纪要/时间/地点"
       整段套上，这不是公式，是模型在"完成任务的表演"

    判断标准：去掉标记后，内容里必须有真正的数学特征。
    """
    if not text:
        return False

    # 去掉 $$ 与环境标记
    cleaned = re.sub(r"\$\$|\\begin\{[^}]*\}|\\end\{[^}]*\}", "", text)
    cleaned = re.sub(r"[\s\[\]{}]", "", cleaned)

    if len(cleaned) < 4:
        return False

    if _LATEX_MATH.search(text):
        return True

    n_math = len(_MATH_CHARS.findall(cleaned))
    n_digits = len(_DIGITS.findall(cleaned))

    if n_math >= 1 and n_digits >= 1:
        return True
    if n_math >= 3:
        return True
    # 矩阵/表格结构：有列分隔符与换行符
    if _MATRIX_ENV.search(text) and "&" in text and "\\\\" in text:
        return True

    return False


# ---------------------------------------------------------------------------
# 标题判断
# ---------------------------------------------------------------------------

# 常见标题开头："第一章"、"第 3 讲"、"1."、"1、"
_HEADING_PREFIX = re.compile(
    r"^(第[一二三四五六七八九十百\d]+[章节讲部分课]|[\d]+[\.、]\s*\S)"
)

# 句末标点（出现则多半是正文）
_SENTENCE_END = "。，；：、,.;:！？!?"


def contains_math(s: str) -> bool:
    """判断一行是否包含数学公式。

    背景：公式行往往"短、无标点"，会被标题启发式误判。
    实测 "S = pi * r^2" 就被错当成标题了。
    """
    # 含等号且不太长，同时存在运算符/函数名
    if s.count("=") >= 1 and len(s) < 60:
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


def looks_like_heading(line: str) -> bool:
    """粗略判断一行是否像标题。

    规则：短、不含句末标点、且不是公式。
    """
    s = line.strip()
    if not s or len(s) > 40:
        return False
    if s[-1] in _SENTENCE_END:
        return False
    # 公式行不是标题
    if contains_math(s):
        return False
    if _HEADING_PREFIX.match(s):
        return True
    # 短行且无句末标点
    if len(s) <= 20 and not any(c in s for c in _SENTENCE_END):
        return True
    return False
