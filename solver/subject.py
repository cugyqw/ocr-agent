"""科目识别。

用规则判断题目属于哪一科，决定路由到哪个解题模型。

为什么不用模型判断？
    规则足够准（科目特征明显），且零成本、零延迟。
    小学阶段的题目科目区分度很高（数学有大量数字符号，
    英语有大量拉丁字母，语文有汉字特征）。

设计取舍：
    宁可错判为 "general"，也不要错判为 "math"——
    因为 general 模型什么都能答一点，而 math 模型完全不会语文。
"""

from __future__ import annotations

import re

# 数学特征：数字、运算符、数学符号、数学术语
MATH_SYMBOLS = re.compile(
    r"[0-9]"                      # 数字
    r"|[+\-×÷*/=<>≤≥≠±∑∫√^²³]"   # 运算符
    r"|\\frac|\\sqrt|\\int|\\lim"  # LaTeX 数学命令
)

MATH_KEYWORDS = [
    "计算", "求解", "解方程", "方程", "函数", "几何", "三角形", "长方形",
    "正方形", "圆", "面积", "周长", "体积", "分数", "小数", "百分数",
    "比", "比例", "速度", "路程", "应用题", "求值", "化简", "证明",
    "平行", "垂直", "角", "度", "统计", "平均数", "可能性",
]

# 英语特征：整句拉丁字母、常见英文单词、英文题型标记
ENGLISH_SENTENCE = re.compile(r"[A-Za-z]{3,}")
ENGLISH_KEYWORDS = [
    "translate", "translation", "choose", "fill in", "complete",
    "read the passage", "answer the question", "true or false",
    "correct answer", "underline", "write", "match",
    "翻译", "选择填空", "阅读理解", "完形填空", "改错",
    "english", "word", "sentence", "grammar",
]

# 语文特征：古诗文、语文术语
CHINESE_KEYWORDS = [
    "古诗", "诗句", "作者", "诗人", "文言文", "解释", "翻译成中文",
    "近义词", "反义词", "造句", "组词", "拼音", "汉字", "笔画",
    "阅读理解", "概括", "中心思想", "修辞", "比喻", "拟人", "排比",
    "课文", "填空", "默写", "改写", "句式", "病句", "标点",
]


def detect_subject(text: str) -> str:
    """识别题目科目。

    Args:
        text: 题目文本（OCR 结果）

    Returns:
        "math" / "english" / "chinese" / "general"
    """
    if not text or not text.strip():
        return "general"

    t = text.strip()
    lower = t.lower()

    # 中文汉字数量
    n_han = len(re.findall(r"[\u4e00-\u9fff]", t))
    # 拉丁字母数量
    n_latin = len(re.findall(r"[A-Za-z]", t))
    # 数字与数学符号数量
    n_math = len(MATH_SYMBOLS.findall(t))

    # ---- 1) 英语题判定 ----
    # 英文字母占压倒性多数，或命中英文题型关键词
    if n_latin > n_han * 1.5 and n_latin > 10:
        return "english"
    if any(k in lower for k in ENGLISH_KEYWORDS):
        # 若同时有大量汉字，可能是"翻译成中文"类语文题
        if n_han > n_latin:
            return "chinese"
        return "english"

    # ---- 2) 数学题判定 ----
    # 命中数学关键词
    if any(k in t for k in MATH_KEYWORDS):
        return "math"
    # 数学符号密度高，且汉字较少
    if n_math >= 3 and n_han < n_latin * 2 + 20:
        return "math"

    # ---- 3) 语文题判定 ----
    if any(k in t for k in CHINESE_KEYWORDS):
        return "chinese"

    # ---- 4) 默认 ----
    return "general"


def describe(subject: str) -> str:
    """科目中文名，用于展示。"""
    return {
        "math": "数学",
        "chinese": "语文",
        "english": "英语",
        "general": "通用",
    }.get(subject, "通用")
