"""图片预处理工具。

PDF 渲染出的页面常有大片留白（页边距、页尾空白），
文字只占画面很小一部分。这会导致 OCR 注意力被稀释、识别不全。

本模块提供裁剪留白功能，让有效内容占满画面，提高识别率。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def trim_whitespace(
    image_path: str | Path,
    out_path: str | Path | None = None,
    threshold: int = 240,
    padding: int = 20,
    min_content_ratio: float = 0.005,
) -> Path:
    """裁掉图片四周的空白区域。

    Args:
        image_path: 输入图片
        out_path: 输出路径（None 则覆盖原图）
        threshold: 灰度阈值，低于此值视为"有内容"
        padding: 裁剪后保留的边距（像素）
        min_content_ratio: 内容占画面比例低于此值时不裁剪
                           （避免把几乎全白的页面裁成一条线）。
                           注意：扫描件内容占比可能很低（实测 4%），
                           这个阈值不能设大，否则会拒绝有价值的裁剪。

    Returns:
        处理后的图片路径
    """
    import numpy as np
    from PIL import Image

    image_path = Path(image_path)
    out_path = Path(out_path) if out_path else image_path

    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    h, w = arr.shape

    # 找到有内容的行/列
    rows = np.where(arr.min(axis=1) < threshold)[0]
    cols = np.where(arr.min(axis=0) < threshold)[0]

    if len(rows) == 0 or len(cols) == 0:
        logger.warning("图片几乎全白，跳过裁剪: %s", image_path.name)
        return image_path

    ratio = ((rows.max() - rows.min()) * (cols.max() - cols.min())) / (h * w)
    if ratio < min_content_ratio:
        logger.info("内容过少(%.2f%%)，跳过裁剪: %s", ratio * 100, image_path.name)
        return image_path

    # 加边距并裁切
    top = max(0, rows.min() - padding)
    bottom = min(h, rows.max() + padding)
    left = max(0, cols.min() - padding)
    right = min(w, cols.max() + padding)

    saved = (h - (bottom - top)) + (w - (right - left))
    if saved < 20:
        logger.debug("裁剪收益过小，跳过: %s", image_path.name)
        return image_path

    original = Image.open(image_path)
    trimmed = original.crop((left, top, right, bottom))
    trimmed.save(out_path)

    logger.info(
        "裁剪留白 %s: %dx%d -> %dx%d",
        image_path.name, w, h, trimmed.size[0], trimmed.size[1],
    )
    return out_path
