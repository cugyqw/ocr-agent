"""PDF 转换主控。

自动判断 PDF 类型并选择处理路径：

    PDF
     ├── 电子版 → pdfplumber 直接提取（快、准、零成本）
     ├── 扫描版 → pdftoppm 转图 → GLM-OCR
     └── 混合版 → 逐页判断，分别处理

输出：Markdown / DOCX
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .detector import PdfType, describe_pdf_type, detect_pdf_type
from .docx_builder import pages_to_docx
from .image_utils import trim_whitespace
from .md_builder import extract_text_pages, ocr_pages, pages_to_markdown, render_pdf_pages

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """一次转换的结果。"""

    pdf: str
    pdf_type: str = ""
    pages: int = 0
    markdown: str = ""
    md_path: str | None = None
    docx_path: str | None = None
    timings: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"PDF:      {self.pdf}",
            f"类型:     {self.pdf_type}",
            f"页数:     {self.pages}",
        ]
        for k, v in self.timings.items():
            lines.append(f"  {k:<14} {v:.2f}s")
        if self.md_path:
            lines.append(f"Markdown: {self.md_path}")
        if self.docx_path:
            lines.append(f"Word:     {self.docx_path}")
        return "\n".join(lines)


class PdfConverter:
    """PDF 转换器。"""

    def __init__(
        self,
        ocr_model_path: str = "/home/yqw/桌面/ocr_agent/GLM-OCR",
        dpi: int = 150,
        max_ocr_pages: int | None = None,
        use_formula_prompt: bool = True,
        trim_whitespace: bool = True,
        trim_padding: int = 20,
    ):
        """
        Args:
            dpi: 扫描件渲染分辨率
            max_ocr_pages: 最多 OCR 多少页（防止大文档跑太久）
            use_formula_prompt: OCR 时是否额外跑公式识别
            trim_whitespace: 是否裁掉页面留白（提高 OCR 识别率）
            trim_padding: 裁剪后保留的边距像素
        """
        self.ocr_model_path = ocr_model_path
        self.dpi = dpi
        self.max_ocr_pages = max_ocr_pages
        self.use_formula_prompt = use_formula_prompt
        self.trim_whitespace = trim_whitespace
        self.trim_padding = trim_padding
        self._ocr_engine = None

    # ------------------------------------------------------------------
    # OCR 引擎（按需加载，用完可释放）
    # ------------------------------------------------------------------
    def _get_ocr(self):
        """懒加载 OCR 引擎。"""
        if self._ocr_engine is not None:
            return self._ocr_engine

        from solver.ocr_stage import OCRStage

        logger.info("加载 OCR 模型（仅扫描版 PDF 需要）")
        # PDF 页面比单张图片大，需要更大的上下文上限。
        # A4 在 150 DPI 下约 2784 token，留出余量设为 8192。
        stage = OCRStage(
            model_path=self.ocr_model_path,
            max_new_tokens=2048,
            max_model_len=8192,
        )
        stage.load()
        self._ocr_engine = stage
        return stage

    def _ocr_call(self, image_path: str, prompt: str) -> str:
        """OCR 调用适配。"""
        stage = self._get_ocr()
        return stage.run(image_path, prompt)

    def release(self) -> None:
        """释放 OCR 引擎占用的显存。"""
        if self._ocr_engine is not None:
            self._ocr_engine.unload()
            self._ocr_engine = None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def convert(
        self,
        pdf_path: str | Path,
        out_dir: str | Path | None = None,
        to_markdown: bool = True,
        to_docx: bool = True,
    ) -> ConversionResult:
        """转换 PDF。

        Args:
            pdf_path: 输入 PDF
            out_dir: 输出目录（默认与 PDF 同目录）
            to_markdown: 是否输出 Markdown
            to_docx: 是否输出 Word

        Returns:
            ConversionResult
        """
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

        out_dir = Path(out_dir) if out_dir else pdf_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        result = ConversionResult(pdf=str(pdf_path))

        # ---- 1. 类型检测 ----
        t0 = time.time()
        pdf_type, detail = detect_pdf_type(pdf_path)
        result.timings["detect"] = time.time() - t0
        result.pdf_type = describe_pdf_type(pdf_type)
        result.pages = detail["total_pages"]

        if pdf_type == PdfType.EMPTY:
            logger.warning("PDF 无内容")
            return result

        # ---- 2. 按类型处理 ----
        if pdf_type == PdfType.TEXT:
            t0 = time.time()
            pages = extract_text_pages(pdf_path)
            result.timings["extract"] = time.time() - t0
            ocr_mode = False
        else:
            # 扫描版或混合版：渲染成图后 OCR
            t0 = time.time()
            with tempfile.TemporaryDirectory() as tmp:
                imgs = render_pdf_pages(
                    pdf_path, tmp, dpi=self.dpi, max_pages=self.max_ocr_pages
                )

                # 裁剪留白。渲染出的页面常有大片空白，
                # 直接送 OCR 会导致注意力被稀释、识别不全。
                if self.trim_whitespace:
                    for img in imgs:
                        try:
                            trim_whitespace(img, padding=self.trim_padding)
                        except Exception as e:  # noqa: BLE001
                            logger.warning("裁剪失败（跳过）: %s", e)
                result.timings["render"] = time.time() - t0

                t0 = time.time()
                pages = ocr_pages(imgs, self._ocr_call) if self.use_formula_prompt \
                    else ocr_pages(imgs, self._ocr_call, prompts=["Text Recognition:"])
                result.timings["ocr"] = time.time() - t0
            ocr_mode = True

        # ---- 3. 输出 ----
        title = pdf_path.stem

        if to_markdown:
            t0 = time.time()
            md = pages_to_markdown(pages, source=str(pdf_path), ocr_mode=ocr_mode)
            result.markdown = md
            md_path = out_dir / f"{pdf_path.stem}.md"
            md_path.write_text(md, encoding="utf-8")
            result.md_path = str(md_path)
            result.timings["markdown"] = time.time() - t0

        if to_docx:
            t0 = time.time()
            docx_path = out_dir / f"{pdf_path.stem}.docx"
            pages_to_docx(pages, docx_path, title=title, ocr_mode=ocr_mode)
            result.docx_path = str(docx_path)
            result.timings["docx"] = time.time() - t0

        return result
