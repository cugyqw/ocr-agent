"""PDF 转换 —— 命令行入口。

自动判断 PDF 类型（电子版 / 扫描版），转换为 Markdown 和 Word。

用法：
    # 单个 PDF（输出 .md 和 .docx 到同目录）
    python pdf_convert.py 报告.pdf

    # 指定输出目录
    python pdf_convert.py 报告.pdf -o ./output

    # 只输出 Markdown
    python pdf_convert.py 报告.pdf --no-docx

    # 只检测类型，不转换
    python pdf_convert.py 报告.pdf --detect

    # 批量处理目录下所有 PDF
    python pdf_convert.py ./pdfs/

    # 限制 OCR 页数（大文档先试几页）
    python pdf_convert.py 扫描件.pdf --max-pages 5
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from pdfmd import PdfConverter, detect_pdf_type
from pdfmd.detector import describe_pdf_type


def collect_pdfs(targets: list[str]) -> list[Path]:
    """收集待处理的 PDF。支持目录、单个文件、多个文件。"""
    out: list[Path] = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            out.extend(sorted(p.glob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            out.append(p)
        else:
            print(f"跳过（非 PDF）: {t}", file=sys.stderr)
    # 去重并保持顺序
    seen = set()
    uniq = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def main():
    parser = argparse.ArgumentParser(description="PDF 转 Markdown / Word")
    parser.add_argument("input", nargs="+", help="PDF 文件（可多个）或包含 PDF 的目录")
    parser.add_argument("-o", "--output", default=None, help="输出目录")
    parser.add_argument("--no-docx", action="store_true", help="不输出 Word")
    parser.add_argument("--no-md", action="store_true", help="不输出 Markdown")
    parser.add_argument("--detect", action="store_true", help="只检测类型，不转换")
    parser.add_argument("--dpi", type=int, default=150,
                        help="扫描件渲染分辨率（默认150）。过高会超出模型 token 上限导致 OCR 失败")
    parser.add_argument("--max-pages", type=int, default=None, help="最多处理多少页")
    parser.add_argument("--no-formula", action="store_true", help="OCR 时跳过公式识别")
    args = parser.parse_args()

    pdfs = collect_pdfs(args.input)
    if not pdfs:
        print(f"未找到 PDF: {' '.join(args.input)}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(f"待处理 {len(pdfs)} 个 PDF")
    print("=" * 70)

    # ---- 只检测类型 ----
    if args.detect:
        for p in pdfs:
            pdf_type, detail = detect_pdf_type(p)
            print(f"\n{p.name}")
            print(f"  类型: {describe_pdf_type(pdf_type)}")
            print(f"  页数: {detail['total_pages']}")
            print(f"  抽样: {detail['chars_per_page']} 字符/页")
        return

    # ---- 转换 ----
    converter = PdfConverter(
        dpi=args.dpi,
        max_ocr_pages=args.max_pages,
        use_formula_prompt=not args.no_formula,
    )

    try:
        for i, pdf in enumerate(pdfs, 1):
            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(pdfs)}] {pdf.name}")
            print("=" * 70)

            result = converter.convert(
                pdf,
                out_dir=args.output,
                to_markdown=not args.no_md,
                to_docx=not args.no_docx,
            )
            print()
            print(result.summary())
    finally:
        converter.release()


if __name__ == "__main__":
    main()
