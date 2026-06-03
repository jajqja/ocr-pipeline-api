#!/usr/bin/env python
"""
Test script for PDF processing with OCR pipeline.

This script demonstrates how to:
1. Extract pages from a PDF
2. Run OCR pipeline on each page
3. Generate three output PDFs:
   - Searchable PDF with OCR text embedded
   - PDF with bounding boxes drawn on pages
   - PDF with OCR text written on blank pages

Prerequisites:
- Install required packages: pip install -r requirements.txt
- Have sample PDF at: examples/sample.pdf (or use existing images)
- Models should be downloaded to: model_path/text_detection and model_path/text_recognition

Usage Examples:
    # Process a PDF file
    python tests/test_pdf_processing.py --pdf_path examples/sample.pdf

    # Process images from a folder
    python tests/test_pdf_processing.py --image_folder examples/

    # Process with custom batch sizes and output directory
    python tests/test_pdf_processing.py --pdf_path examples/sample.pdf --output_dir results/ --detect_batch_size 16 --recognize_batch_size 64

    # Process images folder with custom batch sizes
    python tests/test_pdf_processing.py --image_folder examples/ --detect_batch_size 16
"""

import logging
import sys
import os
import argparse
from pathlib import Path

from app.services.parser import ParserService
import base64
import io
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
from PIL import ImageDraw
from reportlab.lib.colors import transparent
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main test function."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="OCR Pipeline Test - Process PDF or image folder"
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        default=None,
        help="Path to input PDF file (e.g., examples/sample.pdf)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="examples/output",
        help="Output directory for results (default: examples/output)",
    )
    parser.add_argument(
        "--detect_batch_size",
        type=int,
        default=32,
        help="Batch size for detection (default: 32)",
    )
    parser.add_argument(
        "--recognize_batch_size",
        type=int,
        default=128,
        help="Batch size for recognition (default: 128)",
    )

    args = parser.parse_args()

    # Configuration
    input_pdf = args.pdf_path
    output_dir = args.output_dir
    detect_batch_size = args.detect_batch_size
    recognize_batch_size = args.recognize_batch_size

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PDF OCR Pipeline Test")
    logger.info("=" * 70)

    if os.path.exists(input_pdf):
        logger.info(f"Loading PDF: {input_pdf}")
        # Extract pages from PDF
        logger.info(f"Extracting pages from PDF: {input_pdf}")
        page_images = convert_from_path(input_pdf, dpi=96)
        highres_page_images = convert_from_path(input_pdf, dpi=192)
        logger.info(f"✓ Extracted {len(page_images)} pages")

        # Convert to base64
        base64_images = []
        base64_highres_images = []
        for page_img, highres_page_img in zip(page_images, highres_page_images):
            img_buffer = io.BytesIO()
            page_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            base64_images.append(b64)

            highres_buffer = io.BytesIO()
            highres_page_img.save(highres_buffer, format="PNG")
            highres_buffer.seek(0)
            highres_b64 = base64.b64encode(highres_buffer.getvalue()).decode("utf-8")
            base64_highres_images.append(highres_b64)

        demo_mode = False
    else:
        logger.warning(f"PDF not found at {input_pdf}")
        logger.info("Using demo mode with sample images instead...")

        # Demo mode with sample images
        if os.path.exists("examples/001.png") and os.path.exists("examples/002.png"):
            base64_images = []
            base64_highres_images = None
            for img_path in ["examples/001.png", "examples/002.png"]:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                    base64_images.append(b64)
            logger.info(f"Loaded {len(base64_images)} sample images")
        else:
            logger.error("No sample images found either. Exiting.")
            return 1

        demo_mode = True
        page_images = None

    # Step 2: Run OCR pipeline
    logger.info("\n" + "=" * 70)
    logger.info("Running OCR Pipeline...")
    logger.info("=" * 70)

    try:
        service = ParserService()
        ocr_results, processing_time = service.parse_document(
            base64_images,
            base64_highres_images,
            detect_batch_size=detect_batch_size,
            recognize_batch_size=recognize_batch_size,
            padding=1,
            detector_text_threshold=0.65,
            detector_blank_threshold=0.4,
        )

        logger.info(f"✓ OCR completed in {processing_time:.2f}s")
        logger.info(f"✓ Processed {len(ocr_results)} pages")

        # Log results summary
        for result in ocr_results:
            logger.info(f"\n📄 Page {result.image_index}:")
            logger.info(f"   Text items detected: {len(result.results)}")
            logger.info(f"   Total text length: {len(result.full_text)} characters")
            if result.results:
                logger.info(f"   Sample text: {result.full_text[:100]}...")

        # Step 3: Generate output PDFs (only if not demo mode)
        if not demo_mode and page_images:
            logger.info("\n" + "=" * 70)
            logger.info("Generating Output PDFs...")
            logger.info("=" * 70)

            # Output 1: Searchable PDF
            logger.info("\n1️⃣  Creating searchable PDF...")
            output_searchable = os.path.join(output_dir, "output_searchable.pdf")
            _create_searchable_pdf(
                input_pdf, ocr_results, output_searchable, page_images
            )
            logger.info(f"   ✓ Saved: {output_searchable}")

            # Output 2: PDF with bounding boxes
            logger.info("\n2️⃣  Creating PDF with bounding boxes...")
            output_bbox = os.path.join(output_dir, "output_with_bbox.pdf")
            _create_bbox_pdf(page_images, ocr_results, output_bbox)
            logger.info(f"   ✓ Saved: {output_bbox}")

            # Output 3: PDF with OCR text on blank pages
            logger.info("\n3️⃣  Creating PDF with OCR text...")
            output_text = os.path.join(output_dir, "output_ocr_text.txt")
            _create_ocr_text_txt(ocr_results, output_text)
            logger.info(f"   ✓ Saved: {output_text}")

            logger.info("\n" + "=" * 70)
            logger.info("✅ All output files generated successfully!")
            logger.info("=" * 70)
            logger.info("\nOutput files:")
            logger.info(f"  1. Searchable PDF:      {output_searchable}")
            logger.info(f"  2. BBox PDF:            {output_bbox}")
            logger.info(f"  3. OCR Text PDF:        {output_text}")
            logger.info("=" * 70)
        else:
            logger.info("\n✅ OCR pipeline executed successfully!")
            logger.info("In production, PDF outputs would be generated here.")

    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


def _create_searchable_pdf(
    pdf_path: str,
    ocr_results,
    output_path: str,
    images: list,
    font_path: str = "Arial.ttf",
) -> None:
    """
    Create searchable PDF with OCR text stretched perfectly to fill the entire bounding box.
    """
    logger.info(f"Processing perfect-fit searchable PDF for {pdf_path}...")

    try:
        pdfmetrics.registerFont(TTFont("Arial", font_path))
        font_name = "Arial"
    except Exception as e:
        logger.warning(f"Không thể tải font Arial, dùng Helvetica: {e}")
        font_name = "Helvetica"

    if hasattr(ocr_results, "results"):
        pages_ocr = ocr_results.results
    else:
        pages_ocr = ocr_results

    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    for page_num, page in enumerate(reader.pages):
        writer.add_page(page)

        if page_num >= len(pages_ocr) or page_num >= len(images):
            continue

        page_ocr = pages_ocr[page_num]
        original_img = images[page_num]

        if not page_ocr.results:
            continue

        pdf_w = float(page.mediabox.width)
        pdf_h = float(page.mediabox.height)
        img_w = float(original_img.width)
        img_h = float(original_img.height)

        scale_x = pdf_w / img_w
        scale_y = pdf_h / img_h

        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(pdf_w, pdf_h))
        can.setFillColor(transparent)  # Giữ chữ ẩn

        for item in page_ocr.results:
            text = item.text
            if not text.strip():
                continue

            x1 = item.bbox.x1 * scale_x
            y1 = item.bbox.y1 * scale_y
            x2 = item.bbox.x2 * scale_x
            y2 = item.bbox.y2 * scale_y

            box_width = x2 - x1
            box_height = y2 - y1

            # 1. Định dạng kích thước font theo độ cao box (khoảng 85% để bao quát tốt hơn)
            font_size = max(box_height * 0.85, 1)

            # 2. Tạo một TextObject để kiểm soát nâng cao
            text_obj = can.beginText()
            text_obj.setFont(font_name, font_size)

            # Tính toán chiều rộng tự nhiên của chuỗi chữ này khi chưa co giãn
            text_width = can.stringWidth(text, font_name, font_size)

            # 3. ÉP CO GIÃN CHIỀU RỘNG (Tỷ lệ phần trăm ngang)
            if text_width > 0:
                # Tính tỷ lệ phần trăm cần kéo dãn (Ví dụ: 100 nghĩa là giữ nguyên, 150 là dãn rộng ra 1.5 lần)
                horizontal_scale = (box_width / text_width) * 100
                text_obj.setHorizScale(horizontal_scale)

            # Đặt tọa độ góc Bottom-Left cho chữ (bù trừ Baseline khoảng 15%)
            pdf_x = x1
            pdf_y = pdf_h - y2 + (box_height * 0.15)

            text_obj.setTextOrigin(pdf_x, pdf_y)
            text_obj.textLine(text)

            # Vẽ TextObject này lên canvas
            can.drawText(text_obj)

        can.save()
        packet.seek(0)

        text_pdf = PdfReader(packet)
        if len(text_pdf.pages) > 0:
            writer.pages[page_num].merge_page(text_pdf.pages[0])

    with open(output_path, "wb") as f:
        writer.write(f)

    logger.info(f"Successfully saved stretched searchable PDF to {output_path}")


def _create_bbox_pdf(images: list, ocr_results, output_path: str) -> None:
    """Create PDF with bounding boxes drawn from text lines."""

    temp_images = []

    for page_idx, original_img in enumerate(images):
        logger.info(f"   Processing page {page_idx + 1}...")

        # Create a copy to draw on
        img_with_boxes = original_img.copy()
        draw = ImageDraw.Draw(img_with_boxes)

        # Draw bounding boxes
        if page_idx < len(ocr_results):
            for item in ocr_results[page_idx].results:
                x1 = item.bbox.x1
                y1 = item.bbox.y1
                x2 = item.bbox.x2
                y2 = item.bbox.y2

                # Draw box outline in red
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                # Draw text label above the box
                draw.text((x1, y1 - 15), item.text[:30], fill="red")

        temp_images.append(img_with_boxes)

    # Save all images as PDF
    if temp_images:
        temp_images[0].save(
            output_path,
            save_all=True,
            append_images=temp_images[1:] if len(temp_images) > 1 else [],
            format="PDF",
        )


def _create_ocr_text_txt(ocr_results, output_path: str) -> None:
    """
    Create a plain .txt file containing all OCR text results.
    Each page's content is separated by a page break marker.
    """
    logger.info("Extracting OCR text to plain text file...")

    # Trích xuất list kết quả từ response nếu ocr_results là DocumentParserBatchResponse
    if hasattr(ocr_results, "results"):
        pages_ocr = ocr_results.results
    else:
        pages_ocr = ocr_results

    # Mở file txt với định dạng mã hóa utf-8 để tránh lỗi font tiếng Việt
    with open(output_path, "w", encoding="utf-8") as f:
        for page_idx, page_ocr in enumerate(pages_ocr):
            f.write(f"--- PAGE {page_idx + 1} ---\n")

            # Cách 1: Sử dụng luôn thuộc tính `full_text` đã được nối sẵn bằng dấu xuống dòng
            if hasattr(page_ocr, "full_text") and page_ocr.full_text:
                f.write(page_ocr.full_text)
                f.write("\n\n")

            # Cách 2: Nếu không có full_text, ta tự nối từ danh sách các dòng chữ chi tiết
            elif page_ocr.results:
                for item in page_ocr.results:
                    if item.text and item.text.strip():
                        f.write(item.text + "\n")
                f.write("\n")

    logger.info(f"Successfully saved OCR text to {output_path}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
