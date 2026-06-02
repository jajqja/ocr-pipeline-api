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
from PIL import Image
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
from PIL import ImageDraw
from reportlab.lib.colors import transparent


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
        "--image_folder",
        type=str,
        default=None,
        help="Path to folder containing images (e.g., examples/)",
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
    input_pdf = args.pdf_path or "examples/sample.pdf"
    image_folder = args.image_folder
    output_dir = args.output_dir
    detect_batch_size = args.detect_batch_size
    recognize_batch_size = args.recognize_batch_size

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PDF OCR Pipeline Test")
    logger.info("=" * 70)

    # Step 1: Check if PDF exists or use image folder
    if image_folder and os.path.isdir(image_folder):
        logger.info(f"Loading images from folder: {image_folder}")

        # Collect all image files
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
            image_files.extend(sorted(Path(image_folder).glob(ext)))
            image_files.extend(sorted(Path(image_folder).glob(ext.upper())))

        if not image_files:
            logger.error(f"No images found in {image_folder}")
            return 1

        logger.info(f"Found {len(image_files)} images")

        # Convert images to base64
        base64_images = []
        for img_path in image_files:
            logger.info(f"  Loading: {img_path.name}")
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                base64_images.append(b64)

        page_images = [Image.open(str(p)).convert("RGB") for p in image_files]
        demo_mode = False

    elif os.path.exists(input_pdf):
        logger.info(f"Loading PDF: {input_pdf}")
        # Extract pages from PDF
        logger.info(f"Extracting pages from PDF: {input_pdf}")
        page_images = convert_from_path(input_pdf, dpi=200)
        logger.info(f"✓ Extracted {len(page_images)} pages")

        # Convert to base64
        base64_images = []
        for page_img in page_images:
            img_buffer = io.BytesIO()
            page_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            base64_images.append(b64)

        demo_mode = False
    else:
        logger.warning(f"PDF not found at {input_pdf}")
        logger.info("Using demo mode with sample images instead...")

        # Demo mode with sample images
        if os.path.exists("examples/001.png") and os.path.exists("examples/002.png"):
            base64_images = []
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
            detect_batch_size=detect_batch_size,
            recognize_batch_size=recognize_batch_size,
            padding=3,
            detector_text_threshold=0.7,
            detector_blank_threshold=0.45,
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
            _create_searchable_pdf(input_pdf, ocr_results, output_searchable)
            logger.info(f"   ✓ Saved: {output_searchable}")

            # Output 2: PDF with bounding boxes
            logger.info("\n2️⃣  Creating PDF with bounding boxes...")
            output_bbox = os.path.join(output_dir, "output_with_bbox.pdf")
            _create_bbox_pdf(page_images, ocr_results, output_bbox)
            logger.info(f"   ✓ Saved: {output_bbox}")

            # Output 3: PDF with OCR text on blank pages
            logger.info("\n3️⃣  Creating PDF with OCR text...")
            output_text = os.path.join(output_dir, "output_ocr_text.pdf")
            _create_ocr_text_pdf(page_images, ocr_results, output_text)
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

def _create_searchable_pdf(pdf_path: str, ocr_results, output_path: str) -> None:
    """
    Create searchable PDF with OCR text embedded.
    ocr_results: List[ImageParserResult] hoặc đối tượng chứa danh sách ImageParserResult
    """
    logger.info(f"Processing {pdf_path}...")
    
    # Trích xuất list kết quả từ response nếu ocr_results là DocumentParserBatchResponse
    if hasattr(ocr_results, 'results'):
        pages_ocr = ocr_results.results
    else:
        pages_ocr = ocr_results

    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    for page_num, page in enumerate(reader.pages):
        # 1. Thêm trang gốc vào writer trước
        writer.add_page(page)
        
        if page_num >= len(pages_ocr):
            continue
            
        page_ocr = pages_ocr[page_num]
        if not page_ocr.results:
            continue

        # Lấy kích thước trang PDF thực tế (đơn vị: points, 1 inch = 72 points)
        # Nếu OCR chạy bằng pixel, bạn cần scale tỷ lệ. Tạm thời lấy trực tiếp từ PDF:
        page_width = float(page.mediabox.width)
        page_height = float(page.mediabox.height)

        # 2. Tạo một file PDF tạm thời chứa text ẩn bằng ReportLab
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(page_width, page_height))
        
        # Cấu hình text ẩn (chữ trong suốt)
        can.setFillColor(transparent)
        
        for item in page_ocr.results:
            text = item.text
            # Giả định bbox có cấu trúc dạng list/tuple hoặc object có thuộc tính x_min, y_min...
            # Ví dụ: bbox = [x_min, y_min, x_max, y_max] 
            # Bạn hãy map lại cho đúng định dạng thực tế của class BBox của bạn nhé:
            try:
                x_min, y_min, x_max, y_max = item.bbox
            except TypeError:
                # Nếu bbox là object: item.bbox.x_min ...
                x_min = item.bbox.x_min
                y_min = item.bbox.y_min
                x_max = item.bbox.x_max
                y_max = item.bbox.y_max

            # --- ĐẢO TRỤC Y (Chuyển từ Top-Left sang Bottom-Left của PDF) ---
            # Lưu ý: Nếu OCR của bạn trả về pixel, bạn cần nhân thêm tỉ lệ (PDF_points / Image_pixels)
            pdf_x = x_min
            pdf_y = page_height - y_max  # Đảo trục y
            
            # Tính toán font_size tương đối dựa trên chiều cao bbox
            box_height = y_max - y_min
            font_size = max(int(box_height * 0.8), 1)
            
            # Vẽ text ẩn vào canvas
            can.setFont("Arial", font_size)
            can.drawString(pdf_x, pdf_y, text)

        can.save()
        packet.seek(0)
        
        # 3. Đọc lớp text ẩn vừa tạo và đè (merge) lên trang PDF gốc
        text_pdf = PdfReader(packet)
        if len(text_pdf.pages) > 0:
            # Lấy trang hiện tại vừa add ở bước 1 ra để merge
            writer.pages[page_num].merge_page(text_pdf.pages[0])

    # 4. Xuất file cuối cùng
    with open(output_path, "wb") as f:
        writer.write(f)
    logger.info(f"Saved searchable PDF to {output_path}")


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


def _create_ocr_text_pdf(images: list, ocr_results, output_path: str) -> None:
    """Create PDF with OCR results written on blank pages."""
    
    # Trích xuất list kết quả từ response nếu ocr_results là DocumentParserBatchResponse
    if hasattr(ocr_results, 'results'):
        pages_ocr = ocr_results.results
    else:
        pages_ocr = ocr_results

    writer = PdfWriter()

    for page_idx, original_img in enumerate(images):
        logger.info(f"   Processing page {page_idx + 1}...")

        # Lấy kích thước pixel của ảnh gốc
        page_width = original_img.width
        page_height = original_img.height

        # Tạo buffer riêng cho từng trang
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=(page_width, page_height))

        if page_idx < len(pages_ocr):
            page_ocr = pages_ocr[page_idx]
            
            for item in page_ocr.results:
                text = item.text
                if not text.strip():
                    continue
                
                # Giả định bbox chứa: x1 (trái), y1 (trên), x2 (phải), y2 (dưới)
                x_min = item.bbox.x1
                y_min = item.bbox.y1
                x_max = item.bbox.x2
                y_max = item.bbox.y2

                # 1. Tính toán Font size linh hoạt theo chiều cao thực tế của bounding box
                box_height = y_max - y_min
                font_size = max(int(box_height * 0.8), 1)  # Nhân 0.8 để chữ nằm gọn trong box

                # Sử dụng font mặc định "Helvetica" để tránh lỗi thiếu font Arial
                c.setFont("Arial", font_size)

                # 2. Chuyển đổi tọa độ hệ ảnh (Top-Left) sang hệ PDF (Bottom-Left)
                # Dùng y_max (cạnh đáy của dòng chữ trong OCR) để làm gốc tọa độ cho đường chân chữ (Baseline) trong PDF
                x_pos = x_min
                y_pos = page_height - y_max 

                # 3. Vẽ toàn bộ text (bỏ giới hạn [:50] để không bị mất chữ)
                c.drawString(x_pos, y_pos, text)

        c.save()
        pdf_buffer.seek(0)

        # Đọc trang vừa vẽ xong và add trực tiếp vào writer (Không cần mảng tạm temp_pdfs)
        pdf_reader = PdfReader(pdf_buffer)
        if len(pdf_reader.pages) > 0:
            writer.add_page(pdf_reader.pages[0])

    # Ghi toàn bộ các trang ra file output duy nhất
    with open(output_path, "wb") as f:
        writer.write(f)
        
    logger.info(f"Successfully saved OCR PDF to {output_path}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
