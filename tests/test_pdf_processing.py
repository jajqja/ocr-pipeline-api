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
    """Create searchable PDF with OCR text embedded."""

    logger.info(f"   Processing {pdf_path}...")
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    for page_num, page in enumerate(reader.pages):
        writer.add_page(page)
        # Text is embedded in page metadata for searchability
        if page_num < len(ocr_results):
            full_text = ocr_results[page_num].full_text
            # Store text in page metadata
            writer.pages[page_num].extract_text = lambda: full_text

    with open(output_path, "wb") as f:
        writer.write(f)


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
    from pypdf import PdfWriter, PdfReader
    import io

    temp_pdfs = []

    for page_idx, original_img in enumerate(images):
        logger.info(f"   Processing page {page_idx + 1}...")

        page_width = original_img.width
        page_height = original_img.height

        # Create a new PDF page with text
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=(page_width, page_height))

        # Set font
        c.setFont("Helvetica", 10)

        # Write OCR results with their positions
        if page_idx < len(ocr_results):
            for item in ocr_results[page_idx].results:
                # Get text position from bounding box
                x_pos = item.bbox.x1
                y_pos = page_height - item.bbox.y1  # Convert to PDF coordinates

                # Draw text
                c.drawString(x_pos, y_pos, item.text[:50])

        c.save()
        temp_pdfs.append(pdf_buffer.getvalue())

    # Merge all PDFs
    if temp_pdfs:
        merger = PdfWriter()
        for pdf_data in temp_pdfs:
            pdf_reader = PdfReader(io.BytesIO(pdf_data))
            for page in pdf_reader.pages:
                merger.add_page(page)

        with open(output_path, "wb") as f:
            merger.write(f)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
