"""Example script for using the OCR Pipeline API locally."""

import base64
import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
IMAGE_PATH = "examples/sample.png"  # Update with your image path


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_detection(image_data: str):
    """Test text detection endpoint."""
    print("Testing text detection...")
    payload = {"image_data": image_data, "batch_size": None}

    response = requests.post(f"{BASE_URL}/api/v1/detection", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Detections found: {len(result['detections'])}")
    print(f"Processing time: {result['processing_time']:.2f}s\n")

    return result


def test_recognition(image_data: str):
    """Test text recognition endpoint."""
    print("Testing text recognition...")
    payload = {
        "image_data": image_data,
        "task_name": "ocr_with_boxes",
        "batch_size": None,
        "max_tokens": 500,
        "math_mode": True,
    }

    response = requests.post(f"{BASE_URL}/api/v1/recognition", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Text lines found: {len(result['text_lines'])}")
    print(f"Full text:\n{result['full_text'][:500]}...\n")
    print(f"Processing time: {result['processing_time']:.2f}s\n")


def test_parser(image_data: str):
    """Test full OCR pipeline."""
    print("Testing full OCR pipeline...")
    payload = {
        "image_data": image_data,
        "task_name": "ocr_with_boxes",
        "detect_batch_size": None,
        "recognize_batch_size": None,
        "max_tokens": 500,
        "math_mode": True,
        "confidence_threshold": 0.3,
    }

    response = requests.post(f"{BASE_URL}/api/v1/parse", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Detections: {len(result['detections'])}")
    print(f"Text lines: {len(result['text_lines'])}")
    print(f"Full text:\n{result['full_text'][:500]}...\n")
    print(f"Processing time: {result['processing_time']:.2f}s\n")


def test_recognition_with_bboxes(image_data: str, bboxes: list):
    """Test recognition with specific bounding boxes."""
    print("Testing recognition with bounding boxes...")
    payload = {
        "image_data": image_data,
        "bboxes": bboxes,
        "task_name": "ocr_with_boxes",
        "batch_size": None,
        "max_tokens": 500,
    }

    response = requests.post(f"{BASE_URL}/api/v1/recognition/with-bboxes", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Text lines: {len(result['text_lines'])}")
    print(f"Full text:\n{result['full_text']}\n")


def main():
    """Run example tests."""
    # Check if image exists
    if not Path(IMAGE_PATH).exists():
        print(f"Image not found at {IMAGE_PATH}")
        print("Please provide an image file to test with.")
        return

    # Encode image
    print("Encoding image to base64...")
    image_data = encode_image_to_base64(IMAGE_PATH)
    print(f"Image size: {len(image_data)} characters\n")

    # Run tests
    test_health()

    detection_result = test_detection(image_data)

    test_recognition(image_data)

    test_parser(image_data)

    # Test recognition with bboxes if we have detections
    if detection_result["detections"]:
        bboxes = [
            [
                int(d["bbox"]["x1"]),
                int(d["bbox"]["y1"]),
                int(d["bbox"]["x2"]),
                int(d["bbox"]["y2"]),
            ]
            for d in detection_result["detections"][:3]  # First 3 boxes
        ]
        test_recognition_with_bboxes(image_data, bboxes)


if __name__ == "__main__":
    main()
