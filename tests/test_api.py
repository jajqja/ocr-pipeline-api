import base64
import logging
from io import BytesIO
from typing import List
import httpx
from PIL import Image, ImageDraw

# Cấu hình log để theo dõi tiến trình chạy test
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8000/api/v1"  # Thay đổi URL và Port cho đúng với ứng dụng của bạn


def generate_dummy_base64_image(text: str = "Test OCR") -> str:
    """Tạo một ảnh giả lập chứa chữ và chuyển sang định dạng chuỗi Base64"""
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    # Vẽ chữ cơ bản lên ảnh để mô hình có cái nhận diện
    d.text((10, 40), text, fill=(0, 0, 0))
    
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def test_text_detection(client: httpx.Client, sample_images: List[str]):
    """Test Endpoint: POST /api/v1/detection"""
    logger.info("=== Testing Text Detection API ===")
    url = f"{BASE_URL}/detection"
    
    payload = {
        "images_data": sample_images,
        "batch_size": 2,
        "padding": 3,
        "detector_text_threshold": 0.25,
        "detector_blank_threshold": 0.15
    }
    
    response = client.post(url, json=payload, timeout=30.0)
    logger.info(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        res_data = response.json()
        logger.info(f"Success: {res_data['success']}")
        logger.info(f"Processing Time: {res_data['processing_time']}s")
        logger.info(f"Message: {res_data['message']}")
        # In ra cấu hình bboxes của ảnh đầu tiên làm mẫu
        if res_data["results"]:
            det_count = len(res_data["results"][0]["detections"])
            logger.info(f"Image 0 found {det_count} text regions.")
    else:
        logger.error(f"Failed: {response.text}")


def test_text_recognition(client: httpx.Client, sample_images: List[str]):
    """Test Endpoint: POST /api/v1/recognition"""
    logger.info("=== Testing Text Recognition API ===")
    url = f"{BASE_URL}/recognition"
    
    payload = {
        "images_data": sample_images,
        # Giả lập truyền bboxes thủ công dạng 3D [ [ [x1,y1,x2,y2] ], [] ]
        # Ảnh 1 chỉ định sẵn 1 khung hình, ảnh 2 để trống mảng để tự quét toàn bộ
        "bboxes": [
            [[0, 0, 300, 100]],
            []
        ],
        "task_name": "ocr_with_boxes",
        "batch_size": 2,
        "max_tokens": 500,
        "math_mode": True
    }
    
    response = client.post(url, json=payload, timeout=30.0)
    logger.info(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        res_data = response.json()
        logger.info(f"Success: {res_data['success']}")
        logger.info(f"Processing Time: {res_data['processing_time']}s")
        for img_res in res_data["results"]:
            logger.info(f"Image Index {img_res['image_index']} -> Full Text: {repr(img_res['full_text'])}")
    else:
        logger.error(f"Failed: {response.text}")


def test_full_pipeline_parser(client: httpx.Client, sample_images: List[str]):
    """Test Endpoint: POST /api/v1/parser (Đổi sang endpoint thực tế của bạn nếu khác)"""
    logger.info("=== Testing Full Pipeline Parser API ===")
    url = f"{BASE_URL}/parser"  
    
    payload = {
        "images_data": sample_images,
        "task_name": "ocr_with_boxes",
        "detect_batch_size": 2,
        "recognize_batch_size": 2,
        "max_tokens": 500,
        "math_mode": True,
        "padding": 2,
        "detector_text_threshold": 0.3,
        "detector_blank_threshold": 0.1
    }
    
    response = client.post(url, json=payload, timeout=60.0)  # Pipeline chạy 2 model nên để timeout dài hơn
    logger.info(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        res_data = response.json()
        logger.info(f"Success: {res_data['success']}")
        logger.info(f"Processing Time: {res_data['processing_time']}s")
        for img_res in res_data["results"]:
            logger.info(f"Image {img_res['image_index']} -> Extracted Text: {repr(img_res['full_text'])}")
            logger.info(f"Avg Det Conf: {img_res['avg_det_confidence']:.2f} | Avg Rec Conf: {img_res['avg_reg_confidence']:.2f}")
    else:
        logger.error(f"Failed: {response.text}")


if __name__ == "__main__":
    logger.info("Preparing dummy base64 images for batch testing...")
    # Tạo 2 ảnh base64 tượng trưng cho một Batch đầu vào gồm 2 ảnh
    image_1 = generate_dummy_base64_image("Surya OCR Batch Line 1")
    image_2 = generate_dummy_base64_image("Formula: f(x) = x^2 + 1")
    batch_images = [image_1, image_2]
    
    # Khởi tạo HTTP Client để giữ kết nối tốt hơn khi gọi liên tục
    with httpx.Client() as http_client:
        try:
            # 1. Chạy test API Phát Hiện Vùng Văn Bản
            test_text_detection(http_client, batch_images)
            print("\n")
            
            # 2. Chạy test API Nhận Diện Chữ
            test_text_recognition(http_client, batch_images)
            print("\n")
            
            # 3. Chạy test API Hợp Nhất Toàn Bộ Pipeline
            test_full_pipeline_parser(http_client, batch_images)
            
        except httpx.ConnectError:
            logger.error(f"Không thể kết nối tới Server tại địa chỉ {BASE_URL}. Bạn đã khởi chạy Server FastAPI chưa?")