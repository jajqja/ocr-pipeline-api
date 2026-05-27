# OCR Pipeline API

A high-performance FastAPI-based OCR pipeline for GPU-accelerated text detection and recognition using the Surya OCR library.

## Features

- **Text Detection**: Fast and accurate text region detection using Surya's detection model
- **Text Recognition**: State-of-the-art text recognition using Surya's foundation model
- **Full Pipeline**: Combined detection + recognition for end-to-end OCR
- **GPU Support**: Full CUDA/GPU acceleration support
- **Batch Processing**: Efficient batch processing for multiple images
- **RESTful API**: Clean and easy-to-use REST API endpoints
- **Interactive Docs**: Swagger UI and ReDoc documentation

## Requirements

- Python 3.11+
- CUDA 11.8+ (for GPU support)
- GPU with at least 8GB VRAM (recommended)

## Installation

### 1. Clone and Setup

```bash
cd /home/jaqja/New_AI/ocr-pipeline-api

# Create virtual environment (if not already done)
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r pyproject.toml
# or using uv if available
uv pip install -e .
```

### 2. Download Models

The API expects model files in the `model_path` directory. Surya will automatically download models on first run:

```bash
mkdir -p model_path/text_detection
mkdir -p model_path/text_recognition
```

### 3. Configuration

Copy the example environment file and update settings:

```bash
cp .env.example .env
```

Key environment variables:

- `DEVICE`: cuda, cpu, or mps (default: cuda)
- `BATCH_SIZE_DETECTION`: Batch size for text detection (default: 4)
- `BATCH_SIZE_RECOGNITION`: Batch size for text recognition (default: 8)
- `MAX_TOKENS`: Maximum tokens for text generation (default: 500)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for detections (default: 0.3)
- `PORT`: API port (default: 8000)

## Usage

### Start the API Server

```bash
# Development mode
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

### API Endpoints

#### 1. Health Check

```bash
GET /api/v1/health
```

Check API health and GPU availability.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "gpu_available": true,
  "device": "cuda"
}
```

#### 2. Text Detection

```bash
POST /api/v1/detection
```

Detect text regions in an image.

**Request:**
```json
{
  "images_data": [
    "base64_encoded_image_1...",
    "base64_encoded_image_2..."
  ],
  "batch_size": 2
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "image_index": 0,
      "detections": [
        {
          "bbox": {
            "x1": 10.0,
            "y1": 20.0,
            "x2": 100.0,
            "y2": 50.0
          },
          "polygon": {
            "points": [[10, 20], [100, 20], [100, 50], [10, 50]]
          },
          "confidence": 0.95
        }
      ]
    },
    {
      "image_index": 1,
      "detections": [
        {
          "bbox": {
            "x1": 120.0,
            "y1": 30.0,
            "x2": 200.0,
            "y2": 60.0
          },
          "polygon": {
            "points": [[120, 30], [200, 30], [200, 60], [120, 60]]
          },
          "confidence": 0.88
        }
      ]
    }
  ],
  "processing_time": 1.45,
  "message": "Processed 2 images. Total 2 text regions detected."
}
```

#### 3. Text Recognition

```bash
POST /api/v1/recognition
```

Recognize text in the full image.

**Request:**
```json
{
  "image_data": "base64_encoded_image",
  "task_name": "ocr_with_boxes",
  "batch_size": null,
  "max_tokens": 500,
  "math_mode": true
}
```

**Response:**
```json
{
  "success": true,
  "text_lines": [
    {
      "text": "Recognized text",
      "confidence": 0.92,
      "chars": [...],
      "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 50}
    }
  ],
  "full_text": "Recognized text",
  "processing_time": 2.45,
  "message": "Recognized 1 text lines"
}
```

#### 4. Recognition with Bounding Boxes

```bash
POST /api/v1/recognition/with-bboxes
```

Recognize text from specific bounding boxes.

**Request:**
```json
{
  "image_data": "base64_encoded_image",
  "bboxes": [[10, 20, 100, 50], [120, 30, 200, 60]],
  "task_name": "ocr_with_boxes",
  "batch_size": null,
  "max_tokens": 500
}
```

#### 5. Full OCR Pipeline

```bash
POST /api/v1/parse
```

Run complete OCR pipeline (detection + recognition).

**Request:**
```json
{
  "image_data": "base64_encoded_image",
  "task_name": "ocr_with_boxes",
  "detect_batch_size": null,
  "recognize_batch_size": null,
  "max_tokens": 500,
  "math_mode": true,
  "confidence_threshold": 0.3
}
```

**Response:**
```json
{
  "success": true,
  "detections": [...],
  "text_lines": [...],
  "full_text": "Recognized text from detected regions",
  "processing_time": 3.67,
  "message": "Detected 5 regions and recognized 3 lines"
}
```

## Example Usage

### Python Client

```python
import base64
import requests

# Encode image to base64
with open("document.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Send to API
response = requests.post(
    "http://localhost:8000/api/v1/parse",
    json={
        "image_data": image_data,
        "task_name": "ocr_with_boxes"
    }
)

result = response.json()
print(result["full_text"])
```

### cURL

```bash
# Prepare image
IMAGE_BASE64=$(base64 -w0 document.png)

# Call API
curl -X POST http://localhost:8000/api/v1/parse \
  -H "Content-Type: application/json" \
  -d "{\"image_data\": \"$IMAGE_BASE64\", \"task_name\": \"ocr_with_boxes\"}"
```

### Using test script

```bash
python examples/test_api.py
```

## Docker Deployment

### Build Image

```bash
docker build -t ocr-pipeline-api .
```

### Run Container

```bash
docker run --gpus all -p 8000:8000 \
  -e DEVICE=cuda \
  ocr-pipeline-api
```

## Project Structure

```
ocr-pipeline-api/
├── app/
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── router.py        # Main API router
│   │   └── v1/
│   │       ├── detection.py # Detection endpoints
│   │       ├── recognition.py # Recognition endpoints
│   │       ├── parser.py    # Parser endpoints
│   │       └── health.py    # Health check
│   ├── services/
│   │   ├── detection.py     # Detection service (Surya)
│   │   ├── recognition.py   # Recognition service (Surya)
│   │   └── parer.py         # Parser service
│   ├── schemas/
│   │   └── __init__.py      # Pydantic models
│   └── core/
│       ├── config.py        # Configuration
│       └── logger.py        # Logging setup
├── model_path/              # Model directory
│   ├── text_detection/
│   └── text_recognition/
├── examples/
│   └── test_api.py          # Example usage
├── pyproject.toml           # Dependencies
├── Dockerfile               # Docker configuration
├── .env.example             # Environment template
└── README.md                # This file
```

## Available Tasks

The recognition API supports multiple OCR tasks from Surya:

- `ocr_with_boxes`: Full OCR with bounding boxes
- `ocr_without_boxes`: OCR without detailed position info
- `block_without_boxes`: Document structure analysis
- `layout`: Page layout analysis

## Performance Tips

### Optimization

1. **Batch Size**: Increase batch size for throughput, decrease for latency
2. **Model Loading**: Models are cached in memory (singleton pattern)
3. **GPU Memory**: Monitor with `nvidia-smi` if using CUDA
4. **Image Size**: Larger images take more time and GPU memory

### Monitoring

```bash
# Monitor GPU usage
nvidia-smi -l 1  # Update every 1 second

# Monitor API logs
docker logs -f container_id
```

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check device
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Out of Memory

1. Reduce batch size in `.env`
2. Reduce image size
3. Use CPU for testing: `DEVICE=cpu`

### Slow Performance

1. Check if GPU is being used: `nvidia-smi`
2. Increase batch size
3. Use production mode (multiple workers)

## API Limits

- **Max file size**: 100MB (configurable)
- **Max tokens**: 500 (configurable)
- **Supported formats**: JPEG, PNG, GIF, BMP, WebP

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Create a feature branch
2. Make your changes
3. Add tests if applicable
4. Submit a pull request

## License

This project uses the Surya OCR library which is subject to its own license terms.

## References

- [Surya OCR Documentation](https://github.com/VikParuchuri/surya)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## Support

For issues and questions:

1. Check existing issues on GitHub
2. Review the documentation
3. Run the example scripts
4. Check application logs

## Version History

- **0.1.0** (2026-05-27): Initial release
  - Text detection API
  - Text recognition API
  - Full OCR pipeline
  - GPU support
  - Interactive documentation
