# Quick Start Guide

## Prerequisites

- Python 3.11+
- CUDA 11.8+ (for GPU)
- GPU with 8GB+ VRAM (recommended)

## Installation & Running

### Option 1: Local Development

```bash
# 1. Clone/enter project
cd /home/jaqja/New_AI/ocr-pipeline-api

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e .

# 4. Set environment
cp .env.example .env

# 5. Run server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API at: http://localhost:8000/api/docs

### Option 2: Docker

```bash
# 1. Build image
docker build -t ocr-pipeline-api .

# 2. Run container with GPU
docker run --gpus all -p 8000:8000 ocr-pipeline-api
```

Or use docker-compose:

```bash
docker-compose up -d
```

## Testing

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Full OCR (recommended starting point)

```python
import base64
import requests

# Prepare image
with open("document.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Call API
response = requests.post(
    "http://localhost:8000/api/v1/parse",
    json={"image_data": image_data}
)

result = response.json()
print(result["full_text"])
```

### Using provided test script

```bash
python examples/test_api.py
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/health` | GET | Check API health |
| `/api/v1/detection` | POST | Detect text regions |
| `/api/v1/recognition` | POST | Recognize text in image |
| `/api/v1/recognition/with-bboxes` | POST | Recognize from bboxes |
| `/api/v1/parse` | POST | Full OCR pipeline |

## Configuration

Edit `.env` to customize:

```env
DEVICE=cuda              # Use GPU
BATCH_SIZE_DETECTION=4   # Detection batch size
BATCH_SIZE_RECOGNITION=8 # Recognition batch size
PORT=8000               # API port
```

## Documentation

- Full docs: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc
- README: See README.md

## Troubleshooting

**GPU not detected?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of memory?**
- Reduce BATCH_SIZE in .env
- Use CPU: set DEVICE=cpu

**Slow performance?**
- Check GPU usage: `nvidia-smi`
- Increase batch size
- Use production mode with multiple workers

## Next Steps

1. Review the API documentation at `/api/docs`
2. Test with sample images in `examples/`
3. Integrate with your application
4. Deploy using Docker

For more information, see README.md
