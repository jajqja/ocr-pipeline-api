# API Testing Guide

This guide shows how to test the OCR Pipeline API using different methods.

## Prerequisites

Make sure the API is running:
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Using curl

### 1. Health Check

```bash
curl http://localhost:8000/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "gpu_available": true,
  "device": "cuda"
}
```

### 2. Text Detection

```bash
# Prepare image
IMAGE_BASE64=$(base64 -w0 path/to/image.png)

# Call API
curl -X POST http://localhost:8000/api/v1/detection \
  -H "Content-Type: application/json" \
  -d "{\"image_data\": \"$IMAGE_BASE64\"}"
```

### 3. Text Recognition

```bash
# Prepare image
IMAGE_BASE64=$(base64 -w0 path/to/image.png)

# Call API
curl -X POST http://localhost:8000/api/v1/recognition \
  -H "Content-Type: application/json" \
  -d "{
    \"image_data\": \"$IMAGE_BASE64\",
    \"task_name\": \"ocr_with_boxes\",
    \"max_tokens\": 500,
    \"math_mode\": true
  }"
```

### 4. Full OCR Pipeline

```bash
# Prepare image
IMAGE_BASE64=$(base64 -w0 path/to/image.png)

# Call API
curl -X POST http://localhost:8000/api/v1/parse \
  -H "Content-Type: application/json" \
  -d "{
    \"image_data\": \"$IMAGE_BASE64\",
    \"task_name\": \"ocr_with_boxes\"
  }"
```

## Using Python Requests

### Basic Example

```python
import requests
import base64

# Prepare image
with open("path/to/image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Health check
response = requests.get("http://localhost:8000/api/v1/health")
print(response.json())

# Full OCR
response = requests.post(
    "http://localhost:8000/api/v1/parse",
    json={"image_data": image_data}
)
result = response.json()
print(result["full_text"])
```

### Detection Only

```python
response = requests.post(
    "http://localhost:8000/api/v1/detection",
    json={"image_data": image_data}
)
detections = response.json()
for det in detections["detections"]:
    print(f"Found text region: {det['bbox']} (confidence: {det['confidence']})")
```

### Recognition from Bboxes

```python
# First, run detection
det_response = requests.post(
    "http://localhost:8000/api/v1/detection",
    json={"image_data": image_data}
)
detections = det_response.json()["detections"]

# Extract bboxes
bboxes = [
    [int(d["bbox"]["x1"]), int(d["bbox"]["y1"]), 
     int(d["bbox"]["x2"]), int(d["bbox"]["y2"])]
    for d in detections
]

# Recognize from bboxes
rec_response = requests.post(
    "http://localhost:8000/api/v1/recognition/with-bboxes",
    json={"image_data": image_data, "bboxes": bboxes}
)
result = rec_response.json()
print(result["full_text"])
```

## Using JavaScript/Node.js

### Fetch API

```javascript
async function testOCR(imageFile) {
    // Convert file to base64
    const reader = new FileReader();
    reader.onload = async (e) => {
        const imageData = e.target.result.split(',')[1];
        
        // Call API
        const response = await fetch('http://localhost:8000/api/v1/parse', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_data: imageData,
                task_name: 'ocr_with_boxes'
            })
        });
        
        const result = await response.json();
        console.log('Recognized text:', result.full_text);
    };
    reader.readAsDataURL(imageFile);
}

// Usage
const fileInput = document.getElementById('imageInput');
fileInput.addEventListener('change', (e) => testOCR(e.target.files[0]));
```

### axios

```javascript
const axios = require('axios');
const fs = require('fs');

async function testOCR(imagePath) {
    // Read and encode image
    const imageData = fs.readFileSync(imagePath, 'base64');
    
    // Call API
    const response = await axios.post('http://localhost:8000/api/v1/parse', {
        image_data: imageData,
        task_name: 'ocr_with_boxes'
    });
    
    console.log('Recognized text:', response.data.full_text);
}

testOCR('path/to/image.png');
```

## Performance Testing

### Batch Processing

```python
import requests
import base64
import time

# Prepare multiple images
images = []
for i in range(5):
    with open(f"image_{i}.png", "rb") as f:
        images.append(base64.b64encode(f.read()).decode())

# Process sequentially
start = time.time()
results = []
for img in images:
    response = requests.post(
        "http://localhost:8000/api/v1/parse",
        json={"image_data": img}
    )
    results.append(response.json())
elapsed = time.time() - start

print(f"Processed {len(images)} images in {elapsed:.2f}s")
print(f"Average: {elapsed/len(images):.2f}s per image")
```

### Load Testing with Apache Bench

```bash
# Create test script
cat > test.json << 'EOF'
{
    "image_data": "BASE64_IMAGE_DATA_HERE",
    "task_name": "ocr_with_boxes"
}
EOF

# Run load test
ab -n 100 -c 10 -p test.json -T application/json \
    http://localhost:8000/api/v1/parse
```

### Load Testing with wrk

```bash
# Install wrk
# https://github.com/wg/wrk

# Create script
cat > load_test.lua << 'EOF'
request = function()
    local image_data = io.open("image.b64"):read("*a")
    body = string.format('{"image_data": "%s"}', image_data)
    wrk.headers["Content-Type"] = "application/json"
    return wrk.format(nil, body)
end
EOF

# Run test
wrk -t4 -c100 -d30s --script load_test.lua \
    http://localhost:8000/api/v1/parse
```

## Error Handling

### Invalid Image

```python
response = requests.post(
    "http://localhost:8000/api/v1/parse",
    json={"image_data": "invalid_base64"}
)
# Returns 400 Bad Request
print(response.status_code, response.json())
```

### Server Error

```python
try:
    response = requests.post(
        "http://localhost:8000/api/v1/parse",
        json={"image_data": image_data}
    )
    response.raise_for_status()
    result = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e.response.status_code}")
    print(f"Details: {e.response.json()}")
except Exception as e:
    print(f"Error: {e}")
```

## Integration Testing

### Test All Endpoints

```python
import requests
import base64

BASE_URL = "http://localhost:8000/api/v1"

def test_api():
    # Prepare image
    with open("test_image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Test health
    print("Testing health...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    print("✓ Health check passed")
    
    # Test detection
    print("Testing detection...")
    response = requests.post(
        f"{BASE_URL}/detection",
        json={"image_data": image_data}
    )
    assert response.status_code == 200
    detections = response.json()["detections"]
    print(f"✓ Detection passed ({len(detections)} regions found)")
    
    # Test recognition
    print("Testing recognition...")
    response = requests.post(
        f"{BASE_URL}/recognition",
        json={"image_data": image_data}
    )
    assert response.status_code == 200
    text_lines = response.json()["text_lines"]
    print(f"✓ Recognition passed ({len(text_lines)} lines found)")
    
    # Test parser
    print("Testing full pipeline...")
    response = requests.post(
        f"{BASE_URL}/parse",
        json={"image_data": image_data}
    )
    assert response.status_code == 200
    result = response.json()
    print(f"✓ Full pipeline passed")
    print(f"  - Detections: {len(result['detections'])}")
    print(f"  - Text lines: {len(result['text_lines'])}")
    print(f"  - Processing time: {result['processing_time']:.2f}s")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_api()
```

## Debugging

### Verbose Output

```python
import requests
import logging

# Enable HTTP logging
http_client = logging.getLogger("urllib3")
http_client.setLevel(logging.DEBUG)

response = requests.post(
    "http://localhost:8000/api/v1/parse",
    json={"image_data": image_data}
)
```

### Check Response Details

```python
response = requests.post(...)
print("Status Code:", response.status_code)
print("Headers:", response.headers)
print("Body:", response.text)
print("JSON:", response.json())
```

## Documentation Links

- Interactive Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc
- OpenAPI JSON: http://localhost:8000/api/openapi.json
