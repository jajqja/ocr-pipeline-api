# Deployment Guide

This guide covers deploying the OCR Pipeline API to production.

## Local Development

### Setup

```bash
# 1. Enter project directory
cd /home/jaqja/New_AI/ocr-pipeline-api

# 2. Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -e .
# or
pip install -r requirements.txt

# 4. Copy environment
cp .env.example .env
```

### Running

```bash
# Development (with auto-reload)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production (with workers)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn (recommended for production)
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

## Docker Deployment

### Single Container

```bash
# Build
docker build -t ocr-pipeline-api:latest .

# Run with GPU
docker run \
  --gpus all \
  -p 8000:8000 \
  -e DEVICE=cuda \
  -e BATCH_SIZE_DETECTION=4 \
  -e BATCH_SIZE_RECOGNITION=8 \
  -v $(pwd)/model_path:/app/model_path \
  ocr-pipeline-api:latest

# Run with CPU only
docker run \
  -p 8000:8000 \
  -e DEVICE=cpu \
  ocr-pipeline-api:latest
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f ocr-api

# Stop services
docker-compose down

# Rebuild image
docker-compose build --no-cache
```

## Kubernetes Deployment

### Create ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ocr-api-config
data:
  DEVICE: cuda
  BATCH_SIZE_DETECTION: "4"
  BATCH_SIZE_RECOGNITION: "8"
  MAX_TOKENS: "500"
  PORT: "8000"
```

### Create Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ocr-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ocr-api
  template:
    metadata:
      labels:
        app: ocr-api
    spec:
      containers:
      - name: ocr-api
        image: ocr-pipeline-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: ocr-api-config
        resources:
          requests:
            memory: "8Gi"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 20
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/model_path
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: ocr-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ocr-api-service
spec:
  selector:
    app: ocr-api
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### Apply to Kubernetes

```bash
# Create namespace
kubectl create namespace ocr-api

# Apply configurations
kubectl apply -f configmap.yaml -n ocr-api
kubectl apply -f deployment.yaml -n ocr-api

# Check status
kubectl get pods -n ocr-api
kubectl logs deployment/ocr-api -n ocr-api
```

## Reverse Proxy (Nginx)

### Configuration

```nginx
upstream ocr_api {
    server localhost:8000;
    keepalive 64;
}

server {
    listen 80;
    server_name api.example.com;
    client_max_body_size 100M;

    location / {
        proxy_pass http://ocr_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_request_buffering off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }

    # SSL (with Let's Encrypt)
    listen 443 ssl http2;
    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;
}
```

## Performance Optimization

### Batch Size Tuning

```bash
# For high throughput (trade-off latency)
BATCH_SIZE_DETECTION=16
BATCH_SIZE_RECOGNITION=32

# For low latency (trade-off throughput)
BATCH_SIZE_DETECTION=1
BATCH_SIZE_RECOGNITION=1

# Balanced (recommended)
BATCH_SIZE_DETECTION=4
BATCH_SIZE_RECOGNITION=8
```

### GPU Memory Optimization

```bash
# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# Monitor while running
watch -n 1 nvidia-smi

# Reduce batch size if OOM
docker run --gpus all -e BATCH_SIZE_DETECTION=2 -e BATCH_SIZE_RECOGNITION=4 ...
```

### Load Balancing

For multiple instances:

```yaml
# docker-compose.yml
services:
  ocr-api-1:
    image: ocr-pipeline-api:latest
    environment:
      - DEVICE=cuda:0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  ocr-api-2:
    image: ocr-pipeline-api:latest
    environment:
      - DEVICE=cuda:1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]

  nginx:
    image: nginx:latest
    ports:
      - "8000:8000"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - ocr-api-1
      - ocr-api-2
```

## Monitoring

### Prometheus Metrics

Add to FastAPI app for metric collection:

```python
from prometheus_client import Counter, Histogram
from fastapi_prometheus_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### Health Checks

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Check with verbose output
curl -v http://localhost:8000/api/v1/health

# Monitor continuously
watch -n 5 'curl -s http://localhost:8000/api/v1/health | jq .'
```

### Logging

Configure centralized logging:

```python
# In app/core/logger.py
import logging.handlers

handler = logging.handlers.SysLogHandler(address=('localhost', 514))
logger.addHandler(handler)
```

## Backup & Recovery

### Model Backup

```bash
# Backup models
tar -czf models_backup.tar.gz model_path/

# Restore models
tar -xzf models_backup.tar.gz
```

### Database Backup (if using)

```bash
# PostgreSQL
pg_dump -U user dbname > backup.sql

# MongoDB
mongodump --uri="mongodb://localhost:27017/dbname" --out backup/
```

## Security Considerations

### API Key Protection

Uncomment and configure authentication in `app/dependencies/auth.py`

### HTTPS/TLS

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Run with SSL
python -m uvicorn app.main:app \
  --ssl-keyfile=key.pem \
  --ssl-certfile=cert.pem \
  --host 0.0.0.0 --port 8000
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/parse")
@limiter.limit("5/minute")
async def parse_document(request, _=Depends()):
    # Implementation
    pass
```

## Troubleshooting

### Check logs
```bash
docker logs container_id
docker-compose logs ocr-api
kubectl logs deployment/ocr-api -n ocr-api
```

### Test connectivity
```bash
docker exec container_id curl http://localhost:8000/api/v1/health
```

### GPU issues
```bash
# Check CUDA availability
docker run --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
```

## Scaling

### Horizontal Scaling

Add more API instances with load balancer (nginx, HAProxy)

### Vertical Scaling

- Increase GPU allocation
- Increase batch sizes
- Use larger GPU instances

### Caching

Add Redis for response caching:

```python
import redis
cache = redis.Redis(host='localhost', port=6379)
```

## Cost Optimization

- Use spot instances in cloud providers
- Batch process during off-peak hours
- Cache detection/recognition results
- Compress models
- Use quantization for models

## Support & Resources

- Surya OCR: https://github.com/VikParuchuri/surya
- FastAPI: https://fastapi.tiangolo.com/
- Docker: https://docs.docker.com/
- Kubernetes: https://kubernetes.io/docs/
