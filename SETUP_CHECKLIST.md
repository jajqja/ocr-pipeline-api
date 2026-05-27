# Setup Checklist

Use this checklist to ensure your OCR Pipeline API is properly set up.

## 📋 Prerequisites

- [ ] Python 3.11+ installed
- [ ] pip or conda available
- [ ] CUDA 11.8+ (optional, for GPU support)
- [ ] 8GB+ RAM (16GB+ recommended for GPU)
- [ ] Docker & Docker Compose (for containerization)

## 🔧 Installation

- [ ] Repository cloned or project folder created
- [ ] Virtual environment created (`.venv` folder)
- [ ] Virtual environment activated
- [ ] Dependencies installed via `pip install -e .` or `pip install -r requirements.txt`
- [ ] `.env` file created from `.env.example`
- [ ] Model directories created (`model_path/text_detection`, `model_path/text_recognition`)

## ⚙️ Configuration

- [ ] `.env` file reviewed and updated
- [ ] `DEVICE` set to appropriate value (cuda/cpu/mps)
- [ ] `BATCH_SIZE_DETECTION` configured
- [ ] `BATCH_SIZE_RECOGNITION` configured
- [ ] `MAX_TOKENS` set as needed
- [ ] `CONFIDENCE_THRESHOLD` configured
- [ ] `PORT` set if different from 8000

## 🚀 Startup

- [ ] Virtual environment activated
- [ ] Dependencies verified with `pip list | grep -i fastapi`
- [ ] Application starts without errors
  ```bash
  python -m uvicorn app.main:app --reload
  ```
- [ ] No import errors in logs
- [ ] Application listening on configured port

## ✅ Testing

### Local Testing
- [ ] Health endpoint responds: `curl http://localhost:8000/api/v1/health`
- [ ] API documentation accessible: `http://localhost:8000/api/docs`
- [ ] ReDoc accessible: `http://localhost:8000/api/redoc`

### Endpoint Testing
- [ ] Detection endpoint returns results
- [ ] Recognition endpoint returns results
- [ ] Parser endpoint returns combined results
- [ ] Error handling works (try invalid image)

### Test Script
- [ ] Example script runs: `python examples/test_api.py`
- [ ] All endpoints tested successfully
- [ ] Processing times reasonable

## 🐳 Docker Setup (Optional)

- [ ] Docker installed and running
- [ ] Docker Compose installed
- [ ] Dockerfile builds successfully: `docker build -t ocr-pipeline-api .`
- [ ] Docker image created without errors
- [ ] Container runs with: `docker run --gpus all -p 8000:8000 ocr-pipeline-api`
- [ ] Docker Compose starts with: `docker-compose up -d`
- [ ] Container health check passes

## 📊 GPU Verification (if using CUDA)

- [ ] NVIDIA drivers installed
- [ ] CUDA toolkit 11.8+ installed
- [ ] GPU detected: `nvidia-smi` shows device
- [ ] PyTorch recognizes GPU: 
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- [ ] API reports GPU available in health check
- [ ] GPU memory monitored during inference

## 📁 File Structure

Verify these files exist:

```
✓ app/
  ✓ main.py
  ✓ core/config.py
  ✓ core/logger.py
  ✓ api/router.py
  ✓ api/v1/health.py
  ✓ api/v1/detection.py
  ✓ api/v1/recognition.py
  ✓ api/v1/parser.py
  ✓ services/detection.py
  ✓ services/recognition.py
  ✓ services/parer.py
  ✓ schemas/__init__.py

✓ model_path/
  ✓ text_detection/ (will be populated by Surya)
  ✓ text_recognition/ (will be populated by Surya)

✓ examples/test_api.py

✓ Configuration files
  ✓ .env
  ✓ pyproject.toml
  ✓ requirements.txt

✓ Docker files
  ✓ Dockerfile
  ✓ docker-compose.yml

✓ Documentation
  ✓ README.md
  ✓ QUICKSTART.md
  ✓ DEPLOYMENT.md
  ✓ API_TESTING.md
  ✓ PROJECT_SUMMARY.md
```

## 🎯 Quick Verification

Run this to verify everything is working:

```bash
# 1. Check Python
python --version  # Should be 3.11+

# 2. Check environment
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# 3. Check FastAPI installation
python -c "import fastapi; print('FastAPI OK')"

# 4. Check Surya installation
python -c "from surya.detection import DetectionPredictor; print('Surya OK')"

# 5. Check PyTorch/GPU
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"

# 6. Start server
python -m uvicorn app.main:app --reload

# 7. In another terminal, test API
curl http://localhost:8000/api/v1/health
```

## 🚨 Troubleshooting

### GPU Not Detected
- [ ] Check NVIDIA drivers: `nvidia-smi`
- [ ] Check CUDA installation
- [ ] Reinstall PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Out of Memory
- [ ] Reduce `BATCH_SIZE_DETECTION`
- [ ] Reduce `BATCH_SIZE_RECOGNITION`
- [ ] Use CPU: set `DEVICE=cpu`
- [ ] Reduce input image size

### Import Errors
- [ ] Verify virtual environment is activated
- [ ] Reinstall dependencies: `pip install -e .`
- [ ] Check Python version: `python --version`
- [ ] Clear cache: `rm -rf __pycache__ .pytest_cache`

### Port Already in Use
- [ ] Change PORT in `.env`
- [ ] Or kill process on port 8000: `lsof -ti:8000 | xargs kill -9`

### Models Not Downloading
- [ ] Check internet connection
- [ ] Check disk space
- [ ] Set model path: `DETECTION_MODEL_PATH=model_path/text_detection`
- [ ] Manually create directories if needed

## 📊 Performance Baseline

After setup, note these baseline numbers:

- [ ] API startup time: _____ seconds
- [ ] Detection time (single image): _____ seconds
- [ ] Recognition time (single image): _____ seconds
- [ ] Full pipeline time: _____ seconds
- [ ] GPU memory usage (idle): _____ MB
- [ ] GPU memory usage (peak): _____ MB

## 🔄 Production Readiness

### Before Production Deployment

- [ ] `.env` configured for production
- [ ] Error handling tested
- [ ] Logging configured appropriately
- [ ] Rate limiting configured (if needed)
- [ ] Authentication configured (if needed)
- [ ] HTTPS/TLS set up
- [ ] Reverse proxy configured (nginx/Traefik)
- [ ] Monitoring configured
- [ ] Backups configured
- [ ] Load balancing configured (if multiple instances)
- [ ] Auto-scaling configured (if using Kubernetes)

### Performance Tuning

- [ ] Batch sizes optimized for your hardware
- [ ] GPU memory usage acceptable
- [ ] Inference times within SLA
- [ ] Throughput meets requirements
- [ ] Latency acceptable for use case

### Testing Complete

- [ ] Unit tests written and passing
- [ ] Integration tests written and passing
- [ ] Load tests completed
- [ ] End-to-end tests completed
- [ ] Error scenarios tested
- [ ] Edge cases tested

## ✨ Ready to Deploy

Once all checkboxes are checked, your OCR Pipeline API is ready for:

- ✓ Development and testing
- ✓ Staging deployment
- ✓ Production deployment
- ✓ Scaling and optimization

## 📞 Need Help?

1. **Check Documentation**
   - README.md - Main guide
   - QUICKSTART.md - Quick reference
   - API_TESTING.md - Testing examples

2. **Check Logs**
   ```bash
   docker logs container_name
   docker-compose logs ocr-api
   tail -f /var/log/ocr-api.log
   ```

3. **Test Endpoints**
   - Visit: http://localhost:8000/api/docs
   - Try endpoints in Swagger UI

4. **Review Examples**
   - examples/test_api.py

5. **Check Resources**
   - Surya: https://github.com/VikParuchuri/surya
   - FastAPI: https://fastapi.tiangolo.com/
   - PyTorch: https://pytorch.org/

---

**Checklist Version**: 1.0  
**Last Updated**: 2026-05-27
