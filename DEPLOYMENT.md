# Deployment Guide for Audio Deepfake Detection System

## Overview

This guide covers deploying the audio deepfake detection system in production environments.

## Pre-Deployment Checklist

### 1. Model Validation
```python
from src.training import Trainer
from src.models import HybridDeepfakeDetector

# Ensure model meets performance thresholds
trainer = Trainer(HybridDeepfakeDetector())
metrics = trainer.evaluate(X_test, y_test)

assert metrics['f1_score'] >= 0.85, "F1-score below threshold"
assert metrics['roc_auc'] >= 0.90, "ROC-AUC below threshold"
```

### 2. Configuration Management
```bash
# Create production config
cp config.yaml config_production.yaml

# Update paths and parameters
# - Set appropriate data paths
# - Configure logging levels
# - Set inference threshold
```

### 3. Version Control
```bash
git tag -a v1.0.0 -m "Production release"
git push origin v1.0.0
```

## Deployment Options

### Option 1: Docker Container (Recommended)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0"]
```

```bash
docker build -t deepfake-detector:1.0.0 .
docker run -p 8000:8000 deepfake-detector:1.0.0
```

### Option 2: Cloud Deployment (AWS)

```python
# Create Lambda function for serverless inference
import json
from src.inference import DeepfakeDetector

detector = DeepfakeDetector("s3://bucket/model.keras")

def lambda_handler(event, context):
    audio_url = event['audio_url']
    result = detector.detect_single(audio_url)
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### Option 3: REST API

```python
from fastapi import FastAPI, UploadFile, File
from src.inference import DeepfakeDetector

app = FastAPI()
detector = DeepfakeDetector("models/deepfake_detector.keras")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save uploaded file
    with open(f"temp/{file.filename}", "wb") as f:
        f.write(await file.read())
    
    # Run detection
    result = detector.detect_single(f"temp/{file.filename}")
    
    return result
```

## Monitoring and Logging

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram

detection_counter = Counter('detections_total', 'Total detections', ['result'])
inference_time = Histogram('inference_seconds', 'Inference time')

@inference_time.time()
def detect_with_monitoring(audio_path):
    result = detector.detect_single(audio_path)
    detection_counter.labels(result=result['is_deepfake']).inc()
    return result
```

### Structured Logging
```python
import logging
import json

logger = logging.getLogger(__name__)

def log_detection(result):
    logger.info(json.dumps({
        'timestamp': datetime.now().isoformat(),
        'audio_file': result['audio_file'],
        'is_deepfake': result['is_deepfake'],
        'probability': result['probability'],
        'confidence': result['confidence']
    }))
```

## Load Testing

```bash
# Apache Bench
ab -n 1000 -c 10 http://localhost:8000/health

# Locust
locust -f locustfile.py --headless -u 100 -r 10 -t 10m
```

## Database Integration

```python
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.orm import sessionmaker

engine = create_engine('postgresql://user:password@localhost/deepfake_db')

class DetectionResult(Base):
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True)
    audio_file = Column(String)
    is_deepfake = Column(Boolean)
    probability = Column(Float)
    timestamp = Column(DateTime)

# Save results
def save_detection(result):
    session = sessionmaker(bind=engine)()
    detection = DetectionResult(**result)
    session.add(detection)
    session.commit()
```

## Backup and Disaster Recovery

```bash
# Regular model backups
aws s3 sync models/ s3://backup-bucket/models/ --delete

# Database backups
pg_dump -Fc deepfake_db > backup_$(date +%Y%m%d).dump

# Restore from backup
pg_restore -d deepfake_db backup_20250114.dump
```

## Security Considerations

1. **Input Validation**
   - Verify audio file format and size
   - Check file extensions
   - Scan for malicious content

2. **Access Control**
   - Use API keys for authentication
   - Implement rate limiting
   - Log all API access

3. **Data Privacy**
   - Encrypt audio files at rest
   - Use HTTPS for data in transit
   - Implement data retention policies

4. **Model Protection**
   - Don't expose model weights
   - Use model serving frameworks (TF Serving, TorchServe)
   - Implement access controls

## Performance Optimization

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def detect_cached(audio_hash):
    # Only run inference if not cached
    return detector.detect_single(audio_path)
```

### Batch Processing
```python
def batch_detect_async(audio_files):
    # Queue audio files
    task = detect_batch.delay(audio_files)
    return {'task_id': task.id}

# Celery worker processes batch
@celery_app.task
def detect_batch(audio_files):
    results = detector.detect_batch(audio_files)
    save_results(results)
    return results
```

### Model Serving
```bash
# TensorFlow Serving
docker run -p 8501:8501 \
    -v $(pwd)/models:/models \
    -e MODEL_NAME=deepfake_detector \
    tensorflow/serving
```

## Monitoring Checklist

- [ ] Error rates < 0.1%
- [ ] Average inference time < 200ms
- [ ] P95 inference time < 500ms
- [ ] Model accuracy maintained > 90%
- [ ] Data pipeline running without issues
- [ ] Logs being collected and analyzed
- [ ] Alerts configured for anomalies
- [ ] Regular performance audits scheduled

## Troubleshooting in Production

### Issue: High Inference Latency
**Solution:**
- Enable GPU acceleration
- Use batch processing
- Implement caching
- Consider model optimization

### Issue: Memory Leaks
**Solution:**
- Profile with memory_profiler
- Clear cache periodically
- Monitor with monitoring tools
- Implement auto-restart policies

### Issue: Model Performance Degradation
**Solution:**
- Retrain with recent data
- Monitor data distribution changes
- Implement A/B testing
- Update threshold if needed

### Issue: API Overload
**Solution:**
- Implement rate limiting
- Use load balancing
- Auto-scale infrastructure
- Queue requests

## Maintenance Schedule

**Daily:**
- Monitor error logs
- Check system health
- Verify backups

**Weekly:**
- Review performance metrics
- Update security patches
- Test disaster recovery

**Monthly:**
- Audit access logs
- Review cost optimization
- Update documentation
- Test model updates

**Quarterly:**
- Performance audit
- Security assessment
- Model evaluation
- Capacity planning

## Rollback Procedures

```bash
# If new model causes issues
kubectl set image deployment/deepfake-api \
  deepfake-api=deepfake-detector:1.0.0

# Or with Docker Swarm
docker service update \
  --image deepfake-detector:1.0.0 deepfake-api
```

## Contact & Support

For deployment issues, contact the DevOps team or refer to:
- System documentation
- Runbooks in wiki
- Team chat channels
- Incident response procedures
