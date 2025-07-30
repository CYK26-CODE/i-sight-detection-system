# 🚀 Deployment Guide

This guide covers deploying the i-sight Detection System to various platforms.

## 📋 Table of Contents

- [GitHub Repository Setup](#github-repository-setup)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [Heroku Deployment](#heroku-deployment)
- [Docker Deployment](#docker-deployment)
- [Local Production Setup](#local-production-setup)

## 🐙 GitHub Repository Setup

### 1. Initialize Git Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: i-sight Detection System"

# Add remote repository (replace with your GitHub repo URL)
git remote add origin https://github.com/yourusername/i-sight-detection-system.git

# Push to GitHub
git push -u origin main
```

### 2. Repository Structure

Ensure your repository has the following structure:

```
i-sight-detection-system/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── i_sight_flask_integrated.py
├── streamlit_app_simple.py
├── requirements_i_sight_flask.txt
├── requirements_streamlit.txt
├── README.md
├── LICENSE
├── .gitignore
└── DEPLOYMENT_GUIDE.md
```

### 3. GitHub Pages (Optional)

For documentation deployment:

```yaml
# .github/workflows/docs.yml
name: Deploy Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs
```

## ☁️ Streamlit Cloud Deployment

### 1. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set configuration:
   - **Main file path**: `streamlit_app_simple.py`
   - **Python version**: 3.9
   - **Requirements file**: `requirements_streamlit.txt`

### 2. Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### 3. Environment Variables

Set these in Streamlit Cloud:

```bash
# Optional: Set environment variables
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
```

### 4. Deploy

```bash
# Streamlit Cloud will automatically deploy when you push to main branch
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main
```

## 🏗️ Heroku Deployment

### 1. Create Heroku App

```bash
# Install Heroku CLI
# Create new app
heroku create your-app-name

# Set buildpack
heroku buildpacks:set heroku/python
```

### 2. Create Procfile

```procfile
# Procfile
web: gunicorn i_sight_flask_integrated:app --bind 0.0.0.0:$PORT
```

### 3. Update Requirements

Add to `requirements_i_sight_flask.txt`:

```txt
gunicorn==20.1.0
```

### 4. Deploy

```bash
# Deploy to Heroku
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Open app
heroku open
```

## 🐳 Docker Deployment

### 1. Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_i_sight_flask.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_i_sight_flask.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "i_sight_flask_integrated.py"]
```

### 2. Create Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  i-sight:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

### 3. Build and Run

```bash
# Build Docker image
docker build -t i-sight-detection .

# Run with Docker Compose
docker-compose up -d

# Or run directly
docker run -p 5000:5000 i-sight-detection
```

## 🖥️ Local Production Setup

### 1. Production Requirements

```bash
# Install production dependencies
pip install gunicorn
pip install -r requirements_i_sight_flask.txt
```

### 2. Gunicorn Configuration

```python
# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
```

### 3. Systemd Service (Linux)

```ini
# /etc/systemd/system/i-sight.service
[Unit]
Description=i-sight Detection System
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/i-sight-detection-system
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/gunicorn -c gunicorn_config.py i_sight_flask_integrated:app
Restart=always

[Install]
WantedBy=multi-user.target
```

### 4. Nginx Configuration

```nginx
# /etc/nginx/sites-available/i-sight
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /path/to/i-sight-detection-system/static;
    }
}
```

## 🔧 Environment Configuration

### 1. Environment Variables

Create `.env` file:

```bash
# .env
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key-here
CAMERA_INDEX=0
VOICE_ENABLED=true
YOLO_MODEL_PATH=./models/yolov5s.pt
TRAFFIC_SIGN_MODEL_PATH=./Traffic-Sign-Detection/data_svm.dat
```

### 2. Load Environment Variables

```python
# In your application
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY')
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', 0))
```

## 📊 Monitoring and Logging

### 1. Application Logging

```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
        handlers=[
            RotatingFileHandler('logs/i-sight.log', maxBytes=10240000, backupCount=10),
            logging.StreamHandler()
        ]
    )
```

### 2. Health Check Endpoint

```python
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.3.0'
    }
```

## 🔒 Security Considerations

### 1. HTTPS Setup

```bash
# Install SSL certificate
sudo certbot --nginx -d your-domain.com
```

### 2. Security Headers

```python
# Add security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response
```

## 🚀 Quick Deploy Commands

### Streamlit Cloud
```bash
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main
```

### Heroku
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### Docker
```bash
docker build -t i-sight-detection .
docker run -p 5000:5000 i-sight-detection
```

### Local Production
```bash
gunicorn -c gunicorn_config.py i_sight_flask_integrated:app
```

## 📞 Support

For deployment issues:

1. Check the logs: `docker logs <container_id>` or `heroku logs --tail`
2. Verify environment variables
3. Test locally first
4. Check GitHub Actions for build errors

---

**Happy Deploying! 🚀** 