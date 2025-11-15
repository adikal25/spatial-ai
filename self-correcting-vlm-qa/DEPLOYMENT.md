# Deployment Guide

This guide shows how to deploy your Self-Correcting VLM QA system to the cloud.

## Architecture

The system has two parts:
1. **FastAPI Backend** - Handles image processing, depth estimation, Claude API calls
2. **Streamlit Frontend** - User interface

They need to be deployed separately.

---

## Option 1: Render (Recommended - Free Tier Available)

### Step 1: Deploy FastAPI Backend to Render

1. **Push your code to GitHub** (if not already)

2. **Go to [Render](https://render.com)** and sign up

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `vlm-qa-api`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
     - **Instance Type**: Choose based on needs (Free tier available, but limited)

4. **Add Environment Variables** in Render dashboard:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   DEPTH_MODEL=depth_anything_v2
   ENABLE_GPU=false
   ```

5. **Deploy** - Render will give you a URL like: `https://vlm-qa-api.onrender.com`

6. **Note**: Free tier spins down after inactivity. First request may be slow (30s).

### Step 2: Deploy Streamlit Frontend to Streamlit Cloud

1. **Go to [Streamlit Cloud](https://share.streamlit.io)**

2. **Deploy New App**
   - Connect your GitHub repository
   - **Main file path**: `demo/app.py`
   - **Python version**: `3.11`

3. **Add Environment Variable** (IMPORTANT):
   - In "Advanced settings" → "Secrets"
   - Add:
     ```toml
     API_URL = "https://vlm-qa-api.onrender.com"
     ```
   - Replace with your actual Render backend URL

4. **Deploy**

5. **Access your app** at: `https://your-app.streamlit.app`

---

## Option 2: Railway (Better Performance, Paid)

### Step 1: Deploy Backend to Railway

1. **Go to [Railway](https://railway.app)** and sign up

2. **New Project** → **Deploy from GitHub repo**

3. **Add Service** → Select your repository

4. **Configure**:
   - **Root Directory**: `/` (root of repo)
   - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`

5. **Add Environment Variables**:
   ```
   ANTHROPIC_API_KEY=your_key_here
   DEPTH_MODEL=depth_anything_v2
   ENABLE_GPU=false
   PORT=8000
   ```

6. **Generate Domain** - Railway will give you: `https://your-app.up.railway.app`

### Step 2: Deploy Frontend to Streamlit Cloud

Same as Option 1, but use your Railway URL:
```toml
API_URL = "https://your-app.up.railway.app"
```

---

## Option 3: All-in-One Docker Deployment

If you want to deploy both together:

### Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DEPTH_MODEL=depth_anything_v2
      - ENABLE_GPU=false
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    command: streamlit run demo/app.py --server.port 8501 --server.address 0.0.0.0
```

### Create `Dockerfile` (for API):

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Create `Dockerfile.streamlit`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY demo/ demo/

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "demo/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

Deploy to any Docker-compatible platform (Fly.io, DigitalOcean, AWS ECS, etc.)

---

## Important Notes

### 1. GPU Support
- Most cloud platforms don't offer free GPU
- For better performance with Depth Anything V2:
  - Use paid GPU instances (AWS EC2 with GPU, Paperspace, etc.)
  - Or use `DEPTH_MODEL=midas_v3_small` for CPU (faster but less accurate)

### 2. Model Download
- First request will be slow (downloading Depth Anything V2 model ~100MB)
- Consider pre-downloading in Docker build:
  ```dockerfile
  RUN python -c "from transformers import pipeline; pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf')"
  ```

### 3. Cold Starts
- Free tiers (Render, Railway free) spin down after inactivity
- First request after spin-down can take 30-60 seconds
- Consider paid tier for production use

### 4. API Keys
- **Never commit** API keys to git
- Always use environment variables
- For Streamlit Cloud, use the "Secrets" section in settings

---

## Testing Your Deployment

1. **Check API Health**:
   ```bash
   curl https://your-api-url.com/health
   ```

2. **Test in Streamlit**:
   - Upload an image
   - Ask a spatial question
   - Check the browser console for errors if it fails

3. **Common Issues**:
   - **Timeout**: Backend might be slow (depth model loading)
   - **401/403**: Check ANTHROPIC_API_KEY is set correctly
   - **500**: Check backend logs in Render/Railway dashboard

---

## Cost Estimates

| Platform | API Backend | Frontend | Total/Month |
|----------|-------------|----------|-------------|
| Render Free + Streamlit | $0 | $0 | **$0** (limited) |
| Render Starter + Streamlit | $7 | $0 | **$7** |
| Railway Hobby + Streamlit | $5-10 | $0 | **$5-10** |
| AWS/GCP (with GPU) | $50+ | $0 | **$50+** |

---

## Recommended Setup

**For Development/Demo**:
- Render Free + Streamlit Cloud = **Free**

**For Production**:
- Railway/Render Paid + Streamlit Cloud
- Or all on AWS/GCP with GPU for best performance

---

Need help? Check the logs in your platform's dashboard!
