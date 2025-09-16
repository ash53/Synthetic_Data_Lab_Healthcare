# Dockerfile for Hugging Face Spaces (Docker SDK) – Streamlit demo
# Place this at the repo root (same level as demo/ and src/).

FROM python:3.11-slim

# Faster, quieter Python/pip and no cache
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

# Runtime dependency for scikit-learn (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends     libgomp1  && rm -rf /var/lib/apt/lists/*

# Work inside /app
WORKDIR /app

# Install Python deps first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (src/, demo/, data/, etc.)
COPY . .

# HF Spaces expects the app to listen on $PORT (default 7860 for Docker SDK)
ENV PORT=7860     STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Launch the Streamlit app
CMD ["streamlit", "run", "demo/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=${PORT}"]
