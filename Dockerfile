# Use official slim Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgl1-mesa-glx \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install modern pip, setuptools, and wheel to support pyproject.toml-based builds
RUN pip install --upgrade pip setuptools wheel

# Copy requirements.txt separately for Docker layer caching
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Install Gunicorn if not already in requirements.txt
RUN pip install gunicorn

# Copy rest of the app
COPY . .

# Expose port
EXPOSE $PORT

# Start the app with gunicorn (adjust module path if needed)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
