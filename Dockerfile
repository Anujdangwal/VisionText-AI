# Use official Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    gcc \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leverages Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn

# Copy project files after dependencies
COPY . .

# Expose port used by the app
EXPOSE $PORT

# Run the app with gunicorn using the environment-defined PORT
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
