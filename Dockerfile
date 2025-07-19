# Base image with Python 3.12
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
    gcc \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 🔧 Install essential Python build tools early
RUN pip install --upgrade pip setuptools wheel build

# Copy requirements
COPY requirements.txt .

# 🔁 Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port Flask will run on
EXPOSE $PORT

# Run app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
