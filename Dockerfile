FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (to leverage Docker cache)
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    && pip install setuptools wheel \
    pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn

# Copy the rest of the project files
COPY . .

EXPOSE 10000

# Use JSON syntax for CMD (safer with signals)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
