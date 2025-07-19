FROM python:3.10

WORKDIR /app

# Install build tools (important for native packages like PyMuPDF, groq, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    cmake \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# ✅ Debug with verbose logs
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Copy rest of the app
COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
