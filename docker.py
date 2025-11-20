# Dockerfile - stable Playwright + Python build for Render
FROM python:3.11-slim

# Install system deps required by Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget gnupg \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 libxcomposite1 \
    libxrandr2 libxdamage1 libxfixes3 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    libgbm1 fonts-liberation fonts-dejavu-core unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Install Playwright browsers during image build (runs as root so no su issues)
RUN python -m playwright install --with-deps

# Copy app code
COPY . /app

# Expose port (Render sets $PORT)
ENV PORT=8080
# Use python -m uvicorn to avoid missing uvicorn script problems
CMD ["sh", "-c", "python -m uvicorn quiz:app --host 0.0.0.0 --port ${PORT}"]
