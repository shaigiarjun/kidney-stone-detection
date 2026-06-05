FROM python:3.12-slim

# Install system libraries required by opencv-python
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install deps, then force headless opencv (inference pulls in the full one)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python || true && \
    pip install --no-cache-dir opencv-python-headless

COPY . .

CMD python app.py
