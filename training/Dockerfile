FROM python:3.11-slim

WORKDIR /training

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    ultralytics \
    opencv-python==4.10.0.84 \
    numpy==2.0.1 && \
    pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

ENV PYTHONPATH=/training

CMD ["sh"]