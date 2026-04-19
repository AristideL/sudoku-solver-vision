# Sudoku Solver - Solution 2 (YOLO + Classical CV + Template Matching)
# Application layer: Lightweight Sudoku solver using classical computer vision

FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV, display, and GUI
# - libgl1: OpenCV display support
# - libglib2.0-0: OpenCV core libraries
# - libgomp1: OpenMP support (numpy/torch)
# - python3-tk, tk: Tkinter GUI framework
# - xdotool: Required for pyautogui keyboard/mouse automation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    python3-tk \
    tk \
    xdotool \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and required resources
COPY src/ src/
COPY models/ models/
COPY numbers/ numbers/

# Set Python path for proper imports
ENV PYTHONPATH=/app/src

# Health check: ensure main module is importable
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src" || exit 1

# Entry point: launch the application
CMD ["python", "-m", "src"]
