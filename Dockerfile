# Production container for the Nexus SMC Signal Engine.
# Keep the Docker dependency layer aligned with the live runtime dependency set.

FROM python:3.9-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build dependencies are retained in the builder only.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Install only production/live dependencies in the image.
# requirements-live.txt is intentionally copied explicitly because
# requirements.txt also contains development/test dependencies.
COPY requirements-live.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --user --no-cache-dir -r requirements-live.txt

# --- Final Production Stage ---
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/nexus/.local/bin:$PATH

WORKDIR /app

# Run the engine as a non-root user.
RUN groupadd --system nexus \
    && useradd --system --gid nexus --create-home nexus

# Copy only the Python packages installed by the builder.
COPY --from=builder /root/.local /home/nexus/.local
COPY . .

# Ensure the application and installed packages are accessible to the runtime user.
RUN chown -R nexus:nexus /app /home/nexus/.local
USER nexus

# The engine is a long-running process; this check verifies the Python process exists.
# procps is intentionally not added to the runtime image just for pgrep.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os, sys; sys.exit(0 if any('/main.py' in ' '.join(open(f'/proc/{pid}/cmdline', errors='ignore').read().split(chr(0))) for pid in os.listdir('/proc') if pid.isdigit()) else 1)"

CMD ["python", "main.py"]
