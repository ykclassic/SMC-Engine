# 1. Use a specific digest or version for reproducibility
FROM python:3.9-slim as builder

# 2. Set environment variables to optimize Python performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# 3. Install build dependencies (needed for some financial/AI libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. Install dependencies first (Layer Caching)
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# --- Final Production Stage ---
FROM python:3.9-slim

WORKDIR /app

# 5. Create a non-root user for security (The "TechSolute" Standard)
RUN groupadd -r nexus && useradd -r -g nexus nexus

# 6. Copy only the installed packages from builder
COPY --from=builder /root/.local /home/nexus/.local
COPY . .

# 7. Ensure correct ownership
RUN chown -R nexus:nexus /app
USER nexus

# 8. Add local bin to path so 'nexus' user can run installed packages
ENV PATH=/home/nexus/.local/bin:$PATH

# 9. Healthcheck to ensure the engine is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD pgrep python || exit 1

CMD ["python", "main.py"]
