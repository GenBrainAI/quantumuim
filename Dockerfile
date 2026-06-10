# syntax=docker/dockerfile:1

# ---------- Build stage ----------
FROM python:3.12-slim AS builder

WORKDIR /build

# Build tooling only in this stage
RUN pip install --no-cache-dir build

# Copy metadata first for layer caching, then sources
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src/ src/

RUN python -m build --wheel --outdir /build/dist

# ---------- Runtime stage ----------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="quantumium" \
      org.opencontainers.image.description="Quantumium quantum computing toolkit (Q+ circuit compression, fermionic simulation)" \
      org.opencontainers.image.source="https://github.com/GenBrainAI/quantumuim" \
      org.opencontainers.image.licenses="MIT"

# Non-root user
RUN useradd --create-home --shell /usr/sbin/nologin quantumium
USER quantumium
WORKDIR /home/quantumium

ENV PATH="/home/quantumium/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --user /tmp/*.whl

# Default: drop into a Python REPL with the package available
CMD ["python", "-c", "import quantumium; print(f'quantumium {getattr(quantumium, \"__version__\", \"dev\")} ready'); import code; code.interact(local={'quantumium': quantumium})"]
