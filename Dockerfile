# Inference image for the cats vs dogs classifier.
# Two stages so the compiler toolchain and the pip cache never reach the
# final image. The result is roughly a third of the size of a single stage
# build that installs the CUDA flavour of torch.

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

# torch comes from the CPU wheel index first. That keeps it near 200 MB
# instead of the 2.5 GB CUDA build, which would be dead weight because
# serving runs on CPU. The second install then sees torch as satisfied.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       torch==2.13.0 torchvision==0.28.0 \
    && pip install --no-cache-dir -r requirements-api.txt


FROM python:3.12-slim

LABEL org.opencontainers.image.title="cats-vs-dogs-classifier" \
      org.opencontainers.image.description="Baseline CNN inference service" \
      org.opencontainers.image.source="https://github.com/Saqib8/2024AD05127-MLOps-Assignment2"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/cats_dogs_cnn.pt \
    PORT=8000

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Run as an unprivileged user. Kubernetes is configured to enforce this too,
# see k8s/deployment.yaml.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
