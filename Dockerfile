# Runnable ContextLab software; Dockerfile.task remains the isolated code executor.
FROM python:3.13.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.runtime.lock ./
RUN pip install --no-cache-dir -r requirements.runtime.lock
COPY pyproject.toml README.md uv.lock requirements.lock ./
COPY contextlab ./contextlab
COPY harness ./harness
COPY config ./config
COPY examples ./examples
RUN pip install --no-cache-dir --no-deps . \
    && useradd --uid 10001 --create-home contextlab \
    && mkdir -p /app/runs \
    && chown contextlab:contextlab /app/runs
USER 10001:10001
ENTRYPOINT ["python", "-m", "harness"]
CMD ["--help"]
