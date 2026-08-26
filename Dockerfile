# === STAGE 1: BUILD ===
FROM python:3.12-slim AS builder

ARG APP_PATH="/usr/src/app"
ARG VENV_PATH="$APP_PATH/.venv/bin"

WORKDIR $APP_PATH

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    curl\
    && rm -rf /var/lib/apt/lists/*
# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
# Copy only dependency files (for caching)
COPY pyproject.toml uv.lock ./

# Install deps into .venv

# Install project itself (fast, uses cache)
RUN uv sync
RUN uv add 'hypercorn[uvloop]'
# === STAGE 2: RUNTIME ===
FROM python:3.12-slim AS runtime

ARG APP_PATH="/usr/src/app"
ARG VENV_PATH="$APP_PATH/.venv/bin"

WORKDIR $APP_PATH

# Install runtime libs only (no build-essential!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy app + venv
COPY --from=builder $APP_PATH $APP_PATH

COPY . .
# Use venv directly
ENV PATH="$VENV_PATH:$PATH"
# Run app
#
EXPOSE 7860
CMD ["uv","run","detection-gui"]
