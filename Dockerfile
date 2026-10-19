# One image runs both the API and the dashboard (compose.yaml picks the command).
FROM python:3.13-slim

# g++ and cmake build the C++ engine; libgomp1 is the OpenMP runtime LightGBM needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ cmake make libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what the build needs. Dependencies change less often than code, but the C++ extension
# is built from the whole tree, so one COPY keeps this simple.
COPY pyproject.toml CMakeLists.txt README.md ./
COPY cpp ./cpp
COPY src ./src
COPY dashboard ./dashboard
RUN pip install --no-cache-dir ".[service]"

# Train the models once, at build time, so `docker compose up` starts instantly and the
# "adaptive" strategy works. This takes a couple of minutes and is deterministic (fixed seeds).
RUN python -m lobx.cli train --out models

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8000 8501
CMD ["python", "-m", "lobx.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
