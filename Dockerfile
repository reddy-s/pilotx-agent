FROM python:3.12.8 AS builder

WORKDIR /app

COPY dist/pilotx_agent-*-py3-none-any.whl ./

RUN pip install --no-cache-dir pilotx_agent-*-py3-none-any.whl

FROM python:3.12.8-slim

ARG TAG=unknown

LABEL owner="sangram@aicraft.io" \
      group-owner="developer@aicraft.io" \
      name="pilotx-agent" \
      version=${TAG}

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY resources /app/resources
COPY service /app/service
COPY src/** /app/src/

ENV CONFIG_SCHEMA_PATH=/app/resources/config-schema.yaml \
    CONFIG_PATH=/app/resources/config/config.yaml \
    LOG_CONFIG_PATH=/app/resources/logging.yaml \
    RESPONSE_TEMPLATE_FOLDER=/app/resources/templates \
    CACHE_DIR=/app/data/cache \
    LOG_LEVEL=INFO \
    BUILD_VERSION=${TAG} \
    AGENT_HOST=http://0.0.0.0:9999 \
    DEV_MODE=false \
    STATE_PATH=/app/data/db

VOLUME ["/app/data"]

# ADK Web UI
EXPOSE 8888
# Data Explorer A2A Service
EXPOSE 9998
# API service
EXPOSE 9990

WORKDIR /app/service

ENTRYPOINT ["python", "main.py"]


