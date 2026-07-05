FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt requirements-core.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY run.py pytest.ini ./
COPY src/ src/
COPY scripts/ scripts/
COPY data/e2e_prompts.json data/example_prompts.json data/
COPY data/examples/ data/examples/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/app/src
EXPOSE 8090 8080 8001

ENTRYPOINT ["/entrypoint.sh"]
CMD ["proxy"]
