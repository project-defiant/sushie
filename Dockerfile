FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE.txt /app/
COPY sushie /app/sushie

ARG SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SUSHIE=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SUSHIE=${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SUSHIE}
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SUSHIE}

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["sushie"]
