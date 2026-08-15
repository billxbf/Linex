ARG BASE_IMAGE=python:3.12-slim
ARG POLAR_SWEBENCH_IMAGE_VERSION=1
FROM ${BASE_IMAGE}

ARG POLAR_SWEBENCH_IMAGE_VERSION=1
LABEL io.polar.swebench-image-version=${POLAR_SWEBENCH_IMAGE_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    HOME=/home/polar \
    NPM_CONFIG_PREFIX=/home/polar/.npm-global \
    PATH=/home/polar/.npm-global/bin:/opt/miniconda3/envs/testbed/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && useradd -m -s /bin/bash polar \
    && mkdir -p /home/polar/.npm-global /polar/session \
    && chown -R polar:polar /home/polar /polar/session \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER polar
WORKDIR /testbed
