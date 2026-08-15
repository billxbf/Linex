FROM node:22-bookworm-slim AS node

FROM python:3.12-slim-bookworm

ARG POLAR_COUNT_STARS_IMAGE_VERSION=1

COPY --from=node /usr/local/ /usr/local/

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/polar \
    NPM_CONFIG_PREFIX=/home/polar/.local \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    PATH=/home/polar/.local/bin:/home/polar/.venv/bin:${PATH}

LABEL io.polar.count-stars-image-version="${POLAR_COUNT_STARS_IMAGE_VERSION}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        python-is-python3 \
        sudo \
        tmux \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash polar \
    && echo 'polar ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && mkdir -p /polar/session /home/polar/.local /home/polar/.venv \
    && chown -R polar:polar /polar/session /home/polar

RUN echo 'export PATH=/home/polar/.local/bin:/home/polar/.venv/bin:$PATH' \
    > /etc/profile.d/polar-path.sh

USER polar

WORKDIR /polar/session

