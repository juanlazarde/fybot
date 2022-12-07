# syntax=docker/dockerfile:latest

# Dockerfile
# Uses multi-stage builds requiring Docker 17.05 or higher
# See https://docs.docker.com/develop/develop-images/multistage-build/
# Best Practices: https://www.docker.com/blog/intro-guide-to-dockerfile-best-practices/
# Usage: sudo docker build -t fybot .
# Visualize with Buildkit: export DOCKER_BUILDKIT=1 

##########################################################
# Create a Python base with shared environment variables #
##########################################################
ARG PYTHONVERSION="3.9"
FROM python:$PYTHONVERSION-slim AS base

ENV APP_PATH="/usr/src/fybot" \
    VIRTUAL_ENV="/opt/venv" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN set -eux && \
    python -m venv $VIRTUAL_ENV && \
    pip install --upgrade pip setuptools wheel

# streamlit
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE $STREAMLIT_SERVER_PORT/tcp

##########################################################
# Create a build stage to install dependencies           #
##########################################################
FROM base as build

RUN set -eux; apt-get update

# install psycopg2 dependencies
# TODO: Consider psycopg2-binary in requirements.txt and remove this section
RUN set -eux && \
    apt-get -y --no-install-recommends install libpq-dev build-essential

# install ta-lib dependencies
RUN set -eux && \
    apt-get -y --no-install-recommends install wget build-essential dpkg-dev file && \
    cd /tmp && \
    wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz  && \
    cd ta-lib/ && \
    gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)" && \
    ./configure --prefix=/usr --build="$gnuArch" && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# fybot requirements
WORKDIR $APP_PATH
COPY requirements.txt ../
RUN set -eux && \
    pip install --requirement ../requirements.txt

# cleanup installations
RUN set -eux && \
    apt-get --purge -y remove libpq-dev wget build-essential dpkg-dev && \
    apt-get -y autoremove && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

##############################################################
# Create 'development' stage to install all dev dependencies #
##############################################################
# Using Debugpy for VSCode. Usage:
#  1) insert script here.
#  2) install Python extension in VSCode.
#  3) create `launch.json` debug configuration.
#  4) insert breakpoint in the code and hit F5.

FROM base AS development
ENV DEBUG=1

# app
COPY --from=build $APP_PATH $APP_PATH
COPY --from=build $VIRTUAL_ENV $VIRTUAL_ENV
WORKDIR $APP_PATH

# fybot requirements
COPY requirements.dev.txt ../
RUN set -eux && \
    pip install --requirement ../requirements.dev.txt

# VSCode debugging plugin
# https://github.com/microsoft/debugpy/wiki
EXPOSE 5678
RUN pip install debugpy

# update app
ADD /fybot .

# run debug command
CMD ["/bin/sh", "-c", "$VIRTUAL_ENV/bin/python -m debugpy --listen 0.0.0.0:5678 --wait-for-client __main__.py"]

# #############################################################
# # Create 'production' stage that uses the 'builder-base' to #
# # run production dependencies and scripts                   #
# #############################################################
FROM base AS production
ENV DEBUG=0 \
    PYTHONUNBUFFERED=0

# app
COPY --from=build $VIRTUAL_ENV $VIRTUAL_ENV
WORKDIR $APP_PATH
COPY /fybot .

ENTRYPOINT ["/bin/sh", "-c", "$VIRTUAL_ENV/bin/python -m $APP_PATH"]
