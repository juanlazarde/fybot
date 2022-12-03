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
FROM python:3.9-slim AS python-base
ENV HOME_PATH="/usr/src"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    FYBOT_PATH="$HOME_PATH/fybot" \
    VENV_PATH="$HOME_PATH/fybot/.venv"
ENV PATH="$VENV_PATH/bin:$PATH"
RUN python -m venv --system-site-packages $VENV_PATH && \
    pip install --upgrade pip setuptools

######################################################################
# Create a builder-base that includes Psycopg2 & TA-LIB dependencies #
######################################################################
FROM python-base AS build
RUN apt-get --quiet update && \
    apt-get -y --no-install-recommends --quiet --show-progress install apt-utils

# install psycopg2
# ONCE ALL WORKS, CONSIDER USING psycopg2-binary in the requirements, and remove this section completely
RUN apt-get -y --no-install-recommends --quiet --show-progress install \
        libpq-dev \
        build-essential && \
    pip install psycopg2
# install ta-lib (if error try ./configure with '--build aarch64', then 'LDFLAGS="-lm"')
RUN apt-get -y --no-install-recommends --quiet --show-progress install \
        wget \
        build-essential && \
    wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz  && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    pip install ta-lib && \
    cd .. && \
    rm -R ta-lib ta-lib-0.4.0-src.tar.gz
# fybot requirements
WORKDIR $FYBOT_PATH
COPY requirements.txt .
RUN pip install --requirement requirements.txt
# cleanup installations
RUN apt-get --purge -y remove \
    libpq-dev \
    wget \
    build-essential && \
    apt-get -y autoremove && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

# streamlit requirements
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE $STREAMLIT_SERVER_PORT/tcp

##############################################################
# Create 'development' stage to install all dev dependencies #
##############################################################
FROM build AS development
ENV STAGE=development \
    PYTHONUNBUFFERED=1
COPY --from=build $FYBOT_PATH $FYBOT_PATH
COPY /fybot .
RUN pip install debugpy
CMD python -m debugpy --host 0.0.0.0 --port 5678 --wait --multiprocess -m fybot
# Debugpy for VSCode usage: 1) install here, 2) install Python extension in VSCode, 
# CMD python -m debugpy --host 0.0.0.0 --port 5678 --wait --multiprocess -m flask run -h 0.0.0 -p 5000

#############################################################
# Create 'production' stage that uses the 'builder-base' to #
# run production dependencies and scripts                   #
#############################################################
FROM build AS production
ENV STAGE=production \
    PYTHONUNBUFFERED=0
COPY --from=build $VENV_PATH $VENV_PATH
WORKDIR $FYBOT_PATH
COPY /fybot .
ENTRYPOINT ["sh", "-c", "python3 $FYBOT_PATH"]
