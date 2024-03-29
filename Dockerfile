# syntax=docker/dockerfile:latest

# Build: sudo docker build -t fybot .
# export DOCKER_BUILDKIT=1 

# Python base
FROM python:3.9-slim
# Maintainer property
LABEL maintainer="fybot@lazarde.com"
# Upgrades pip
RUN pip install --upgrade pip
# Volume working directory
WORKDIR /usr/src/fybot
# Copy dependencies
COPY requirements.txt .
# Install Ta-Lib which is a pain in the 4$$
RUN apt install -y wget && \
    wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install
RUN rm -R ta-lib ta-lib-0.4.0-src.tar.gz
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy source code to working directory
COPY /fybot .
# Python variable to print logs real-time, set to 0 for production
ENV PYTHONUNBUFFERED=0
# Expose Default Streamlit port on Docker
EXPOSE 8501/tcp
# Force Streamlit to use specified port
ENV STREAMLIT_SERVER_PORT=8501
# Run command
ENTRYPOINT ["python3", "/usr/src/fybot"]
