FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-mpmath \
    python3-numpy \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/ /app/
