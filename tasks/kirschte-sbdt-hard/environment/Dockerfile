FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
 && rm -rf /var/lib/apt/lists/*

COPY repo /app/repo
COPY paper /app/paper
