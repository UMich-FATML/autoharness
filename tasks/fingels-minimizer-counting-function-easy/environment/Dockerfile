FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY repo /app/repo
COPY paper /app/paper

RUN pip install --no-cache-dir numpy==1.26.4
