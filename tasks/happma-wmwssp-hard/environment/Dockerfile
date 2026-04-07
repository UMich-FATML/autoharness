FROM python:3.12-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y r-base && rm -rf /var/lib/apt/lists/*

COPY wmwssp /app/wmwssp
