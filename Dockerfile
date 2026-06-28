FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py /app/server.py
WORKDIR /app

ENV PORT=8080
EXPOSE 8080

CMD pip install --upgrade yt-dlp && gunicorn --bind 0.0.0.0:8080 --timeout 300 --workers 2 server:app
