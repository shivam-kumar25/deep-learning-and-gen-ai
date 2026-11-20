# Contained browser tests for the BDM notes and app: headless Chromium + Node + a static file server, nothing on the host desktop.
# Build:  docker build -t ml-notes-test tools/notes/test
# Run:    see tools/notes/test/docker-test.sh
FROM node:22-bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends chromium python3 fonts-dejavu-core ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /tests
COPY package.json ./
RUN npm install --omit=dev --no-audit --no-fund
ENV CHROME_PATH=/usr/bin/chromium \
    CHROME_ARGS="--no-sandbox --disable-dev-shm-usage" \
    NODE_PATH=/tests/node_modules
