#!/usr/bin/env bash
# Run a notes browser test inside Docker. The repo is mounted read-only; only tools/notes/test/shots is writable.
# A static server on 127.0.0.1:8765 inside the container serves the repo root.
# Usage:  tools/notes/test/docker-test.sh ml-run.mjs foundation/07_x.html "regex" ...
#         tools/notes/test/docker-test.sh ml-shots.mjs foundation/index.html ...
#         tools/notes/test/docker-test.sh app-e2e.mjs http://127.0.0.1:4300   (the app running on this machine, PORT=4300)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
mkdir -p "$HERE/shots"
docker image inspect ml-notes-test >/dev/null 2>&1 || docker build -t ml-notes-test "$HERE"
script="$1"; shift
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$ROOT:/repo:ro" -v "$HERE/shots:/shots" -e SHOTS=/shots --add-host=host.docker.internal:host-gateway \
  ml-notes-test bash -c '
    cd /repo && python3 -m http.server 8765 --bind 127.0.0.1 >/dev/null 2>&1 &
    # 127.0.0.1:4300 in here -> the app on the host, so it is "localhost" to the browser (its session cookie is Secure)
    node -e "require(\"net\").createServer(c => { const u = require(\"net\").connect(4300, \"host.docker.internal\"); c.pipe(u).pipe(c); u.on(\"error\", () => c.destroy()); c.on(\"error\", () => u.destroy()); }).listen(4300, \"127.0.0.1\")" &
    for i in $(seq 50); do (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null && break; sleep 0.1; done
    cp /repo/tools/notes/test/*.mjs /tests/ && cd /tests && node "$0" /shots "$@"' "$script" "$@"
