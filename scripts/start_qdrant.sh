#!/usr/bin/env bash
# Start a local Qdrant server for AskMe's persistent conversation memory.
#
# The bot talks to Qdrant over gRPC on port 6334 (QDRANT_URL in .env).
# The REST API stays available on port 6333 for inspection/UI.
#
# Data is stored in ./qdrant_data (gitignored) so it survives restarts.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOLUME_NAME="askme-qdrant-data"
IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:latest}"
PORT_GRPC="${QDRANT_GRPC_PORT:-6334}"
PORT_HTTP="${QDRANT_HTTP_PORT:-6333}"

already_running() {
    docker ps --filter "name=^/askme-qdrant$" --format '{{.Names}}' | grep -q askme-qdrant
}

if already_running; then
    echo "Qdrant is already running (container askme-qdrant)."
    docker ps --filter "name=^/askme-qdrant$" --format '  {{.Ports}}'
    exit 0
fi

echo "Starting Qdrant (volume: $VOLUME_NAME, gRPC :$PORT_GRPC, HTTP :$PORT_HTTP)..."

docker run -d \
    --name askme-qdrant \
    --restart unless-stopped \
    -p "$PORT_GRPC":6334 \
    -p "$PORT_HTTP":6333 \
    -v "$VOLUME_NAME":/qdrant/storage \
    "$IMAGE"

echo "Started. Waiting for the server to accept connections..."
for _ in $(seq 1 30); do
    if curl -sf "http://localhost:$PORT_HTTP/readyz" >/dev/null 2>&1; then
        echo "Qdrant is ready at http://localhost:$PORT_GRPC (gRPC) / :$PORT_HTTP (REST)."
        echo "Set QDRANT_URL=http://localhost:$PORT_GRPC in .env"
        exit 0
    fi
    sleep 1
done

echo "Qdrant did not become ready in time. Check: docker logs askme-qdrant" >&2
exit 1
