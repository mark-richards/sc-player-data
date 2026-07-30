#!/usr/bin/env bash
# Redeploy sc-dashboard without rebuilding the Docker image.
#
# Usage:
#   bash redeploy.sh          # deploy code + live data (normal use)
#   bash redeploy.sh --full   # also copy data/processed/ (slower, use after pipeline run)
#
# To rebuild the image (only needed when requirements.txt changes):
#   docker build -t mediaserver-sc-dashboard . && bash redeploy.sh
set -e

IMAGE=mediaserver-sc-dashboard
FULL=${1:-}

# pwd -W gives Windows-format path (e.g. G:/My Drive/...) required by Docker Desktop.
# MSYS_NO_PATHCONV=1 stops Git Bash from mangling /app → C:\Program Files\Git\app.
DIR="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"

echo "Stopping old container..."
docker stop sc-dashboard 2>/dev/null || true
docker rm   sc-dashboard 2>/dev/null || true

echo "Creating new container..."
MSYS_NO_PATHCONV=1 docker create \
  --name sc-dashboard \
  -p 5001:5000 \
  --restart unless-stopped \
  --env-file "$DIR/.env" \
  "$IMAGE"

echo "Copying code..."
MSYS_NO_PATHCONV=1 docker cp "$DIR/website"             sc-dashboard:/app/
MSYS_NO_PATHCONV=1 docker cp "$DIR/waiver"              sc-dashboard:/app/
MSYS_NO_PATHCONV=1 docker cp "$DIR/service_account.json" sc-dashboard:/app/

echo "Starting container..."
docker start sc-dashboard

echo "Copying live data..."
MSYS_NO_PATHCONV=1 docker exec sc-dashboard mkdir -p /app/data/processed
MSYS_NO_PATHCONV=1 docker cp "$DIR/data/live"                            sc-dashboard:/app/data/live
MSYS_NO_PATHCONV=1 docker cp "$DIR/data/processed/master_player_data.csv" sc-dashboard:/app/data/processed/master_player_data.csv

echo "Copying draft data..."
MSYS_NO_PATHCONV=1 docker exec sc-dashboard mkdir -p "/app/draft_prep"
MSYS_NO_PATHCONV=1 docker cp "$DIR/draft_prep/SC 2026"  "sc-dashboard:/app/draft_prep/SC 2026"

if [ "$FULL" = "--full" ]; then
  echo "Copying processed data (full mode)..."
  MSYS_NO_PATHCONV=1 docker cp "$DIR/data/processed"   sc-dashboard:/app/data/processed
fi

echo "Restarting container..."
docker restart sc-dashboard

# Reconnect to nginx's network so it can resolve "sc-dashboard" by name
docker network connect mediaserver_media_default sc-dashboard 2>/dev/null || true

echo "sc-dashboard restarted"
docker logs sc-dashboard --tail 10
