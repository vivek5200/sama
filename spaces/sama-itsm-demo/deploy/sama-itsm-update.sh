#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${APP_DIR:-/root/projects/sama/spaces/sama-itsm-demo}"
IMAGE_REF="ghcr.io/vivek5200/sama-itsm-demo:latest"

cd "$APP_DIR"

previous_image_id="$(docker inspect --format '{{.Image}}' sama-itsm 2>/dev/null || true)"
docker compose pull sama-itsm
new_image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")"

if [ -n "$previous_image_id" ] && [ "$previous_image_id" = "$new_image_id" ]; then
    docker compose up -d --no-build sama-itsm
    exit 0
fi

docker compose up -d --no-build sama-itsm

wait_for_healthy() {
    for attempt in $(seq 1 120); do
        health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}' sama-itsm 2>/dev/null || true)"
        if [ "$health" = healthy ]; then
            docker compose ps sama-itsm
            return 0
        fi
        if [ "$health" = unhealthy ] || [ "$health" = missing ]; then
            return 1
        fi
        sleep 15
    done
    return 1
}

if wait_for_healthy; then
    exit 0
fi

docker compose logs --tail=100 sama-itsm
if [ -n "$previous_image_id" ] && docker image inspect "$previous_image_id" >/dev/null 2>&1; then
    echo "Deployment failed; restoring image $previous_image_id"
    docker tag "$previous_image_id" "$IMAGE_REF"
    docker compose up -d --no-build sama-itsm
    if ! wait_for_healthy; then
        docker compose logs --tail=100 sama-itsm
        echo "Rollback container did not become healthy" >&2
    fi
fi

exit 1
