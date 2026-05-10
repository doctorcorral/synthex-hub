#!/usr/bin/env bash
#
# Build and push a multi-arch synthex-worker image so the
# `curl https://synthex.fit/install | sh` one-liner can pull it on
# both Apple-Silicon laptops and x86_64 Linux boxes.
#
# Defaults to docker.io/doctorcorral/synthex-worker:latest. Override with:
#
#     IMAGE=ghcr.io/doctorcorral/synthex-worker:latest ./deploy/publish-worker.sh
#     PLATFORMS=linux/amd64 ./deploy/publish-worker.sh        # single arch
#     TAG=v0.3.0          ./deploy/publish-worker.sh          # extra tag
#
# Requires `docker buildx` (Docker Desktop ships with it). You must be
# logged into the registry: `docker login` for Docker Hub, or
# `docker login ghcr.io` for GitHub Container Registry.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/worker"

IMAGE="${IMAGE:-doctorcorral/synthex-worker}"
TAG="${TAG:-latest}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
BUILDER="${BUILDER:-synthex-builder}"

echo "→ Image:     $IMAGE:$TAG"
echo "→ Platforms: $PLATFORMS"
echo

# Ensure a buildx builder exists (idempotent).
if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  echo "→ Creating buildx builder '$BUILDER'…"
  docker buildx create --name "$BUILDER" --use
else
  docker buildx use "$BUILDER"
fi
docker buildx inspect --bootstrap >/dev/null

# Bake `latest` plus optional explicit tag.
TAG_ARGS=(-t "${IMAGE}:${TAG}")
if [ "$TAG" != "latest" ]; then
  TAG_ARGS+=(-t "${IMAGE}:latest")
fi

echo "→ Building and pushing…"
docker buildx build \
  --platform "$PLATFORMS" \
  "${TAG_ARGS[@]}" \
  --push \
  .

echo
echo "✓ Pushed ${IMAGE}:${TAG} (${PLATFORMS})."
echo
echo "Friends can now run:"
echo "  curl -fsSL https://synthex.fit/install | sh"
