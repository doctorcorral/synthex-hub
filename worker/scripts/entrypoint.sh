#!/bin/sh
# Worker entrypoint — bootstraps a stable WORKER_ID before launching
# the Elixir app.
#
# Why we need this: Docker containers are ephemeral, but per-experiment
# contribution tracking on the hub depends on a stable worker_id. We
# generate a UUID v4 on first start and persist it to /var/synthex/
# worker_id (mounted as a named volume — see docker-compose.yml). On
# every subsequent start, we re-read the same UUID, so all chunks the
# worker submits are attributed to the same identity even after
# container restarts, image rebuilds, etc.
#
# WORKER_NAME is what shows on the public leaderboard. Defaults to
# "anonymous" — set explicitly to opt into recognition.

set -e

WORKER_ID_FILE="${WORKER_ID_FILE:-/var/synthex/worker_id}"

# Ensure parent dir exists; harmless if the volume already created it.
mkdir -p "$(dirname "$WORKER_ID_FILE")" 2>/dev/null || true

if [ -z "${WORKER_ID:-}" ]; then
  if [ -s "$WORKER_ID_FILE" ]; then
    WORKER_ID=$(cat "$WORKER_ID_FILE")
  else
    if command -v uuidgen >/dev/null 2>&1; then
      WORKER_ID=$(uuidgen | tr 'A-Z' 'a-z')
    elif [ -r /proc/sys/kernel/random/uuid ]; then
      WORKER_ID=$(cat /proc/sys/kernel/random/uuid)
    else
      # Fallback: hostname + nanoseconds (still better than nothing).
      WORKER_ID="w-$(hostname 2>/dev/null || echo unknown)-$(date +%s%N)"
    fi
    # Best-effort persistence — fine if the volume isn't mounted, the
    # ID will just regenerate on next start. We log this so users know.
    if echo "$WORKER_ID" > "$WORKER_ID_FILE" 2>/dev/null; then
      echo "[entrypoint] new worker_id generated and saved: $WORKER_ID"
    else
      echo "[entrypoint] WARN: could not persist worker_id to $WORKER_ID_FILE — set up a Docker volume to survive restarts"
      echo "[entrypoint]       generated worker_id: $WORKER_ID (will regenerate next start)"
    fi
  fi
fi

export WORKER_ID

# WORKER_NAME defaults to "anonymous" so unconfigured installs don't
# pollute the leaderboard with hostnames or container hashes.
export WORKER_NAME="${WORKER_NAME:-anonymous}"

echo "[entrypoint] worker_id=$WORKER_ID  worker_name=$WORKER_NAME  pool_size=${POOL_SIZE:-auto}"

exec "$@"
