#!/bin/sh
# shellcheck shell=sh
#
# Synthex Hub — worker installer.
#
# Run via:
#
#     curl -fsSL https://synthex.fit/install | sh
#
# Default behavior: build the worker image directly from the public
# GitHub repo — no registry, no `git clone`, no `docker login`. Docker
# fetches the source, builds, runs.
#
# No token required: workers connect anonymously. Only batch
# submission (the master) is authenticated.
#
# Optional env:
#     SERVER_URL         default: https://synthex.fit/api
#     WORKER_NAME        display name on the public leaderboard.
#                        default: prompted (suggesting $USER@$(hostname));
#                        type "anonymous" or leave blank to opt out
#                        of recognition.
#     POOL_SIZE          absolute number of python interpreters to run.
#                        when set, takes priority over DONATE_PCT and
#                        skips the interactive "% of cores" prompt.
#     DONATE_PCT         percent of CPU cores to donate (1-100).
#                        used only if POOL_SIZE is unset.
#                        default: prompted (suggesting 50%); skipped
#                        in NONINTERACTIVE mode where it falls back
#                        to 50%.
#     CONTAINER_NAME     default: synthex-worker
#     VOLUME_NAME        named docker volume that persists the stable
#                        worker_id (UUID) across restarts so per-worker
#                        contribution credit doesn't reset.
#                        default: synthex-worker-data
#     API_TOKEN          only set if running against a hub that
#                        explicitly authenticates workers.
#     NONINTERACTIVE     skip the WORKER_NAME prompt entirely (handy
#                        for ssh + here-doc, CI, etc).
#
#     # Build mode (default):
#     BUILD_FROM         git URL + ref + subdir for the build context.
#                        default: https://github.com/doctorcorral/synthex-hub.git#main:worker
#     LOCAL_TAG          local tag for the built image.
#                        default: synthex-worker:local
#
#     # Pull mode (override):
#     IMAGE              if set, skip the build step and `docker pull`
#                        this image instead. e.g. ghcr.io/doctorcorral/synthex-worker:latest
#
set -eu

# ── colors (only on a TTY) ──────────────────────────────────
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_RESET=$(printf '\033[0m')
  C_BOLD=$(printf '\033[1m')
  C_DIM=$(printf '\033[2m')
  C_GREEN=$(printf '\033[32m')
  C_YELLOW=$(printf '\033[33m')
  C_RED=$(printf '\033[31m')
  C_CYAN=$(printf '\033[36m')
else
  C_RESET=''; C_BOLD=''; C_DIM=''
  C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN=''
fi

say()   { printf '%s%s%s\n'   "$C_CYAN"   "→ $*" "$C_RESET" >&2; }
ok()    { printf '%s%s%s\n'   "$C_GREEN"  "✓ $*" "$C_RESET" >&2; }
warn()  { printf '%s%s%s\n'   "$C_YELLOW" "! $*" "$C_RESET" >&2; }
die()   { printf '%s%s%s\n'   "$C_RED"    "✗ $*" "$C_RESET" >&2; exit 1; }
hr()    { printf '%s%s%s\n'   "$C_DIM"    "$(printf '%.0s─' $(seq 1 60))" "$C_RESET" >&2; }
banner() {
  printf '%s\n' "$C_BOLD"
  cat <<'B'
   ███████ ██    ██ ███    ██ ████████ ██   ██ ███████ ██   ██
   ██       ██  ██  ████   ██    ██    ██   ██ ██       ██ ██
   ███████   ████   ██ ██  ██    ██    ███████ █████     ███
        ██    ██    ██  ██ ██    ██    ██   ██ ██       ██ ██
   ███████    ██    ██   ████    ██    ██   ██ ███████ ██   ██
                                                            hub
B
  printf '%s%s   distributed coinductive policy synthesis · synthex.fit%s\n\n' \
         "$C_DIM" "" "$C_RESET"
}

# ── prerequisites ───────────────────────────────────────────
need_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    warn "Docker is not installed."
    case "$(uname -s)" in
      Darwin)
        printf '  Install Docker Desktop:    https://www.docker.com/products/docker-desktop/\n'
        printf '  Or via Homebrew:           brew install --cask docker\n'
        ;;
      Linux)
        printf '  Get Docker:                https://docs.docker.com/engine/install/\n'
        ;;
      *)
        printf '  See https://docs.docker.com/get-docker/\n'
        ;;
    esac
    die "re-run this installer once Docker is available."
  fi

  if ! docker info >/dev/null 2>&1; then
    die "Docker is installed but not running. Start Docker Desktop (or 'sudo systemctl start docker') and re-run."
  fi
  ok "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo '(unknown)') is running."
}

# ── defaults / detection ────────────────────────────────────
detect_cores() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu 2>/dev/null || echo 4
  elif command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4
  else
    echo 4
  fi
}

# ── main ────────────────────────────────────────────────────
banner
need_docker

SERVER_URL="${SERVER_URL:-https://synthex.fit/api}"
TOTAL_CORES="$(detect_cores)"
CONTAINER_NAME="${CONTAINER_NAME:-synthex-worker}"
VOLUME_NAME="${VOLUME_NAME:-synthex-worker-data}"

# Default donation percentage. Picked low on purpose: pegging
# someone's laptop at 100% on the first install is a great way to
# guarantee they uninstall five minutes later. They can dial it up
# (or pass POOL_SIZE explicitly) once they've confirmed nothing
# catches fire.
DEFAULT_DONATE_PCT=50

# Prompt for a display name unless one was passed via env / NONINTERACTIVE
# is set. The default suggestion is $USER@$(hostname); an empty answer
# means "anonymous" (your contributions still count, but won't show
# up under your name on the leaderboard).
suggested_name="$(printf '%s@%s' "${USER:-anon}" "$(hostname 2>/dev/null | cut -d. -f1)")"

if [ -z "${WORKER_NAME:-}" ] && [ -z "${NONINTERACTIVE:-}" ] && [ -r /dev/tty ]; then
  printf '\n%sChoose a display name for the public leaderboard.%s\n' "$C_BOLD" "$C_RESET" >&2
  printf '%s  · type a handle (e.g. "alice", "bob@univ")%s\n' "$C_DIM" "$C_RESET" >&2
  printf '%s  · press ENTER to use the suggested default%s\n' "$C_DIM" "$C_RESET" >&2
  printf '%s  · type "anonymous" to opt out of recognition%s\n' "$C_DIM" "$C_RESET" >&2
  printf '\n  display name [%s%s%s]: ' "$C_BOLD" "$suggested_name" "$C_RESET" >&2
  read -r entered_name </dev/tty || entered_name=""
  WORKER_NAME="${entered_name:-$suggested_name}"
fi
# Final fallback for non-interactive cases.
WORKER_NAME="${WORKER_NAME:-anonymous}"

# Resolve POOL_SIZE.
#   1. POOL_SIZE explicitly set in env → honor it verbatim.
#   2. else interactive shell → prompt for donation %, default 50.
#   3. else non-interactive → DONATE_PCT env var, default 50.
# In every branch we end up with a positive integer in POOL_SIZE.
resolve_pool_size() {
  if [ -n "${POOL_SIZE:-}" ]; then
    return 0
  fi

  pct=""
    if [ -z "${DONATE_PCT:-}" ] && [ -z "${NONINTERACTIVE:-}" ] && [ -r /dev/tty ]; then
    printf '\n%sHow much of this machine to donate?%s\n' "$C_BOLD" "$C_RESET" >&2
    printf '%s  detected %s CPU cores. one python interpreter per donated core.%s\n' \
      "$C_DIM" "$TOTAL_CORES" "$C_RESET" >&2
    printf '%s  · enter a percent 1-100 (e.g. 25, 50, 75)%s\n' "$C_DIM" "$C_RESET" >&2
    printf '%s  · press ENTER for the default — you can change it any time%s\n' "$C_DIM" "$C_RESET" >&2

    while :; do
      printf '\n  cores to donate [%s%s%%%s]: ' "$C_BOLD" "$DEFAULT_DONATE_PCT" "$C_RESET" >&2
      read -r entered_pct </dev/tty || entered_pct=""
      entered_pct="${entered_pct%\%}"
      if [ -z "$entered_pct" ]; then
        pct="$DEFAULT_DONATE_PCT"
        break
      fi
      case "$entered_pct" in
        ''|*[!0-9]*)
          warn "please enter an integer 1-100 (e.g. 50)"
          continue
          ;;
      esac
      if [ "$entered_pct" -ge 1 ] && [ "$entered_pct" -le 100 ]; then
        pct="$entered_pct"
        break
      fi
      warn "out of range — please enter 1-100"
    done
  else
    pct="${DONATE_PCT:-$DEFAULT_DONATE_PCT}"
  fi

  POOL_SIZE=$(( TOTAL_CORES * pct / 100 ))
  if [ "$POOL_SIZE" -lt 1 ]; then
    POOL_SIZE=1
  fi
  DONATE_PCT_RESOLVED="$pct"
}

resolve_pool_size

# Image acquisition mode: explicit IMAGE wins (registry pull); otherwise
# we build from the git URL. This keeps the default frictionless — the
# operator never has to publish anywhere.
BUILD_FROM="${BUILD_FROM:-https://github.com/doctorcorral/synthex-hub.git#main:worker}"
LOCAL_TAG="${LOCAL_TAG:-synthex-worker:local}"

if [ -n "${IMAGE:-}" ]; then
  ACQUIRE_MODE=pull
  RUN_IMAGE="$IMAGE"
else
  ACQUIRE_MODE=build
  RUN_IMAGE="$LOCAL_TAG"
fi

hr
printf '  %sserver%s     %s\n' "$C_DIM" "$C_RESET" "$SERVER_URL"
printf '  %sname%s       %s' "$C_DIM" "$C_RESET" "$WORKER_NAME"
if [ "$WORKER_NAME" = "anonymous" ]; then
  printf ' %s(opted out of recognition)%s' "$C_DIM" "$C_RESET"
fi
printf '\n'
if [ -n "${DONATE_PCT_RESOLVED:-}" ]; then
  printf '  %spool size%s  %s python interpreters %s(%s%% of %s detected cores)%s\n' \
    "$C_DIM" "$C_RESET" "$POOL_SIZE" "$C_DIM" "$DONATE_PCT_RESOLVED" "$TOTAL_CORES" "$C_RESET"
else
  printf '  %spool size%s  %s python interpreters %s(POOL_SIZE override; %s cores detected)%s\n' \
    "$C_DIM" "$C_RESET" "$POOL_SIZE" "$C_DIM" "$TOTAL_CORES" "$C_RESET"
fi
printf '  %sid volume%s  %s %s(persists worker_id across restarts)%s\n' \
  "$C_DIM" "$C_RESET" "$VOLUME_NAME" "$C_DIM" "$C_RESET"
if [ "$ACQUIRE_MODE" = pull ]; then
  printf '  %simage%s      %s %s(pulled from registry)%s\n' \
    "$C_DIM" "$C_RESET" "$RUN_IMAGE" "$C_DIM" "$C_RESET"
else
  printf '  %ssource%s     %s\n' "$C_DIM" "$C_RESET" "$BUILD_FROM"
  printf '  %simage%s      %s %s(built locally)%s\n' \
    "$C_DIM" "$C_RESET" "$RUN_IMAGE" "$C_DIM" "$C_RESET"
fi
printf '  %scontainer%s  %s\n' "$C_DIM" "$C_RESET" "$CONTAINER_NAME"
hr

# Reachability sanity check (non-fatal — laptops behind captive portals etc).
if command -v curl >/dev/null 2>&1; then
  if curl -fsS --max-time 5 "${SERVER_URL%/api}/health" >/dev/null 2>&1; then
    ok "Hub is reachable."
  else
    warn "Could not reach ${SERVER_URL%/api}/health — proceeding anyway."
  fi
fi

# Acquire the image.
case "$ACQUIRE_MODE" in
  pull)
    say "Pulling ${RUN_IMAGE}…"
    if ! docker pull "$RUN_IMAGE"; then
      die "could not pull $RUN_IMAGE. If the image is private, ask the operator to grant access or set BUILD_FROM=… to build from source instead."
    fi
    ;;
  build)
    say "Building ${RUN_IMAGE} from ${BUILD_FROM} (one-time, ~3-5 min)…"
    say "Subsequent installs reuse the layer cache; faster after the first run."
    if ! docker build --tag "$RUN_IMAGE" --pull "$BUILD_FROM"; then
      die "build failed. Common causes:
  • outbound HTTPS to github.com is blocked
  • the repo at $BUILD_FROM is not publicly accessible
  • disk full

If the worker repo is private, set IMAGE=… to use a registry image,
or BUILD_FROM=… to point at a fork/mirror you control."
    fi
    ;;
esac

# Stop + remove any prior container so re-running is idempotent.
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  say "Removing existing container ‘${CONTAINER_NAME}’…"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# Launch.
say "Starting worker…"

# Build the env-flag list. API_TOKEN is only forwarded if explicitly
# set — anonymous workers are first-class.
DOCKER_ENV_FLAGS="-e SERVER_URL=$SERVER_URL -e WORKER_NAME=$WORKER_NAME -e POOL_SIZE=$POOL_SIZE"
if [ -n "${API_TOKEN:-}" ]; then
  DOCKER_ENV_FLAGS="$DOCKER_ENV_FLAGS -e API_TOKEN=$API_TOKEN"
fi

# Make sure the named volume exists. Mounting it at /var/synthex
# inside the container persists worker_id across container removal,
# image rebuilds, and machine restarts — so contribution credit
# accumulates against ONE identity instead of fragmenting per
# container instance.
docker volume create "$VOLUME_NAME" >/dev/null

# shellcheck disable=SC2086
CONTAINER_ID=$(
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -v "$VOLUME_NAME":/var/synthex \
    $DOCKER_ENV_FLAGS \
    "$RUN_IMAGE"
)

# Give it a moment, then check it's still running.
sleep 2
if ! docker ps --filter "id=$CONTAINER_ID" --format '{{.ID}}' | grep -q .; then
  warn "Container exited immediately. Last 30 log lines:"
  docker logs --tail 30 "$CONTAINER_NAME" >&2 || true
  die "worker failed to start."
fi

ok "Worker ${C_BOLD}${WORKER_NAME}${C_RESET}${C_GREEN} is connected.${C_RESET}"
hr
cat <<EOF >&2
Useful commands:

  ${C_BOLD}docker logs -f ${CONTAINER_NAME}${C_RESET}${C_DIM}     # tail logs${C_RESET}
  ${C_BOLD}docker stats ${CONTAINER_NAME}${C_RESET}${C_DIM}       # cpu / memory${C_RESET}
  ${C_BOLD}docker stop ${CONTAINER_NAME}${C_RESET}${C_DIM}        # pause donating compute${C_RESET}
  ${C_BOLD}docker rm -f ${CONTAINER_NAME}${C_RESET}${C_DIM}       # uninstall${C_RESET}

EOF
if [ "$ACQUIRE_MODE" = build ]; then
  cat <<EOF >&2
To update to the latest worker code:

  ${C_BOLD}docker rm -f ${CONTAINER_NAME}${C_RESET}
  ${C_BOLD}curl -fsSL ${SERVER_URL%/api}/install | sh${C_RESET}${C_DIM}
                                # rebuilds from the latest commit${C_RESET}

EOF
fi
cat <<EOF >&2
Watch the cluster:

  ${C_BOLD}open ${SERVER_URL%/api}/${C_RESET}${C_DIM}              # landing page${C_RESET}

EOF
