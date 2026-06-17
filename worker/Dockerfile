FROM elixir:1.18-slim

ENV MIX_ENV=prod \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    build-essential libosmesa6-dev patchelf \
    swig \
    ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# WARP=1 builds a GPU worker: adds warp-lang + mujoco-warp on top of
# the CPU deps. The warp-lang wheel bundles the CUDA runtime, so this
# image needs no CUDA base — only the host's NVIDIA driver, injected at
# `docker run --gpus all` via the nvidia-container-toolkit. The same
# image still runs plain MuJoCo (CPU) chunks, so a GPU worker is a
# strict superset.
ARG WARP=0

COPY environments/gymnasium/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    if [ "$WARP" = "1" ]; then \
      pip install --no-cache-dir warp-lang mujoco-warp ; \
    fi

WORKDIR /app

RUN mix local.hex --force && mix local.rebar --force

COPY mix.exs mix.lock ./
COPY config ./config
RUN mix deps.get --only prod && mix deps.compile

COPY lib ./lib
COPY environments ./environments
COPY scripts ./scripts
RUN mix compile

# Configurable at `docker run`:
#   SERVER_URL    e.g. https://synthex.fit/api
#   API_TOKEN     OPTIONAL — only set if the hub authenticates workers.
#                 Most public hubs leave workers anonymous.
#   POOL_SIZE     defaults to System.schedulers_online()
#   WORKER_NAME   display name on the public leaderboard. Set to
#                 "anonymous" (or leave empty) to opt out of recognition.
#   WORKER_ID     stable internal UUID. Auto-generated and persisted
#                 to /var/synthex/worker_id on first start; mount
#                 a Docker volume there to survive container removal.
ENV SERVER_URL="http://host.docker.internal:4000/api"

VOLUME ["/var/synthex"]

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["mix", "run", "--no-halt"]
