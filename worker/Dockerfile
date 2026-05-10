FROM elixir:1.18-slim

ENV MIX_ENV=prod \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    build-essential libosmesa6-dev patchelf \
    ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY environments/gymnasium/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

RUN mix local.hex --force && mix local.rebar --force

COPY mix.exs mix.lock ./
COPY config ./config
RUN mix deps.get --only prod && mix deps.compile

COPY lib ./lib
COPY environments ./environments
RUN mix compile

# Configurable at `docker run`:
#   SERVER_URL    e.g. https://synthex.fit/api
#   API_TOKEN     OPTIONAL — only set if the hub authenticates workers.
#                 Most public hubs leave workers anonymous.
#   POOL_SIZE     defaults to System.schedulers_online()
#   WORKER_NAME   stable identifier across restarts (e.g. "alice-mac")
ENV SERVER_URL="http://host.docker.internal:4000/api"

CMD ["mix", "run", "--no-halt"]
