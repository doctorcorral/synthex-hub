# Synthex Hub at synthex.fit via Cloudflare Tunnel

End-to-end walkthrough: from a clean Mac/Linux machine to
`https://synthex.fit/api` reachable worldwide, using only outbound
connections (no port-forwarding, no static IP, no VPS).

## Prerequisites

- Domain `synthex.fit` (or any domain you own).
- A Cloudflare account (free).
- Docker Desktop (or Docker Engine on Linux).
- ~10 minutes.

## 1. Move synthex.fit's DNS to Cloudflare

If your registrar already uses Cloudflare DNS, skip this. Otherwise:

1. <https://dash.cloudflare.com> → Add a site → `synthex.fit`.
2. Pick the Free plan.
3. Cloudflare imports your existing records and gives you two
   nameservers (e.g. `cory.ns.cloudflare.com`,
   `lily.ns.cloudflare.com`).
4. Update the nameservers at your domain registrar.
5. Wait for propagation (Cloudflare emails you; usually < 1 hour).

## 2. Install and authenticate cloudflared (locally)

```bash
brew install cloudflared              # macOS
# or: see https://github.com/cloudflare/cloudflared/releases for Linux

cloudflared tunnel login
```

`tunnel login` opens a browser; pick `synthex.fit`. This drops a
certificate at `~/.cloudflared/cert.pem` that lets you create
tunnels and DNS records for that domain.

## 3. Create the tunnel and DNS route

```bash
# Create a named tunnel (one-time, idempotent if it exists).
cloudflared tunnel create synthex-hub

# Wire the public hostname to the tunnel.
cloudflared tunnel route dns synthex-hub synthex.fit

# Print the per-tunnel token. Copy this. You'll paste it as
# TUNNEL_TOKEN in .env below.
cloudflared tunnel token synthex-hub
```

The token is a long base64 blob that looks like
`eyJhIjoiYWFhYS…`. It identifies *this specific tunnel*; treat it as
a secret.

## 4. Configure and bring up the stack

```bash
cd deploy/cloudflared
cp .env.example .env
${EDITOR:-vim} .env
```

Set at minimum:

```dotenv
TUNNEL_TOKEN=<the long blob from step 3>
API_TOKEN=<openssl rand -hex 32>
POSTGRES_PASSWORD=<openssl rand -hex 16>
```

Then build and launch:

```bash
docker compose up -d --build
```

First boot takes ~2 minutes (Elixir build). Subsequent boots are
seconds. Check it:

```bash
# Local healthcheck (bypasses the tunnel)
curl -s http://localhost:4000/health

# Public healthcheck (through Cloudflare)
curl -s https://synthex.fit/health

# Authenticated status
curl -s https://synthex.fit/api/status \
  -H "Authorization: Bearer $(grep ^API_TOKEN= .env | cut -d= -f2)" | jq
```

You should see `{"status":"ok",...}` from both, and a JSON `status`
object from the authenticated call.

## 5. Hand a worker invocation to a friend

Send them this and only this:

```bash
docker run --rm \
  -e SERVER_URL=https://synthex.fit/api \
  -e API_TOKEN=<your shared token> \
  -e WORKER_NAME=$(hostname) \
  -e POOL_SIZE=8 \
  doctorcorral/synthex-worker:latest
```

(Or push your worker image to a registry your friends can pull from
— `docker buildx build --push -t ghcr.io/<you>/synthex-worker:latest`.)

## 6. Daily ops

```bash
# Tail logs from any service
docker compose logs -f server
docker compose logs -f cloudflared
docker compose logs -f postgres

# Restart only the server (e.g. after pulling a new image)
docker compose up -d --build server

# Stop everything but keep the database
docker compose down

# Nuke everything including the database (irreversible)
docker compose down -v

# psql into the database
docker compose exec postgres psql -U synthex -d synthex_hub
```

## Tuning notes

- **Cloudflare Tunnel response timeout** is 100 s on Free plans. Our
  worker → hub HTTP calls are sub-second, so this is not a real
  constraint. (If you ever build a long-poll endpoint, bump the
  Cloudflare plan or use websockets.)
- **DDoS / rate limiting** is on by default; if friends suddenly see
  random 5xxs, it's almost always Cloudflare deciding their
  burst-rate looks bot-like. Add their IPs to a Cloudflare WAF rule
  exception or scope down rate-limit rules in the dashboard.
- **Multiple tunnels for HA**: `docker compose up -d --scale cloudflared=2`
  gives you two cloudflared replicas connecting to the same tunnel;
  Cloudflare load-balances across them automatically.
- **TLS**: terminated at Cloudflare's edge (HTTPS to the world,
  encrypted tunnel from cloudflared back to your machine). The
  origin connection between cloudflared and `server:4000` is plain
  HTTP inside Docker's bridge network. That's fine.

## Why not put cloudflared on the host instead of in compose?

You can. `brew services start cloudflared` works. We containerize it
to keep the entire stack as one `docker compose up -d`, which is
easier to reproduce on a friend's machine if you ever need to. If
you'd rather run cloudflared on the host:

```bash
cloudflared tunnel --token "$TUNNEL_TOKEN" run synthex-hub
```

…and remove the `cloudflared:` service from `docker-compose.yml`.

## Keeping a Mac laptop online (the elephant)

If you're running this on your daily-driver MacBook, sleep will kill
it. Mitigations, in increasing order of robustness:

1. `caffeinate -dis -w $(pgrep -f docker | head -1)` — keeps the Mac
   awake while Docker is alive (lid open).
2. **Settings → Battery → Options → Prevent automatic sleeping when
   the display is off** + plug in. Closes-lid-and-still-runs only if
   you have an external display attached.
3. Mac mini ($600) or a small Linux box. Set-and-forget.
4. Move the stack to a $5/mo VPS (see `../vps-caddy/`).

## What's actually running

```
                                 ┌──────────────────┐
   workers worldwide ──HTTPS──►  │  Cloudflare edge │  ──QUIC tunnel──►
                                 └──────────────────┘                  │
                                                                       ▼
                                                  ┌────────────────────────┐
                                                  │ cloudflared (in compose)│
                                                  └─────────┬──────────────┘
                                                            │ HTTP :4000
                                                            ▼
                                                  ┌────────────────────────┐
                                                  │ server (mix release)    │
                                                  │  ↳ Bandit               │
                                                  │  ↳ Oban                 │
                                                  └─────────┬──────────────┘
                                                            │ TCP :5432
                                                            ▼
                                                  ┌────────────────────────┐
                                                  │ postgres (volume-backed)│
                                                  └────────────────────────┘
```
