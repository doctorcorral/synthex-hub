# Synthex Hub on Fly.io + Neon Postgres

Set-and-forget deploy at `synthex.fit` for ~$5/mo. ~15 minutes from
zero. The only thing this costs you that the laptop+Cloudflare path
doesn't is a single shared-cpu Fly machine; the Postgres tier is
free up to 0.5 GB, which we'll never approach.

## Why this is the right pick for most cases

- **Always-on**, no laptop sleep / power / battery / IP-change drama.
- **Native Elixir support**: `release_command` runs migrations, the
  Fly proxy speaks HTTP/2 and HTTPS for you, `fly logs` and
  `fly ssh console` "just work".
- **Same Docker image** as `deploy/cloudflared/` — no separate build.
- **Neon for Postgres** is logically managed, branchable, and free
  for our footprint.

## Prerequisites

- A Fly.io account (one signup; free machine credits cover small apps).
- A Neon account (free tier).
- The `flyctl` CLI: `brew install flyctl` then `fly auth login`.
- Your `synthex.fit` DNS controllable somewhere (we'll add a CNAME
  + AAAA record).

## 1. Provision Postgres on Neon (~2 min)

1. <https://console.neon.tech> → Create project → name it
   `synthex-hub`, region close to your Fly primary (we'll use IAD =
   us-east-1; Neon's `aws-us-east-2` works fine).
2. Copy the **pooled** connection string (looks like
   `postgres://...neon.tech/neondb?sslmode=require`). Pooled is
   important so Fly machines don't exhaust direct connections after
   each Oban job.

## 2. Launch the Fly app (~5 min)

The Fly config (`fly.toml`) lives at [`../../server/fly.toml`](../../server/fly.toml)
because Fly's Dockerfile + build-context resolution wants `fly.toml`
next to the mix project. The walkthrough you're reading now is just
documentation; all `fly` commands run from `server/`.

```bash
cd ../../server     # i.e. synthex-hub/server, where fly.toml lives
```

If this is your first ever Fly app you'll do a one-time
`fly auth login`, then:

```bash
# Picks up fly.toml; doesn't actually deploy yet.
fly launch --no-deploy --copy-config
```

`fly launch` will see the existing `fly.toml` and ask whether to
copy it. Say yes. If the global app name `synthex-hub` is taken,
pass `--name <something-else>` and update the README's curl
examples accordingly.

Set the secrets:

```bash
TOKEN=$(openssl rand -hex 32)
echo "API_TOKEN: $TOKEN  (save this in your password manager)"

fly secrets set \
  DATABASE_URL='postgres://...neon.tech/neondb?sslmode=require' \
  API_TOKEN="$TOKEN"
```

Deploy:

```bash
fly deploy
```

The first deploy:

1. Pushes the build context to a Fly remote builder.
2. Builds the multi-stage Dockerfile.
3. Spins up a one-off Machine and runs `release_command` →
   `Server.Release.migrate()`. If that fails, the deploy aborts.
4. Rolls out the new Machine; the old one drains.
5. Healthcheck on `/health` flips green; you're live.

Total: ~3 minutes. Subsequent deploys are 60–90 s.

```bash
fly status                          # is it running?
curl -s https://synthex-hub.fly.dev/health
fly logs                            # tail logs
```

## 3. Point synthex.fit at the app (~5 min)

```bash
fly certs create synthex.fit
fly certs show synthex.fit
```

`fly certs show` prints the DNS records you need. Typical output:

```
Hostname               synthex.fit
DNS Provider           ...
Certificate Authority  Let's Encrypt

Add an A or AAAA record (or CNAME if root-domain-CNAME is supported)
pointing at:
  A     66.241.124.84
  AAAA  2a09:8280:1::abcd
Add an _acme-challenge.synthex.fit CNAME pointing at:
  synthex-hub.fly.dev
```

In your DNS host (Cloudflare DNS works fine — keep "DNS only", no
proxy), set:

```
synthex.fit                  A      66.241.124.84
synthex.fit                  AAAA   2a09:8280:1::abcd
_acme-challenge.synthex.fit  CNAME  synthex-hub.fly.dev
```

`fly certs check synthex.fit` validates and issues the cert; takes
~1 minute. Then:

```bash
curl -s https://synthex.fit/health
curl -s https://synthex.fit/api/status \
  -H "Authorization: Bearer $YOUR_TOKEN" | jq
```

## 4. Hand a worker invocation to friends

Identical to the cloudflared setup:

```bash
docker run --rm \
  -e SERVER_URL=https://synthex.fit/api \
  -e API_TOKEN=<your shared token> \
  -e WORKER_NAME=$(hostname) \
  -e POOL_SIZE=8 \
  doctorcorral/synthex-worker:latest
```

## Day-to-day ops

```bash
fly logs                         # tail
fly logs -i <machine-id>         # one machine
fly ssh console                  # bash on the running container
fly ssh console -C "/app/bin/server remote"   # IEx into the BEAM
fly status                       # health, machines, regions
fly scale count 2                # if you ever want HA
fly scale memory 1024            # bump RAM
fly secrets set FOO=bar          # rotate secrets; triggers redeploy
fly deploy                       # ship a new version
fly releases                     # history; can rollback
fly releases rollback v17        # ↑
```

## Cost (rough, sanity-check the current pricing page)

| Component | Tier | ~$/mo |
|---|---|---|
| Fly shared-cpu-1x, 512 MB | always-on | a few dollars; Fly's free monthly credit usually covers single small apps |
| Neon Postgres | Free tier (0.5 GB) | $0 |
| Fly TLS / cert | included | $0 |
| Bandwidth | first 160 GB/mo free, then ~$0.02/GB | effectively $0 |

For one CEGAR run on Humanoid your egress is well under 1 GB.
Realistic monthly bill: **$0–$10**, dominated by whether you've
upgraded your Fly account beyond the free credit allowance.

## When *not* to pick Fly

- You want zero recurring bills and your laptop is always on →
  `../cloudflared/`.
- You want full control of the ingress (mTLS, geo-allowlists, raw
  TCP, gRPC streaming with multi-hour deadlines) → `../vps-caddy/`.
- You're paranoid about a third party seeing your Postgres
  connection string. (Note: it's encrypted in `fly secrets`; only
  your CLI plus the Machines themselves can read it.)

## Migrating later

The data model lives entirely in Postgres. Switching from Fly to
the cloudflared stack (or to a VPS) is just:

1. `pg_dump` from Neon
2. `docker compose up -d postgres` somewhere else
3. `pg_restore`
4. `fly destroy synthex-hub`

There's nothing Fly-specific in the codebase, only in `fly.toml`.
