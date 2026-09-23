# Security policy

## Reporting a vulnerability

Use GitHub's private vulnerability reporting: the **Security** tab → **Report a
vulnerability**. Please do not open a public issue for anything security-shaped,
even if you think it is minor — blast radius is easy to misjudge and impossible
to un-publish.

What makes triage fast:

- what you did (the request, the config, the command),
- what you could reach that you should not have been able to,
- whether it needs a non-default `config/plans.yaml` to reproduce.

You will get an acknowledgement within a few days, a verdict when the
investigation is done, and credit in the fix commit unless you would rather
not.

## The trust boundary, stated plainly

- **The gateway (`:4000`) trusts anyone holding `LITELLM_MASTER_KEY`.** It is
  one static bearer key over plain HTTP in the stock config, with no per-client
  keys set up. Everything behind it — every plan, every credential path — is
  one boundary. Do not publish the port to a network whose users you do not
  trust.
- **The gateway and the portal bind to `127.0.0.1` by default.** A bare
  `"${GATEWAY_PORT:-4000}:4000"` in `docker-compose.yml` would bind `0.0.0.0`,
  exposing the master-key-protected gateway and the unauthenticated portal to
  every host on the network the moment an operator runs `docker compose up
  -d` on a LAN, VPS or café Wi-Fi (issue #118). The fix: every published
  port now reads `"${BIND_ADDR:-127.0.0.1}:${PORT}:PORT"`, so a fresh clone
  comes up loopback-only. Set `BIND_ADDR=0.0.0.0` in `.env` to expose on
  purpose; the default is the safe one.
- **The portal (`:4001`) has no authentication at all.** It shows your quota
  headroom, monthly costs and plan inventory, and it can store a vendor
  session cookie pasted into it. The loopback default above already
  mitigates this on a single-host setup; on a shared or multi-host network,
  firewall `:4001` separately or set `BIND_ADDR` only for the hosts you
  control.
- **The sidecars and the token proxy publish no host ports.** They are
  reachable only on the compose network. A neighbouring host cannot get at
  them; a compromised container on that network is a different story.
- **`secrets/` holds live OAuth stores.** The sidecar CLIs rotate tokens in
  place, so read access to these directories is read access to your
  subscriptions. Treat them like `.env` — which is why both are gitignored.

## Known trade-offs (not vulnerabilities)

- Pasting a vendor session cookie into `config/plans.yaml` for a quota probe
  is as powerful as being logged in to that vendor. The portal stores it
  fingerprinted and never echoes it back, but the cookie itself is live
  access. The trade is deliberate and documented in the README's provenance
  section; deleting that plan's `probe:` block is the supported way to opt
  out.
- A vendor rewording its rate-limit message can turn a 429 into a 502 on a
  lane. That is availability, not confidentiality — the README asks you to
  verify the mapping once per sidecar.
- Reports on whether a vendor consented to a given usage surface are welcome,
  but read the provenance section first: the line drawn there is deliberate,
  and the reasoning is written down.

## Supported versions

Only `main`. This is a small self-hosted deployment: fixes land as commits,
and you pick them up by pulling and rebuilding the images that baked the
changed code — `docker compose build gateway portal claude-max-sidecar
xai-token-proxy`, then `up -d`. A bare `restart` re-runs the old code.
