# Deploying Circa to a free GCP VM

Target: **e2-micro Always Free**. $0/month, always on, and it lives in the same
Google account as the OAuth client.

## Why this shape

The plan originally called for a paid host plus managed Postgres. Two findings
during Phase 0 made that unnecessary:

1. **Circa polls; it does not use webhooks.** Google Health webhooks are
   project-level, need a service account, a public HTTPS endpoint and ECDSA
   P-256 signature verification — a lot of surface area for one user. Polling
   means no inbound traffic, so no public endpoint and no TLS certificate.
2. **Intermittent uptime is safe.** Data lives in Google's cloud and is fetched
   by watermark, clamped to the device's `lastSyncTime`. A collector that is
   down for three days backfills three days on restart. Only calendar freshness
   degrades, and the 48-hour write horizon covers that.

So the requirements collapse to: a machine that can make outbound HTTPS calls
and hold ~50 MB/year of SQLite.

## Free-tier constraints that actually bite

| Constraint | Consequence |
|---|---|
| Region must be `us-west1`, `us-central1` or `us-east1` | Anywhere else is billed |
| Machine type must be `e2-micro` | e2-small is billed |
| Disk must be **standard** persistent disk, ≤30 GB | Balanced and SSD disks are **not** free — this is the most common accidental charge |
| Network tier must be **Standard** | Premium tier egress is billed |
| 1 GB/month North America egress | Circa's egress is a few calendar writes; never close to the cap |
| 1 GB RAM | numpy/scipy is tight — the setup script adds 2 GB of swap |

A billing account with a card must be on file even for Always Free. Set a
**$1 budget alert** at <https://console.cloud.google.com/billing/budgets> so any
accidental drift out of the free tier is immediately visible.

## Create the instance

```bash
gcloud compute instances create circa \
  --zone=us-central1-a \
  --machine-type=e2-micro \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-type=pd-standard \
  --boot-disk-size=30GB \
  --network-interface=network-tier=STANDARD,stack-type=IPV4_ONLY \
  --metadata=enable-oslogin=TRUE
```

Note `pd-standard` and `network-tier=STANDARD` — the console defaults to
`pd-balanced` and Premium, both of which are billed.

## Provision

```bash
gcloud compute scp --zone=us-central1-a --recurse . circa:/tmp/circa-src
gcloud compute ssh circa --zone=us-central1-a
sudo mv /tmp/circa-src /opt/circa && sudo bash /opt/circa/deploy/setup-vm.sh
```

The script creates a service user, a 2 GB swapfile, a Python 3.12 venv via `uv`,
`/etc/circa/circa.env` with a generated encryption key, a hardened systemd unit
and log rotation.

## Authorise

The OAuth redirect points at `localhost`, so forward the port from your own
machine rather than exposing anything:

```bash
gcloud compute ssh circa --zone=us-central1-a -- -L 8721:localhost:8721
sudo -u circa /opt/circa/.venv/bin/circa auth --no-browser
```

Open the printed URL in your laptop's browser. Then:

```bash
sudo systemctl start circa
sudo -u circa /opt/circa/.venv/bin/circa doctor
```

## Reaching the dashboard

```bash
gcloud compute ssh circa --zone=us-central1-a -- -L 8720:localhost:8720
# then open http://localhost:8720
```

The web app binds to `127.0.0.1` on purpose. It has no authentication and the
database holds detailed sleep and cardiovascular data — reach it through an SSH
tunnel, not an open firewall port.

## Operations

```bash
sudo systemctl status circa
sudo journalctl -u circa -f
tail -f /var/log/circa/circa.log
sudo -u circa /opt/circa/.venv/bin/circa status
```

Backups — the whole state is one SQLite file:

```bash
sudo -u circa sqlite3 /var/lib/circa/circa.db ".backup '/tmp/circa-backup.db'"
gcloud compute scp --zone=us-central1-a circa:/tmp/circa-backup.db ./
```

Use `.backup` rather than copying the file: the database runs in WAL mode, so a
plain `cp` can capture a torn state.

## If the free tier ever stops being free

The whole deployment is a venv, a SQLite file and a systemd unit, so moving is
copying two files. Oracle Cloud Always Free (2 OCPU/12 GB ARM, needs a keepalive
because it reclaims instances under 10% CPU over 7 days) and Hetzner (~€3.79/mo)
are the realistic alternatives.
