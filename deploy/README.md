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
| 1 GB RAM | Measured: a full pipeline run peaks at ~270 MB and the service idles near 100 MB. The 2 GB swapfile the setup script adds is a safety net, not a requirement |

### The external IP, which is the part everyone gets wrong

External IPv4 addresses attached to a running VM have been billable since
February 2024, and the Always Free compute list does not mention them — which
reads like $3.65/month on top of a "free" VM. It is not, quite. From Google's
own billing catalogue (SKU `C054-7F72-A02E`, *External IP Charge on a Standard
VM*, global):

```
from      0 hours: $0.000000
from    720 hours: $0.005000
```

720 hours is exactly one always-on address for a 30-day month. So one VM with
one IP is free in 30-day months and costs **$0.12 in 31-day months** (24 hours
over the tier at half a cent each). February is well under. This is the only
line that will ever appear — and if even that bothers you, see
[dropping the external IP](#going-from-012-to-000-drop-the-external-ip),
which takes it to exactly zero.

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

The repository is public, so the VM can fetch the script itself. Check it is the
one you expect before handing it to root:

```bash
gcloud compute ssh circa --zone=us-central1-a --command='
  curl -fsSL -o /tmp/setup-vm.sh https://raw.githubusercontent.com/kmandve/Circa/main/deploy/setup-vm.sh
  sha256sum /tmp/setup-vm.sh'
shasum -a 256 deploy/setup-vm.sh        # compare, then:
gcloud compute ssh circa --zone=us-central1-a --command='sudo bash /tmp/setup-vm.sh'
```

The script creates a service user, a 2 GB swapfile, a Python 3.12 venv via `uv`,
`/etc/circa/circa.env` with a generated encryption key, a hardened systemd unit
and log rotation. It clones the repo to `/opt/circa` and installs `.[science]` —
the modelling stack is an optional dependency group, and a plain `-e .` gives
you a collector that fills the database and never estimates anything.

Then fill in `/etc/circa/circa.env`: OAuth client id and secret, and your real
`CIRCA_TIMEZONE` / `CIRCA_LATITUDE` / `CIRCA_LONGITUDE`. `circa doctor` warns
while the coordinates are still the example ones, because the light proxy
clamps to clear-sky irradiance at that latitude and being in the wrong place is
wrong quietly.

## Moving an existing install, instead of starting over

If you already have Circa running somewhere with history worth keeping, copy the
database rather than re-authorising — the refresh tokens travel with it.

```bash
# On the old machine. Use .backup, not cp: WAL mode means a plain copy can tear.
python -c "
import sqlite3, pathlib
s = sqlite3.connect(f'file:{pathlib.Path.home()}/.circa/circa.db?mode=ro', uri=True)
d = sqlite3.connect('/tmp/circa.db'); s.backup(d); d.execute('VACUUM'); d.close()"
gzip -9 /tmp/circa.db
gcloud compute scp /tmp/circa.db.gz circa:/tmp/ --zone=us-central1-a
```

Then on the VM, **set `CIRCA_SECRET_KEY` in `/etc/circa/circa.env` to the
contents of the old machine's `~/.circa/secret.key`** before starting anything.
The tokens in that database are Fernet-encrypted with it; the setup script
generated a fresh key, and with the wrong one every credential in the copied
database is unreadable and you are back to re-authorising.

```bash
sudo systemctl stop circa
gunzip /tmp/circa.db.gz
sudo install -o circa -g circa -m 640 /tmp/circa.db /var/lib/circa/circa.db
sudo rm -f /var/lib/circa/circa.db-wal /var/lib/circa/circa.db-shm
sudo -u circa /opt/circa/.venv/bin/circa doctor     # should be all green
sudo systemctl start circa
```

Uploading to GCP is ingress and free. Pulling a backup *down* is egress and
spends against the 1 GB monthly allowance — a 150 MB database gzips to about
25 MB, so occasional backups are fine and nightly ones are not.

Stop the old collector before starting this one. Two collectors with separate
databases will both push to the same calendars and fight over every event.

## Authorise

Only needed on a fresh install — skip it if you copied a database across.

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

## Going from $0.12 to $0.00: drop the external IP

The 12 cents is the IP. Remove it and there is nothing left to bill. Circa only
ever contacts `health.googleapis.com`, `oauth2.googleapis.com` and
`www.googleapis.com` at runtime — `accounts.google.com` appears in the source
but only as a URL handed to your browser, never fetched by the VM — so
**Private Google Access** covers the whole of normal operation, for free.

Do these in order. Removing the IP before IAP works will lock you out.

```bash
# 1. Let the subnet reach Google APIs without an external address.
gcloud compute networks subnets update default --region=us-central1 \
  --enable-private-ip-google-access

# 2. Allow SSH from IAP's forwarding range.
gcloud compute firewall-rules create allow-ssh-from-iap \
  --network=default --direction=INGRESS --action=allow \
  --rules=tcp:22 --source-ranges=35.235.240.0/20

# 3. Prove it works *while you still have the IP as a fallback*.
gcloud compute ssh circa --zone=us-central1-a --tunnel-through-iap \
  --command='echo ok'

# 4. Only now, remove the address.
gcloud compute instances delete-access-config circa \
  --zone=us-central1-a --access-config-name="external-nat"

# 5. And drop the rules the default VPC opens to the whole internet.
gcloud compute firewall-rules delete default-allow-ssh default-allow-rdp

# 6. Detach the default Compute service account, which carries roles/editor.
#    Circa authenticates to Google with its own OAuth client and never calls a
#    GCP API, so that identity buys nothing and is pure blast radius: anything
#    running on the box could mint a project-editor token from the metadata
#    server. Needs a stop/start; nothing is lost, since data is fetched by
#    watermark on the way back up.
gcloud compute instances stop circa --zone=us-central1-a
gcloud compute instances set-service-account circa --zone=us-central1-a \
  --no-service-account --no-scopes
gcloud compute instances start circa --zone=us-central1-a
```

The trade: `apt`, PyPI and GitHub are unreachable while it runs that way, so
the VM gets no security updates on its own. Lend it an address when you want to
update, and take it back afterwards — an hour costs half a cent, and only in a
month that already passed 720 hours:

```bash
gcloud compute instances add-access-config circa --zone=us-central1-a \
  --access-config-name="external-nat" --network-tier=STANDARD
# ... apt upgrade, git pull, uv pip install -e ".[science]", systemctl restart ...
gcloud compute instances delete-access-config circa --zone=us-central1-a \
  --access-config-name="external-nat"
```

Worth doing monthly. An internet-unreachable box is a much smaller target than
an exposed one, but an unpatched kernel is still an unpatched kernel.

## Reaching the dashboard

```bash
gcloud compute ssh circa --zone=us-central1-a --tunnel-through-iap \
  -- -N -L 8720:localhost:8720
# then open http://localhost:8720
```

Forwarded over SSH rather than tunnelling 8720 with IAP directly: that would
need the firewall opened to the entire IAP range on 8720, whereas this needs
only port 22 and puts SSH authentication on top of IAP's.

The web app binds to `127.0.0.1` on purpose. It has no authentication and the
database holds detailed sleep and cardiovascular data — reach it through the
tunnel, never by opening a firewall port.

## Who can reach any of this

Worth being able to answer precisely, since it is health data.

| | |
|---|---|
| The VM | No external IP, no inbound firewall rule except SSH from IAP's range. Reaching it requires an IAM identity on the project *and* the SSH key registered against it. |
| The project | One human owner. The rest of the IAM policy is Google's own service agents. |
| The dashboard | Bound to `127.0.0.1`, so it exists only inside the machine. The tunnel is the only path. |
| The database | One file, `0640 circa:circa`, on a disk encrypted at rest with a Google-managed key. |
| Refresh tokens | Fernet-encrypted inside that database, with the key in `/etc/circa/circa.env` (`0640 root:circa`), not in the database itself. |
| Google Calendar | Circa holds `calendar.app.created`, which cannot read or write any calendar it did not create. It never calls the ACL API, so the calendars it makes stay private to your account. Verified by trying: the ACL read comes back refused. |
| What leaves the machine | Requests to `health.googleapis.com`, `oauth2.googleapis.com` and `www.googleapis.com`, and events on your own calendars. Nothing else — there is nowhere else it can reach. |
| The repository | Public, and carries no data: the history was rebuilt from a single commit, and both the tree and every blob were scanned for credentials, coordinates and measurements before it was pushed. |

The honest limit: Google already holds this data — it comes from their Health
API — and Google hosts the VM and manages the disk-encryption key. "Private"
here means no third party and no other person, not that it is hidden from
Google.

## Operations

```bash
sudo systemctl status circa
sudo journalctl -u circa -f
tail -f /var/log/circa/circa.log
sudo -u circa /opt/circa/.venv/bin/circa status
sudo -u circa /opt/circa/.venv/bin/circa doctor
```

Those last two read `/etc/circa/circa.env` themselves, so they see the same
configuration the service does.

To update:

```bash
cd /opt/circa && sudo -u circa git pull origin main
sudo -u circa /usr/local/bin/uv pip install -e ".[science]"
sudo systemctl restart circa
```

Reboots need no attention — the unit is enabled and the swapfile is in
`/etc/fstab`. Verified by actually resetting the instance.

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
