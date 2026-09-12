# Circa setup

Phase 0 is about getting data accumulating. Every day without a collector is a
day of history you cannot recover, and the legacy Fitbit Web API is being turned
down in September 2026 — so this is the urgent part.

Roughly 20 minutes of clicking, then it runs itself.

---

## 1. Google Cloud project

1. Go to <https://console.cloud.google.com/projectcreate> and create a project.
   Name it whatever you like (`circa` is fine).
2. Enable the two APIs Circa needs:
   - **Google Health API** — <https://console.cloud.google.com/apis/library/health.googleapis.com>
   - **Google Calendar API** — <https://console.cloud.google.com/apis/library/calendar-json.googleapis.com>

> **Cost check.** These are consumer APIs and should be free — the Fitbit Web
> API they replace was. Google publishes no pricing page for the Health API, and
> web searches for one return the *Cloud Healthcare API*, which is a completely
> different, paid product. Do not conflate them. If the console asks you to
> attach a billing account merely to enable the Health API, stop and check
> before continuing — that would change the cost assumptions this project is
> built on.

## 2. OAuth consent screen — the 7-day trap

<https://console.cloud.google.com/apis/credentials/consent>

1. User type: **External**.
2. Fill in app name (`Circa`), your email for both support and developer contact.
3. Add yourself as a test user.
4. **Publish the app to "In production."**

That last step is not optional and is the single most common way this setup
silently fails. A consent screen left in **Testing** issues refresh tokens that
**expire after 7 days** — so collection works perfectly for a week and then dies.
Published apps issue refresh tokens that last until revoked or unused for ~6
months.

Publishing does **not** require Google's verification review. The app stays
"unverified," capped at 100 users, and you click through a warning screen once.
Verification (with its $500–$4,500 third-party CASA assessment) only becomes
relevant above 100 users, which will never happen here.

## 3. OAuth client

<https://console.cloud.google.com/apis/credentials> → **Create credentials** →
**OAuth client ID**

- Application type: **Web application**
- Name: `Circa`
- Authorised redirect URI — exactly this, including the port:

  ```
  http://localhost:8721/oauth/callback
  ```

Copy the client ID and client secret.

## 4. Configure Circa

```bash
cp .env.example .env
```

Then edit `.env`:

```ini
CIRCA_GOOGLE_CLIENT_ID=...apps.googleusercontent.com
CIRCA_GOOGLE_CLIENT_SECRET=...
CIRCA_TIMEZONE=Europe/London
CIRCA_LATITUDE=51.5072
CIRCA_LONGITUDE=-0.1276
```

Latitude/longitude are a **static setting, not location tracking**. They are used
only to compute solar elevation for the light proxy — Circa never reads your
position. Set them to your home city; being off by a few miles is irrelevant at
circadian resolution.

## 5. Authorise and verify

```bash
uv run circa init
uv run circa auth
```

A browser opens. Google will warn the app is unverified — choose **Advanced** →
**Go to Circa (unsafe)**. That warning is what "published but unverified" looks
like; it is expected.

**You will be asked to authorise twice.** That is deliberate, not a bug: the
Google Health API rejects any access token that also carries Calendar scopes —

```
403 PERMISSION_DENIED  DISALLOWED_OAUTH_SCOPES
disallowed_scopes: cl_app_created,cl_readonly
```

— so Health and Calendar are granted separately and their tokens stored apart.
Circa also deliberately does **not** use incremental authorisation
(`include_granted_scopes`), which would merge the two scope sets back onto one
token and break every Health request.

Circa requests these scopes:

| Scope | Why |
|---|---|
| `googlehealth.sleep.readonly` | Sleep sessions and stages |
| `googlehealth.health_metrics_and_measurements.readonly` | HR, HRV, SpO₂, temperature, respiratory rate |
| `googlehealth.activity_and_fitness.readonly` | Steps, exercise |
| `calendar.app.created` | Create and manage **only** the calendars Circa itself creates |
| `calendar.readonly` | Read your existing schedule, to tell a forced wake from a natural one |

Note the calendar split: Circa deliberately does **not** request blanket
`auth/calendar`. It physically cannot modify or delete your real appointments —
only the `Circa · …` calendars it made. If Google rejects the narrower scope for
your client, `circa auth --broad-calendar` falls back, but try the default first.

Then confirm what the API actually returns:

```bash
uv run circa probe --out probe.json
```

This fetches one small page of every data type and reports its real shape. It
matters because Google describes v4 as "actively evolving" and its docs
contradict themselves in places — for instance, skin temperature is listed both
as a standalone type and as a daily-only derivation. **Check these three things
in the output:**

1. **Heart rate point count** — expect thousands of points per day (~5-second
   resolution). If it is far lower, the HR phase channel needs rethinking.
2. **Which types show `filter: rejected`** — those fall back to client-side
   filtering, which is slower but still correct.
3. **`daily-sleep-temperature-derivations`** — confirm it really is one value per
   night. If it turns out to be a time series, that would restore a phase
   channel the current design writes off.

## 6. First sync

```bash
uv run circa sync --force
uv run circa doctor
```

`doctor` reports your confidence tier. Under 7 nights you are in **tier 0 (cold
start)** — that is expected and the system is designed to be honest about it
rather than pretend otherwise.

## 7. Request a Google Takeout export — do this today

<https://takeout.google.com/> → deselect all → select **Fitbit** → JSON format.

Takeout is the only practical route to per-minute historical heart rate from
before Circa started collecting. It takes hours to days to generate, so start it
now; `circa` will import it once it arrives.

## 8. Deploy to the free VM

See [`deploy/README.md`](../deploy/README.md) for the GCP e2-micro Always Free
setup. Short version: `us-west1`/`us-central1`/`us-east1`, standard persistent
disk (not SSD — balanced and SSD disks are not free), 2 GB swap because numpy on
1 GB RAM is tight.

---

## Troubleshooting

**`invalid_grant` after about a week** — the consent screen is still in
"Testing." Publish it (step 2), then re-run `circa auth`.

**403 from the Health API** — the API is not enabled on the project, a scope was
not granted, or the account is a legacy Fitbit account rather than a Google
account. Legacy Fitbit accounts cannot use the Google Health API at all.

**"Google did not return a refresh token"** — the app was already authorised.
Revoke Circa at <https://myaccount.google.com/permissions> and run `circa auth`
again.

**No data despite a successful sync** — check the watch actually synced. Data
only reaches the API after the Fitbit syncs to your phone. `circa doctor` shows
the last device sync time; open the Google Health app with the watch in range.
