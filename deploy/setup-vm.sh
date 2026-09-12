#!/usr/bin/env bash
# Provision a GCP e2-micro Always Free VM to run Circa.
#
# Assumes Debian 12/13. Run as root on a freshly created instance:
#   sudo bash setup-vm.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/kmandve/Circa.git}"   # override to deploy a fork
INSTALL_DIR=/opt/circa
DATA_DIR=/var/lib/circa
LOG_DIR=/var/log/circa
ENV_FILE=/etc/circa/circa.env

echo "==> Packages"
apt-get update -qq
apt-get install -y -qq curl git ca-certificates sqlite3 >/dev/null

# e2-micro has 1 GB RAM. numpy/scipy in the modelling phases will not fit
# reliably without swap, and the Always Free disk allowance (30 GB) has room
# to spare.
if ! swapon --show | grep -q /swapfile; then
  echo "==> 2 GB swapfile"
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  # Swap sparingly - it is there as a safety net, not as working memory.
  sysctl -w vm.swappiness=10 >/dev/null
  grep -q 'vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf
fi

echo "==> Service account"
id -u circa >/dev/null 2>&1 || useradd --system --home "$INSTALL_DIR" --shell /usr/sbin/nologin circa
mkdir -p "$INSTALL_DIR" "$DATA_DIR" "$LOG_DIR" /etc/circa
chown -R circa:circa "$INSTALL_DIR" "$DATA_DIR" "$LOG_DIR"
chmod 750 "$DATA_DIR"

if [[ -n "$REPO_URL" && ! -d "$INSTALL_DIR/.git" ]]; then
  echo "==> Clone"
  sudo -u circa git clone "$REPO_URL" "$INSTALL_DIR"
fi

echo "==> Python toolchain (uv)"
export UV_INSTALL_DIR=/usr/local/bin
curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh >/dev/null
cd "$INSTALL_DIR"
sudo -u circa /usr/local/bin/uv venv --python 3.12
sudo -u circa /usr/local/bin/uv pip install -e .

if [[ ! -f "$ENV_FILE" ]]; then
  echo "==> Environment file"
  SECRET=$(sudo -u circa "$INSTALL_DIR/.venv/bin/python" -c \
    "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
  cat > "$ENV_FILE" <<ENVEOF
CIRCA_DATA_DIR=$DATA_DIR
CIRCA_SECRET_KEY=$SECRET
CIRCA_WEB_HOST=127.0.0.1
CIRCA_WEB_PORT=8720

# Fill these in before starting the service - see docs/SETUP.md.
# Latitude and longitude are used only for the solar-elevation ceiling in the
# light proxy. Your nearest city is precise enough; Circa never reads a device
# location. `circa doctor` refuses to pass while these are still the example.
CIRCA_GOOGLE_CLIENT_ID=
CIRCA_GOOGLE_CLIENT_SECRET=
CIRCA_TIMEZONE=Europe/London
CIRCA_LATITUDE=51.5072
CIRCA_LONGITUDE=-0.1276
ENVEOF
  chmod 640 "$ENV_FILE"
  chown root:circa "$ENV_FILE"
fi

echo "==> systemd"
install -m 644 "$INSTALL_DIR/deploy/circa.service" /etc/systemd/system/circa.service
systemctl daemon-reload
systemctl enable circa >/dev/null

cat > /etc/logrotate.d/circa <<'LOGEOF'
/var/log/circa/*.log {
  weekly
  rotate 8
  compress
  missingok
  notifempty
  copytruncate
}
LOGEOF

cat <<'DONE'

==> Provisioned.

Next:
  1. Edit /etc/circa/circa.env and add your OAuth client id/secret.
  2. Authorise. The redirect goes to localhost, so forward the port from your
     own machine:

       ssh -L 8721:localhost:8721 <vm>
       sudo -u circa /opt/circa/.venv/bin/circa auth --no-browser

     then open the printed URL in your laptop's browser.
  3. systemctl start circa
  4. Watch the dashboard the same way:  ssh -L 8720:localhost:8720 <vm>
     then open http://localhost:8720

The web app binds to 127.0.0.1 deliberately - there is no authentication on it,
and this database holds detailed health data. Reach it over an SSH tunnel rather
than opening a firewall port.
DONE
