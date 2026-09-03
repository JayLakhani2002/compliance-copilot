#!/usr/bin/env bash
# deploy/deploy.sh — automates the server-side half of docs/DEPLOY.md
# (ADR-0033): first-login hardening + Docker install + repo clone + compose
# up. Targets Ubuntu 24.04 LTS (the image docs/DEPLOY.md provisions).
#
# HONESTY NOTE: this script has been dry-validated only — `bash -n` (syntax)
# and manual read-through against docs.docker.com (fetched 2026-09-03) — no
# real VPS exists yet to run it against (ADR-0033). Re-verify the Docker
# install commands "on the day" if docs.docker.com has changed since.
#
# Idempotent: every step checks current state before acting, so re-running
# this after a partial failure (or a later `git pull` of this repo) is safe.
# Run as a user with sudo (root is fine for first login). No secrets live in
# this file — it expects the operator to have already placed a real `.env`
# next to `docker-compose.prod.yml` in the cloned repo (see docs/DEPLOY.md
# step 4); this script does not create, fetch, or print one.
#
# Usage: HOSTNAME=compliance-copilot.example.com \
#        REPO_URL=https://github.com/<you>/<repo>.git \
#        APP_USER=deploy \
#        sudo -E bash deploy/deploy.sh
set -euo pipefail

HOSTNAME="${HOSTNAME:?Set HOSTNAME, e.g. compliance-copilot.example.com}"
REPO_URL="${REPO_URL:?Set REPO_URL, e.g. https://github.com/you/compliance-copilot.git}"
APP_USER="${APP_USER:-deploy}"
REPO_DIR="/home/${APP_USER}/compliance-copilot"

log() { printf '\n=== %s ===\n' "$1"; }

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root (or via sudo -E) — hardening steps need it." >&2
  exit 1
fi

# --- 1. Non-root sudo user (idempotent: skip if it already exists) --------
log "Non-root sudo user: ${APP_USER}"
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  adduser --disabled-password --gecos "" "$APP_USER"
  usermod -aG sudo "$APP_USER"
  mkdir -p "/home/${APP_USER}/.ssh"
  if [[ -f /root/.ssh/authorized_keys ]]; then
    cp /root/.ssh/authorized_keys "/home/${APP_USER}/.ssh/authorized_keys"
  fi
  chown -R "${APP_USER}:${APP_USER}" "/home/${APP_USER}/.ssh"
  chmod 700 "/home/${APP_USER}/.ssh"
  chmod 600 "/home/${APP_USER}/.ssh/authorized_keys" 2>/dev/null || true
else
  echo "user ${APP_USER} already exists, skipping"
fi

# --- 2. ufw: 22/80/443 only ------------------------------------------------
log "ufw firewall (22/80/443)"
apt-get update -y
apt-get install -y ufw
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# --- 3. unattended-upgrades ------------------------------------------------
log "unattended-upgrades"
apt-get install -y unattended-upgrades
echo 'Unattended-Upgrade::Allowed-Origins {"${distro_id}:${distro_codename}-security";};' \
  > /etc/apt/apt.conf.d/51unattended-upgrades-security-only
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF
systemctl enable --now unattended-upgrades

# NOTE (fail2ban, optional per docs/DEPLOY.md): not installed by this
# script — a single-key API with no SSH password auth has a much smaller
# brute-force surface than a typical VPS. Add `apt-get install -y fail2ban`
# here if real SSH scan traffic in the logs ever justifies it.

# --- 4. Docker + compose plugin (official apt repo) ------------------------
log "Docker engine + compose plugin"
if ! command -v docker >/dev/null 2>&1; then
  apt-get install -y ca-certificates curl
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
else
  echo "docker already installed, skipping"
fi
usermod -aG docker "$APP_USER"

# --- 5. Clone (or update) the repo -----------------------------------------
log "Repo: ${REPO_URL} -> ${REPO_DIR}"
if [[ -d "${REPO_DIR}/.git" ]]; then
  sudo -u "$APP_USER" git -C "$REPO_DIR" pull
else
  sudo -u "$APP_USER" git clone "$REPO_URL" "$REPO_DIR"
fi

# --- 6. Compose up (only once the operator has placed a real .env) --------
log "docker compose up -d --build"
if [[ ! -f "${REPO_DIR}/.env" ]]; then
  cat <<EOF

.env not found at ${REPO_DIR}/.env — stopping here.
Copy .env.example to .env in that directory, fill in real secrets
(see docs/DEPLOY.md step 4), set DEPLOY_HOSTNAME=${HOSTNAME}, then re-run
this script (it will pick up from here — the clone/hardening above is
already done and will be skipped).
EOF
  exit 0
fi
sudo -u "$APP_USER" DEPLOY_HOSTNAME="$HOSTNAME" \
  docker compose -f "${REPO_DIR}/docker-compose.prod.yml" --project-directory "$REPO_DIR" up -d --build

log "Done"
echo "Next: DNS A record for ${HOSTNAME} -> this server's IP (if not already set), then run the smoke tests in docs/DEPLOY.md step 5."
