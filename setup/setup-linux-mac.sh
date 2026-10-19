#!/usr/bin/env bash
# Glyph Studio - one-time setup for Linux (systemd) and macOS
#
# Same result as setup-windows.ps1:
#   https://<this-machine>.<tailnet>.ts.net/        -> the studio
#   https://<this-machine>.<tailnet>.ts.net:8443/   -> Ollama
# private to your tailnet, kept running across reboots by Tailscale.
#
# Run:
#   curl -fsSL https://raw.githubusercontent.com/bella-breaux/Glyph-Creator/main/setup/setup-linux-mac.sh | bash
# Re-run any time to update the studio or repair the setup.

set -euo pipefail

MODEL="${MODEL:-gemma4:e2b-it-qat}"
SITE_URL="https://raw.githubusercontent.com/bella-breaux/Glyph-Creator/main/index.html"
OLLAMA_PORT=11434
OLLAMA_HTTPS_PORT=8443
OS="$(uname -s)"

step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }
ok()   { printf '    \033[32m%s\033[0m\n' "$1"; }
warn() { printf '    \033[33m%s\033[0m\n' "$1"; }
fail() { printf '\n  \033[31m%s\033[0m\n' "$1"; exit 1; }

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

# ---------- 1. Ollama + Tailscale ----------
step "Checking Ollama and Tailscale"
if ! command -v ollama >/dev/null 2>&1; then
  if [ "$OS" = "Darwin" ]; then
    fail "Install Ollama from https://ollama.com/download, open it once, then run this again."
  fi
  warn "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

TS="tailscale"
if [ "$OS" = "Darwin" ] && ! command -v tailscale >/dev/null 2>&1; then
  for c in /Applications/Tailscale.app/Contents/MacOS/Tailscale /usr/local/bin/tailscale /opt/homebrew/bin/tailscale; do
    [ -x "$c" ] && TS="$c" && break
  done
fi
if ! command -v "$TS" >/dev/null 2>&1 && [ ! -x "$TS" ]; then
  if [ "$OS" = "Darwin" ]; then
    fail "Install Tailscale (the standalone download from https://tailscale.com/download/mac, not the App Store version), then run this again."
  fi
  warn "Installing Tailscale..."
  curl -fsSL https://tailscale.com/install.sh | sh
fi
ok "Ollama and Tailscale are installed."

# ---------- 2. Tailscale sign-in ----------
step "Checking Tailscale sign-in"
ts_json() { $SUDO "$TS" status --json 2>/dev/null || true; }
json_get() { python3 -c "import sys,json;d=json.load(sys.stdin);print($1)" 2>/dev/null || true; }
state="$(ts_json | json_get "d.get('BackendState','')")"
if [ "$state" != "Running" ]; then
  warn "Signing in to Tailscale. Open the link it prints and use the same account as your phone."
  $SUDO "$TS" up
  state="$(ts_json | json_get "d.get('BackendState','')")"
fi
[ "$state" = "Running" ] || fail "Tailscale isn't connected. Run 'sudo tailscale up', sign in, then run this again."
DNS="$(ts_json | json_get "d['Self']['DNSName'].rstrip('.')")"
[ -n "$DNS" ] || fail "Tailscale didn't report this machine's name. Turn on MagicDNS at https://login.tailscale.com/admin/dns."
ORIGIN="https://$DNS"
ok "This machine is $DNS"

# ---------- 3. Studio files ----------
step "Installing the studio"
SITE_DIR="$HOME/.local/share/glyph-studio/site"
mkdir -p "$SITE_DIR"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"
if [ -n "$HERE" ] && [ -f "$HERE/../index.html" ]; then
  cp "$HERE/../index.html" "$SITE_DIR/index.html"; ok "Copied the studio from this folder."
else
  curl -fsSL "$SITE_URL" -o "$SITE_DIR/index.html" || fail "Couldn't download the studio from GitHub."
  ok "Downloaded the latest studio from GitHub."
fi
chmod -R a+rX "$HOME/.local/share/glyph-studio"

# ---------- 4. Ollama settings ----------
step "Configuring Ollama"
ORIGINS="$ORIGIN,https://bella-breaux.github.io"
if [ "$OS" = "Darwin" ]; then
  launchctl setenv OLLAMA_HOST "0.0.0.0:$OLLAMA_PORT"
  launchctl setenv OLLAMA_ORIGINS "$ORIGINS"
  osascript -e 'quit app "Ollama"' 2>/dev/null || true
  sleep 2; open -a Ollama
  warn "macOS forgets launchctl settings on reboot. Re-run this script after restarting."
else
  $SUDO mkdir -p /etc/systemd/system/ollama.service.d
  printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0:%s"\nEnvironment="OLLAMA_ORIGINS=%s"\n' "$OLLAMA_PORT" "$ORIGINS" \
    | $SUDO tee /etc/systemd/system/ollama.service.d/glyph-studio.conf >/dev/null
  $SUDO systemctl daemon-reload
  $SUDO systemctl restart ollama
fi
for _ in $(seq 1 30); do curl -fs "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null && break; sleep 1; done
curl -fs "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null || fail "Ollama didn't start. Check 'systemctl status ollama'."
ok "Ollama restarted. OLLAMA_ORIGINS = $ORIGINS"

# ---------- 5. Firewall ----------
step "Blocking Ollama from your local network"
if [ "$OS" = "Linux" ] && command -v ufw >/dev/null 2>&1 && $SUDO ufw status | grep -q "Status: active"; then
  $SUDO ufw deny in proto tcp to any port "$OLLAMA_PORT" >/dev/null
  ok "ufw now blocks port $OLLAMA_PORT for other devices. Tailscale still reaches it locally."
else
  warn "No active firewall found. Block TCP $OLLAMA_PORT from your LAN so only Tailscale can use Ollama."
fi

# ---------- 6. Model ----------
step "Checking models"
if ! ollama list 2>/dev/null | tail -n +2 | grep -q .; then
  warn "No models installed. Downloading $MODEL (a few GB)..."
  ollama pull "$MODEL"
else
  ok "Installed: $(ollama list | tail -n +2 | awk '{print $1}' | paste -sd, -)"
fi

# ---------- 7. Tailscale serve ----------
step "Publishing on your tailnet"
$SUDO "$TS" serve reset >/dev/null 2>&1 || true
$SUDO "$TS" serve --bg --https=443 "$SITE_DIR" \
  || fail "Tailscale couldn't publish the site. Turn on HTTPS certificates at https://login.tailscale.com/admin/dns. (On macOS this needs the standalone Tailscale, not the App Store version.)"
$SUDO "$TS" serve --bg --https="$OLLAMA_HTTPS_PORT" "http://127.0.0.1:$OLLAMA_PORT" \
  || fail "Tailscale couldn't publish Ollama on port $OLLAMA_HTTPS_PORT."
# Optional pixel model (training/pixel_server.py). Harmless until that server is running.
$SUDO "$TS" serve --bg --https=8444 "http://127.0.0.1:11501" >/dev/null 2>&1 || true

# ---------- 8. Check ----------
step "Testing"
site_ok=0; api_ok=0
for _ in $(seq 1 20); do
  [ $site_ok = 1 ] || { curl -fs "$ORIGIN/" | grep -q "Glyph Studio" && site_ok=1; } || true
  [ $api_ok = 1 ]  || { curl -fs -H "Origin: $ORIGIN" "$ORIGIN:$OLLAMA_HTTPS_PORT/api/tags" >/dev/null && api_ok=1; } || true
  [ $site_ok = 1 ] && [ $api_ok = 1 ] && break
  sleep 3
done
[ $site_ok = 1 ] && ok "Studio is up." || warn "The studio didn't answer yet. The first HTTPS certificate can take a minute."
[ $api_ok = 1 ]  && ok "Ollama is reachable over Tailscale." || warn "Ollama didn't answer over Tailscale yet. Check 'tailscale serve status'."

printf '\n  Done. On your phone, turn Tailscale on and open:\n\n      \033[32m%s/\033[0m\n\n' "$ORIGIN"
printf '  The AI connects by itself. Keep this machine on and awake to use it.\n  To turn it all off later:  sudo tailscale serve reset\n\n'
