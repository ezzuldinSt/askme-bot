#!/usr/bin/env bash
# Install (or update) the askme-bot systemd service and start it now.
# The bot then runs in the background indefinitely and restarts on crashes
# and across OS reboots. First-time OTP login must be done in the foreground
# (`cargo run`) BEFORE installing — systemd services have no TTY for the OTP
# prompt. The cached token in .token.json is reused afterwards.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Building release binary"
cargo build --release

if [[ ! -f .token.json ]]; then
    echo "!! No .token.json found. Run 'cargo run' once in the foreground to"
    echo "!! complete the OTP login, then re-run this script."
    exit 1
fi

UNIT_DST="/etc/systemd/system/askme-bot.service"
echo "==> Installing systemd unit to $UNIT_DST"
install -m 644 scripts/askme-bot.service "$UNIT_DST"

systemctl daemon-reload
systemctl enable --now askme-bot

echo
echo "==> askme-bot is now running as a service (enabled at boot)"
systemctl --no-pager --lines=5 status askme-bot || true
echo
echo "Useful commands:"
echo "  journalctl -u askme-bot -f     # follow logs"
echo "  systemctl restart askme-bot    # restart"
echo "  systemctl stop askme-bot       # stop"
echo "The admin panel URL is printed at startup (see journalctl) — default http://localhost:1330"
