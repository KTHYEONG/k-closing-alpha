#!/usr/bin/env bash
# Install KCA systemd user units and enable timers (WSL boot persistence).
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/systemd" && pwd)"
DEST_DIR="${HOME}/.config/systemd/user"

mkdir -p "${DEST_DIR}"
cp "${SRC_DIR}"/kca-*.timer "${SRC_DIR}"/kca-*.service "${DEST_DIR}/"

systemctl --user daemon-reload

systemctl --user enable --now \
  kca-collect.timer \
  kca-predict.timer \
  kca-paper-exit.timer \
  kca-finalize-close.timer \
  kca-archive-intraday.timer \
  kca-price-ingest.timer \
  kca-daily-audit.timer \
  kca-backup.timer \
  kca-backup-prune.timer

# Keep user timers alive without an active login session (WSL).
loginctl enable-linger "${USER}"
