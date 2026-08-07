#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

mkdir -p "${SYSTEMD_USER_DIR}"
cp "${ROOT_DIR}/systemd/ptoas-repo-checker.service" "${SYSTEMD_USER_DIR}/"
cp "${ROOT_DIR}/systemd/ptoas-repo-checker.timer" "${SYSTEMD_USER_DIR}/"

systemctl --user daemon-reload
systemctl --user enable --now ptoas-repo-checker.timer
systemctl --user list-timers ptoas-repo-checker.timer

