#!/usr/bin/env zsh
set -euo pipefail
cd "$(dirname "$0")"
lsof -ti :8776 -i :8777 -i :8778 | xargs kill -9 2>/dev/null || true
sleep 0.2
exec "$PWD/.venv/bin/python" "$PWD/web_app.py" "$@"
