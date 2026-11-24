#!/usr/bin/env zsh
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

needs_setup=0
if [ ! -x "$PWD/.venv/bin/python" ]; then
  needs_setup=1
elif ! "$PWD/.venv/bin/python" -c "import sys, psutil, webview, aiohttp; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
  needs_setup=1
fi

if [ "$needs_setup" -eq 1 ]; then
  echo "[ARIES] Setting up standalone Python 3.12 environment..."
  if ! command -v uv >/dev/null 2>&1; then
    echo "[ARIES] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  fi
  rm -rf "$PWD/.venv"
  uv venv --python 3.12 "$PWD/.venv"
  uv pip install --python "$PWD/.venv/bin/python" -r "$PWD/requirements.txt"
fi

lsof -ti :8776 -i :8777 -i :8778 | xargs kill -9 2>/dev/null || true
sleep 0.2
exec "$PWD/.venv/bin/python" "$PWD/web_app.py" "$@"
