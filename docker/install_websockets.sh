#!/bin/sh
set -e

# Install websockets — try uv first, then pip as fallback
PYTHON=/app/.venv/bin/python

if "$PYTHON" -c "import websockets" 2>/dev/null; then
    echo "[entrypoint] websockets already installed"
else
    echo "[entrypoint] websockets not found — installing..."

    if [ -x /usr/local/bin/uv ]; then
        UV_OFFLINE=0 /usr/local/bin/uv pip install websockets --python "$PYTHON"
    elif [ -x /root/.local/bin/uv ]; then
        UV_OFFLINE=0 /root/.local/bin/uv pip install websockets --python "$PYTHON"
    elif [ -x /home/app/.local/bin/uv ]; then
        UV_OFFLINE=0 /home/app/.local/bin/uv pip install websockets --python "$PYTHON"
    else
        "$PYTHON" -m ensurepip --upgrade
        "$PYTHON" -m pip install websockets
    fi

    echo "[entrypoint] websockets installed"
fi
