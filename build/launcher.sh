#!/bin/bash
# Luduan launcher — bundled inside Luduan.app/Contents/MacOS/
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/../Resources/venv/bin/python"

if [[ ! -f "$PYTHON" ]]; then
    osascript -e 'display alert "Luduan" message "Python environment not found inside the app bundle. Please run: make app" as critical'
    exit 1
fi

exec "$PYTHON" -m luduan.main
