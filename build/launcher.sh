#!/bin/bash
# Luduan launcher — bundled inside Luduan.app/Contents/MacOS/
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/../Resources/venv/bin/python"

if [[ ! -f "$PYTHON" ]]; then
    osascript -e 'display alert "Luduan" message "Python environment not found inside the app bundle. Please run: make app" as critical'
    exit 1
fi

# Redirect HuggingFace model cache to a local path so it works even when
# external volumes (where ~/.cache/huggingface may be symlinked) are not mounted.
export HF_HOME="$HOME/.config/luduan/models"
mkdir -p "$HF_HOME"

exec "$PYTHON" -m luduan.main
