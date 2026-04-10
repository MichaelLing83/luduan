# Luduan

A macOS audio-to-text dictation tool with better accuracy than the built-in Mac dictation.
Uses [Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Apple Silicon optimized via MLX) for speech recognition and a local [Ollama](https://ollama.com) LLM for post-processing.

## Features
- **Global hotkey** toggle (default: `Cmd+Shift+Space`) — press to start, press again to stop
- **Streaming transcription** — Whisper processes audio in chunks as you speak
- **LLM post-processing** — Ollama cleans up grammar and punctuation
- **Auto-paste** — transcribed text is pasted directly into the currently focused app
- **Menu bar app** — lives in your macOS menu bar with status icons
- **Fully local** — no cloud, all processing on-device

## Requirements
- macOS (Apple Silicon recommended, M1+)
- [Ollama](https://ollama.com) installed and running
- Python 3.11+

## Installation

### Build as a macOS app (recommended)

Building as a standalone `.app` means you grant permissions to **Luduan** specifically — not to Terminal.

```bash
make app          # builds dist/Luduan.app
make install      # copies to /Applications/Luduan.app
```

After installing:
1. Open **System Settings → Privacy & Security → Input Monitoring**
2. Add **Luduan** and enable it (required for the global hotkey)
3. Launch Luduan from Spotlight or double-click in `/Applications`

### Run directly from terminal (dev mode)

```bash
make dev          # set up .venv
make run          # run without building the app
```

> Note: running from terminal requires granting Input Monitoring to **Terminal** instead of Luduan.

### Hotkey
- **Cmd+Shift+Space** — toggle recording on/off
- Menu bar icon shows current state: idle 🎙 / recording 🔴 / processing ⏳

## Configuration

Config file: `~/.config/luduan/config.toml`

```toml
[hotkey]
keys = ["cmd", "shift", "space"]

[whisper]
model = "base"
language = "en"

[ollama]
host = "http://localhost:11434"
model = "gemma3:4b"
enabled = true

[audio]
sample_rate = 16000
channels = 1
```
