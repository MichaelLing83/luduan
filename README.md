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

```bash
# Install in a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
luduan
```

On first run, Luduan will guide you through microphone permissions and Ollama model selection.

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
