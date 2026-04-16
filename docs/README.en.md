# Luduan

[English](README.en.md) | [简体中文](README.zh-CN.md) | [Svenska](README.sv.md) | [Français](README.fr.md)

A macOS audio-to-text dictation tool with better accuracy than the built-in Mac dictation.
Uses [Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Apple Silicon optimized via MLX) for speech recognition and a local [Ollama](https://ollama.com) LLM for post-processing.

## Name origin

The name **Luduan** comes from **甪端**, an auspicious creature in Chinese mythology.
In Mandarin Chinese, **甪端** is pronounced **lù duān** (Pinyin), and for English
speakers you can say it approximately as **/luːˈdwɑːn/**.
It is often described as a one-horned beast associated with discernment, justice,
and peaceful rule, and in some stories it can understand human speech or many
languages. That makes it a fitting name for a multilingual dictation app focused
on listening accurately and turning speech into clear text.

## Features
- **Global hotkey** toggle (default: `Cmd+Shift+Space`) — press to start, press again to stop
- **Streaming transcription** — Whisper processes audio in chunks as you speak
- **LLM post-processing** — Ollama cleans up grammar and punctuation
- **Optional app context** — can read nearby text before the cursor to help with names and acronyms
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
make dmg          # builds dist/Luduan.dmg
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
- Menu bar also lets you pick **Language**, toggle **Use App Context**, and run **Prepare Offline Use**

## Configuration

Config file: `~/.config/luduan/config.toml`

```toml
[hotkey]
keys = ["cmd", "shift", "space"]

[whisper]
model = "base"
language = ""

[ollama]
host = "http://localhost:11434"
model = "gemma3:4b"
enabled = true

[audio]
sample_rate = 16000
channels = 1

[context]
enabled = false
max_chars = 500
```

## Offline preparation

Use the menu bar app to pre-download everything Luduan needs:

**Luduan menu bar icon → Prepare Offline Use**

This will:
- cache the configured Whisper model locally
- ensure the configured Ollama model is present locally

Progress and result are shown in the menu as an **Offline:** status line, and a completion notification appears when it finishes.

After that, Luduan can be used offline as long as those models stay on disk and Ollama itself is already installed on your Mac.
